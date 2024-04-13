# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .base import BaseSelfSupervisor

from .diffusion_predictor import DiffusionPredictor
from .decoder import Decoder
from .vision_transformer import TransformerPredictor

@MODELS.register_module()
class DiffSimSiam(BaseSelfSupervisor):
    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 target_generator: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None,
                 
                 loss_original: float = 1,
                 decoder_cfg: dict = None,
                 crop_cfg: dict = None,
                 diff_pred: bool = False,
                 diff_cfg: Optional[Union[List[dict], dict]] = None,
                 loss_add_cos: float = 0,                 
                 ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=None,  # change for head with TransformerPredictor
            target_generator=target_generator,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        # additional part
        self.loss_add_cos = loss_add_cos
        self.loss_original = loss_original
        if self.loss_add_cos or self.loss_original:
            if head.type == 'LatentPredictHead':
                self.predictor = MODELS.build(head.predictor)
            elif head.type == 'TransformerPredictor':
                self.predictor = TransformerPredictor(
                    embed_dim=head.predictor.in_channels,
                    predictor_embed_dim=head.predictor.predictor_embed_dim,
                    depth=head.predictor.depth,
                    num_heads=head.predictor.num_attention_heads,
                    cat_or_add='add'
                )
            self.loss_module = MODELS.build(head.loss)
        
        self.decoder_cfg = decoder_cfg
        if self.decoder_cfg is not None:
            self.decoder = Decoder(img_size=decoder_cfg.img_size, decoder_layer=decoder_cfg.decoder_layer) if decoder_cfg.img_size != 0 else None
                    
        self.crop_cfg = dict(
            num_views=[2, 0],    # default:[2, 0]. If num_views is not [2, 0], then train_pipeline need to be changed.
            pred_map=[1, 0, 0, 0],
        ) if crop_cfg is None else crop_cfg
        self.diff_pred = diff_pred
        if self.diff_pred:
            self.DMmodel = DiffusionPredictor(diff_cfg)

    def loss(self, inputs: List[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        # 原来代码：
        # assert isinstance(inputs, list)
        # img_v1 = inputs[0]
        # img_v2 = inputs[1]

        # z1 = self.neck(self.backbone(img_v1))[0]  # NxC
        # z2 = self.neck(self.backbone(img_v2))[0]  # NxC

        # loss_1 = self.head.loss(z1, z2)
        # loss_2 = self.head.loss(z2, z1)

        # losses = dict(loss=0.5 * (loss_1 + loss_2))
        
        losses = dict()
        views = self._distribute_views(inputs)
        feat_maps, feat_projs = [], []
        for idx, view in enumerate(views):
            if isinstance(view, torch.Tensor):
                feat_map = self.backbone(view)
                feat_maps.append(feat_map[0])
                feat_projs.append(self.neck(feat_map)[0])
            else:
                feat_maps.append(view)
                feat_projs.append(view)
        
        projs_target_online = self._match_target_online(feat_projs, detach=True)
        num_pairs = len(projs_target_online)
        
        if self.loss_original > 0:
            assert self.predictor is not None
            losses['loss_original'] = 0
            for proj_target, proj_online in projs_target_online:
                pred = self.predictor([proj_online])
                pred = pred[0] if isinstance(pred, list) or isinstance(pred, tuple) else pred
                target = proj_target.detach()
                losses['loss_original'] += 2. * self.loss_module(pred, target) * self.loss_original
            losses['loss_original'] /= (num_pairs/2)
            
        if self.loss_add_cos > 0:
            losses['loss_add_cos'] = 0
            for proj_target, proj_online in projs_target_online:
                pred_part1 = self.predictor([proj_online])
                pred_part1 = pred_part1[0] if isinstance(pred_part1, list) or isinstance(pred, tuple) else pred_part1
                target = proj_target.detach()
                pred_part2 = self.DMmodel([proj_target, proj_online], output_pred=True)[0]
                losses['loss_add_cos'] += 2. * self.loss_module(pred_part1 + pred_part2, target) * self.loss_add_cos
            losses['loss_add_cos'] /= (num_pairs/2)
            
        if self.diff_pred:
            diff_losses = self.DMmodel(projs_target_online)
            losses.update(diff_losses)
        
        if self.decoder_cfg is not None:
            global_num, local_num = self.crop_cfg.num_views
            if self.decoder_cfg.decoder_layer == "neck":
                decoder_feat_view_zip = zip(feat_projs[:global_num], views[:global_num])
            elif self.decoder_cfg.decoder_layer == "backbone":
                decoder_feat_view_zip = zip(feat_maps[:global_num], views[:global_num])
            if self.decoder is not None:
                losses['loss_decoder'] = 0
                for proj, img in decoder_feat_view_zip:
                    proj = proj.detach()
                    img = img.detach()
                    image_online = self.decoder(proj_online)
                    losses['loss_decoder'] += F.mse_loss(image_online.float(), img.float(), reduction="mean")
        return losses
        
            
    def _distribute_views(self, inputs):
        views = inputs[:]
        global_num, local_num = self.crop_cfg.num_views
        pred_map = self.crop_cfg.pred_map   # [GpG, GpL, LpG, LpL]
        if not pred_map[0] and not pred_map[1] and not pred_map[2]:     # [0,0,0,x] only local
            views[:global_num] = [False] * global_num
        elif not pred_map[1] and not pred_map[2] and not pred_map[3]:   # [x,0,0,0] only global
            views[global_num:] = [False] * local_num
        return views
    
    
    def _match_target_online(self, projs, detach=False):
        projs_target, projs_online = projs[:], projs[:]
        if detach:
            projs_target = [proj.detach() for proj in projs_target]
        pred_map = self.crop_cfg.pred_map
        global_num, local_num = self.crop_cfg.num_views
        GpG = make_pairs(projs_target[:global_num], projs_online[:global_num], True) if pred_map[0] else []
        GpL = make_pairs(projs_target[global_num:], projs_online[:global_num], False) if pred_map[1] else []
        LpG = make_pairs(projs_target[:global_num], projs_online[global_num:], False) if pred_map[2] else []
        LpL = make_pairs(projs_target[global_num:], projs_online[global_num:], True) if pred_map[3] else []
        projs_pairs = GpG + GpL + LpG + LpL
        return projs_pairs
    
    
def make_pairs(list1, list2, noSamePos=True):
    pairs = []
    for idx1, item1 in enumerate(list1):
        for idx2, item2 in enumerate(list2):
            if noSamePos:
                if idx1 == idx2:  # 如果下标相同，则跳过
                    continue
            pairs.append([item1, item2])
    return pairs    

