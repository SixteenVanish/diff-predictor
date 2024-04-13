# Copyright (c) OpenMMLab. All rights reserved.
import math
from functools import reduce
from operator import mul
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmengine.dist import all_gather, get_rank
from mmpretrain.models.backbones import VisionTransformer
from mmpretrain.models.utils import (build_2d_sincos_position_embedding,
                                     to_2tuple)
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from ..utils import CosineEMA
from .base import BaseSelfSupervisor

from .diffusion_predictor import DiffusionPredictor
from .decoder import Decoder
from .vision_transformer import TransformerPredictor

@MODELS.register_module()
class DiffMoCoV3(BaseSelfSupervisor):
    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 base_momentum: float = 0.01,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None,
                 
                 loss_original: float = 1,
                 loss_diff_CE: float = 0,  
                 decoder_cfg: dict = None,
                 crop_cfg: dict = None,
                 diff_pred: bool = False,
                 diff_cfg: Optional[Union[List[dict], dict]] = None,
                 ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=None,  # change for head with TransformerPredictor
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # create momentum model
        self.momentum_encoder = CosineEMA(
            nn.Sequential(self.backbone, self.neck), momentum=base_momentum)

        # additional part
        self.loss_original = loss_original
        self.loss_diff_CE = loss_diff_CE
        self.temperature = head.temperature
        self.loss_module = MODELS.build(head.loss)
        if self.loss_original:
            if head.type == 'MoCoV3Head':
                self.predictor = MODELS.build(head.predictor)
            elif head.type == 'TransformerPredictor':
                self.predictor = TransformerPredictor(
                    embed_dim=head.predictor.in_channels,
                    predictor_embed_dim=head.predictor.predictor_embed_dim,
                    depth=head.predictor.depth,
                    num_heads=head.predictor.num_attention_heads,
                    cat_or_add='add'
                )
            
        self.decoder_cfg = decoder_cfg
        if self.decoder_cfg is not None:
            self.decoder_online = Decoder(img_size=decoder_cfg.online_img_size, decoder_layer=decoder_cfg.decoder_layer) if decoder_cfg.online_img_size != 0 else None
            self.decoder_target = Decoder(img_size=decoder_cfg.target_img_size, decoder_layer=decoder_cfg.decoder_layer) if decoder_cfg.target_img_size != 0 else None
                    
        self.crop_cfg = dict(
            num_views=[2, 0],    # default:[2, 0]. If num_views is not [2, 0], then train_pipeline need to be changed.
            pred_map=[1, 0, 0, 0],
        ) if crop_cfg is None else crop_cfg
        self.diff_pred = diff_pred
        self.diff_cfg = diff_cfg
        if self.diff_pred:
            self.DMmodel = DiffusionPredictor(diff_cfg)
        
    def loss(self, inputs: List[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        # 原来代码
        # assert isinstance(inputs, list)
        # view_1 = inputs[0]
        # view_2 = inputs[1]

        # # compute query features, [N, C] each
        # q1 = self.neck(self.backbone(view_1))[0]
        # q2 = self.neck(self.backbone(view_2))[0]

        # # compute key features, [N, C] each, no gradient
        # with torch.no_grad():
        #     # update momentum encoder
        #     self.momentum_encoder.update_parameters(
        #         nn.Sequential(self.backbone, self.neck))

        #     k1 = self.momentum_encoder(view_1)[0]
        #     k2 = self.momentum_encoder(view_2)[0]

        # loss = self.head.loss(q1, k2) + self.head.loss(q2, k1)

        # losses = dict(loss=loss)
        
        losses = dict()
        views_online, views_target = self._distribute_views(inputs)
        maps_online, maps_target, projs_target, projs_online = [], [], [], []
        for idx, view in enumerate(views_online):
            if isinstance(view, torch.Tensor):
                map_online = self.backbone(view)
                maps_online.append(map_online[0])
                projs_online.append(self.neck(map_online)[0])
            else:
                maps_online.append(view)
                projs_online.append(view)
        with torch.no_grad():
            self.momentum_encoder.update_parameters(
                nn.Sequential(self.backbone, self.neck))
            for idx, view in enumerate(views_target):
                if isinstance(view, torch.Tensor):
                    maps_target.append(self.momentum_encoder.module[0](view)[0])
                    projs_target.append(self.momentum_encoder(view)[0])
                else:
                    maps_target.append(view)
                    projs_target.append(view)
        
        projs_target_online = self._match_target_online(projs_target, projs_online)
        num_pairs = len(projs_target_online)
        
        def compute_loss_diff_CE(pred, target):
            # normalize
            pred = nn.functional.normalize(pred, dim=1)
            target = nn.functional.normalize(target, dim=1)

            # get negative samples
            target = torch.cat(all_gather(target), dim=0)

            # Einstein sum is more intuitive
            logits = torch.einsum('nc,mc->nm', [pred, target]) / self.temperature

            # generate labels
            batch_size = logits.shape[0]
            labels = (torch.arange(batch_size, dtype=torch.long) +
                    batch_size * get_rank()).to(logits.device)

            return self.loss_module(logits, labels)
                
        if self.loss_original > 0:
            assert self.predictor is not None
            losses['loss_original'] = 0
            for proj_target, proj_online in projs_target_online:
                # predictor computation
                pred = self.predictor([proj_online])[0]
                losses['loss_original'] += compute_loss_diff_CE(pred, proj_target) * self.loss_original
            losses['loss_original'] /= (num_pairs/2)
            
        if self.diff_pred:
            assert self.DMmodel is not None
            diff_output = self.DMmodel(projs_target_online)
            pred_list, diff_losses = diff_output['diff_pred_target_list'], diff_output['losses']
            
            if self.loss_diff_CE > 0:
                losses['loss_diff_CE'] = 0
                for (proj_target, proj_online), pred in zip(projs_target_online, pred_list):
                    losses['loss_diff_CE'] += compute_loss_diff_CE(pred, proj_target) * self.loss_diff_CE
                losses['loss_diff_CE'] /= (num_pairs/2)
                
            if self.diff_cfg is not None:
                for key, value in self.diff_cfg.loss.items():
                    if value > 0:
                        losses.update(diff_losses)
                        break
                
        return losses

    def _distribute_views(self, inputs):
        views_online, views_target = inputs[:], inputs[:]
        global_num, local_num = self.crop_cfg.num_views
        pred_map = self.crop_cfg.pred_map   # [GpG, GpL, LpG, LpL]
        if not pred_map[0] and not pred_map[1]:     # [0,0,x,x]
            views_online[:global_num] = [False] * global_num
        elif not pred_map[2] and not pred_map[3]:   # [x,x,0,0]
            views_online[global_num:] = [False] * local_num
        if not pred_map[0] and not pred_map[2]:     # [0,x,0,x]
            views_target[:global_num] = [False] * global_num
        elif not pred_map[1] and not pred_map[3]:   # [x,0,x,0]
            views_target[global_num:] = [False] * local_num
        return views_online, views_target
    
    
    def _match_target_online(self, projs_target, projs_online):
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
