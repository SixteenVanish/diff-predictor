# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmengine.config import ConfigDict

from ..utils import CosineEMA
from .base import BaseSelfSupervisor

from .diffusion_predictor import DiffusionPredictor
from .decoder import Decoder
from .vision_transformer import TransformerPredictor

@MODELS.register_module()
class DiffBYOL(BaseSelfSupervisor):
    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 base_momentum: float = 0.004,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None,
                 
                 loss_byol_cos: float = 1,
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
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # create momentum model
        self.target_net = CosineEMA(
            nn.Sequential(self.backbone, self.neck), momentum=base_momentum)
     
        # additional part
        self.loss_add_cos = loss_add_cos
        self.loss_byol_cos = loss_byol_cos
        if self.loss_add_cos or self.loss_byol_cos:
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
            self.decoder = Decoder(img_size=decoder_cfg.img_size, cfg=decoder_cfg) if decoder_cfg.img_size != 0 else None

        self.crop_cfg = dict(
            num_views=[2, 0],
            pred_map=[1, 0, 0, 0],
        ) if crop_cfg is None else crop_cfg
        
        self.diff_pred = diff_pred
        if self.diff_pred:
            self.DMmodel = DiffusionPredictor(diff_cfg)
        
    def loss(self, inputs: List[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        assert isinstance(inputs, list)
        # 原来代码：
        # img_v1 = inputs[0]
        # img_v2 = inputs[1]
        # # compute online features
        # proj_online_v1 = self.neck(self.backbone(img_v1))[0]
        # proj_online_v2 = self.neck(self.backbone(img_v2))[0]
        # # compute target features
        # with torch.no_grad():
        #     # update the target net
        #     self.target_net.update_parameters(
        #         nn.Sequential(self.backbone, self.neck))

        #     proj_target_v1 = self.target_net(img_v1)[0]
        #     proj_target_v2 = self.target_net(img_v2)[0]

        # loss_1 = self.head.loss(proj_online_v1, proj_target_v2)
        # loss_2 = self.head.loss(proj_online_v2, proj_target_v1)

        # losses = dict(loss=2. * (loss_1 + loss_2))
        
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
            self.target_net.update_parameters(
                nn.Sequential(self.backbone, self.neck))
            for idx, view in enumerate(views_target):
                if isinstance(view, torch.Tensor):
                    maps_target.append(self.target_net.module[0](view)[0])
                    projs_target.append(self.target_net(view)[0])
                else:
                    maps_target.append(view)
                    projs_target.append(view)
        
        projs_target_online = self._match_target_online(projs_target, projs_online)
        num_pairs = len(projs_target_online)
        
        if self.loss_byol_cos > 0:
            assert self.predictor is not None
            losses['loss_byol_cos'] = 0
            byol_output = {
                'middle_embedding_list': [],
                'byol_pred_target_list': []
            }
            for proj_target, proj_online in projs_target_online:
                pred = self.predictor([proj_online])
                pred = pred[0] if isinstance(pred, list) or isinstance(pred, tuple) else pred
                byol_output['byol_pred_target_list'].append(pred)
                target = proj_target.detach()
                losses['loss_byol_cos'] += 2. * self.loss_module(pred, target) * self.loss_byol_cos
            losses['loss_byol_cos'] /= (num_pairs/2)
            
        if self.loss_add_cos > 0:
            losses['loss_add_cos'] = 0
            for proj_target, proj_online in projs_target_online:
                pred_part1 = self.predictor([proj_online])
                pred_part1 = pred_part1[0] if isinstance(pred_part1, list) else pred_part1
                byol_output['byol_pred_target_list'].append(pred_part1)
                target = proj_target.detach()
                pred_part2 = self.DMmodel([[proj_target, proj_online]])['diff_pred_cal_list'][0]
                losses['loss_add_cos'] += 2. * self.loss_module(pred_part1 + pred_part2, target) * self.loss_add_cos
            losses['loss_add_cos'] /= (num_pairs/2)
            
        if self.diff_pred:
            diff_output = self.DMmodel(projs_target_online)
            diff_losses = diff_output['losses']
            losses.update(diff_losses)
        
        if self.decoder_cfg is not None and self.decoder is not None:
            global_num, local_num = self.crop_cfg.num_views
            if self.decoder_cfg.decoder_layer == "online_neck":
                codes = projs_online
                imgs = views_online
            elif self.decoder_cfg.decoder_layer == "target_neck":
                codes = projs_target
                imgs = views_target
            elif self.decoder_cfg.decoder_layer == "online_backbone":
                codes = maps_online
                imgs = views_online
            elif self.decoder_cfg.decoder_layer == "target_backbone":
                codes = maps_target
                imgs = views_target
            elif self.decoder_cfg.decoder_layer == "predictor_mid": # TODO byol_output
                codes = diff_output['middle_embedding_list'] if self.diff_pred else byol_output['middle_embedding_list']
                imgs = views_online
            elif self.decoder_cfg.decoder_layer == "predictor":
                if self.diff_pred:
                    if len(diff_output['diff_pred_infer_list'])>0 :   # 优先使用infer所得特征
                        codes = diff_output['diff_pred_infer_list']
                    else:
                        codes = diff_output['diff_pred_cal_list']
                else:
                    codes = byol_output['byol_pred_target_list']
                imgs = views_online
            decoder_code_img_zip = zip(codes[:global_num], imgs[:global_num])
            
            losses['loss_decoder'] = 0
            for code, img in decoder_code_img_zip:
                if self.decoder_cfg.detach:
                    code = code.detach()
                    img = img.detach()
                else:  
                    code = code
                    img = img
                image_decoded = self.decoder(code)
                losses['loss_decoder'] += F.mse_loss(image_decoded.float(), img.float(), reduction="mean") * self.decoder_cfg.loss_decoder
            
        return losses


    def extract_feat(self, inputs: List[torch.Tensor], stage='backbone', DMtimes=5, DMstep=0):
        inputs = inputs[0]
        outs = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(device)
        self.backbone.to(device)
        if stage == 'backbone':
            outs = self.backbone(inputs)
        elif stage == 'neck':
            outs = self.neck(self.backbone(inputs))
        elif stage == 'target_backbone':
            outs = self.target_net.module[0](inputs)
        elif stage == 'target_neck':
            # feats = self.target_net.module[0](inputs)
            outs = self.target_net(inputs)
        elif stage == 'predictor_mid':
            target_feat = self.target_net(inputs)[0]
            online_feat = self.neck(self.backbone(inputs))[0]
            if self.diff_pred:
                latent_condition_pairs = [[target_feat, online_feat]]
                output = self.DMmodel(latent_condition_pairs)
                outs = output['middle_embedding_list']
            else:    # TODO
                outs = self.predictor(outs)
        elif stage == 'predictor':
            target_feat = self.target_net(inputs)[0]
            online_feat = self.neck(self.backbone(inputs))[0]
            if self.diff_pred:
                latent_condition_pairs = [[target_feat, online_feat]]
                output = self.DMmodel(latent_condition_pairs)
                outs = output['diff_pred_cal_list']
            else:
                outs = self.predictor([online_feat])
        elif stage == 'DM' and DMtimes!=0:
            feats = self.backbone(inputs)
            condition_feat = self.neck(feats)[0]
            self.scheduler.set_timesteps(self.diff_cfg.scheduler.num_train_timesteps)
            bsz, dim = condition_feat.shape
            for i in range(DMtimes):  # 对所有数据进行DMtimes次denoise loop
                latent = torch.randn(condition_feat.shape, device=condition_feat.device, dtype=condition_feat.dtype)
                for t in self.scheduler.timesteps:  # denoise loop
                    t_ = t.view(-1).to(condition_feat.device)
                    model_output = self.DMmodel(latent, condition_feat, t_.repeat(bsz), self.diff_cfg.cat_or_add, self.diff_cfg.remove_condition_T)
                    latent = self.denoise_step(model_output, t, latent)
                    if t == DMstep:
                        break
                outs.append(latent)
                # feats = torch.stack(feat_list)  # [DMtimes, bsz, dim]
                # feats = feats.permute(1, 0, 2).contiguous()  # [bsz, DMtimes, dim]
            outs = tuple(outs)
        
        return outs
    
    
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
