# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from ..utils import CosineEMA
from .base import BaseSelfSupervisor


@MODELS.register_module()
class MemReconBYOL(BaseSelfSupervisor):
    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 base_momentum: float = 0.004,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # create momentum model
        # self.target_net = CosineEMA(
        #     nn.Sequential(self.backbone, self.neck), momentum=base_momentum)
        self.target_backbone = CosineEMA(self.backbone, momentum=base_momentum)
        self.target_neck = CosineEMA(self.neck, momentum=base_momentum)

    def loss(self, inputs: List[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        assert isinstance(inputs, list)
        img_v1 = inputs[0]
        img_v2 = inputs[1]
        # compute online features
        y_src_v1 = self.backbone(img_v1)
        y_src_v2 = self.backbone(img_v2)
        proj_online_v1 = self.neck(y_src_v1)[0]
        proj_online_v2 = self.neck(y_src_v2)[0]
        # compute target features
        y_tgt_v1, y_tgt_v2, = None, None
        with torch.no_grad():
            # update the target net
            # self.target_net.update_parameters(
            #     nn.Sequential(self.backbone, self.neck))
            self.target_backbone.update_parameters(
                self.backbone)
            self.target_neck.update_parameters(
                self.neck)

            # proj_target_v1 = self.target_net(img_v1)[0]
            # proj_target_v2 = self.target_net(img_v2)[0]
            y_tgt_v1 = self.target_backbone(img_v1)
            y_tgt_v2 = self.target_backbone(img_v2)
            proj_target_v1 = self.target_neck(y_tgt_v1)[0]
            proj_target_v2 = self.target_neck(y_tgt_v2)[0]

        infos1 = [proj_target_v2, y_src_v1[0], y_tgt_v2[0]]
        infos2 = [proj_target_v1, y_src_v2[0], y_tgt_v1[0]]
        loss_1 = self.head.loss(proj_online_v1, infos1)
        loss_2 = self.head.loss(proj_online_v2, infos2)

        losses = dict(loss=2. * (loss_1 + loss_2))
        return losses
