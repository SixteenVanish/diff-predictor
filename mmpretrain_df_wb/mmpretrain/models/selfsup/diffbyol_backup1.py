# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from copy import deepcopy

import pdb
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSelfSupervisor
from ..utils import CosineEMA

from pytorchvideo.layers.batch_norm import NaiveSyncBatchNorm1d
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmengine.config import ConfigDict

from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.attention import BasicTransformerBlock

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
                 
                 loss_byol_cos: float = 0,
                 loss_byol_mse: float = 0,
                 loss_add_cos: float = 0,
                 img_size: int = 64,
                 test_cfg: Optional[ConfigDict] = None,

                 num_crops: Union[int, List[int]] = 2,   
                 OLcrops: bool = False,
                 noGpL: bool = False,
                 enable_pred_map: bool = False,
                 pred_map: List[int] = [0,1,0,0],
                 
                 diff_pred: bool = True,
                 diff_cfg: Optional[ConfigDict] = None,

                 decoder_layer: str = "neck",
                 load_test: int = 0,
                 ) -> None:

        self.iter_count = 0

        # byol初始结构的初始化
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=None,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        # create momentum model
        self.base_momentum = base_momentum
        self.target_net = CosineEMA(
            nn.Sequential(self.backbone, self.neck), momentum=base_momentum) if self.base_momentum \
        else deepcopy(nn.Sequential(self.backbone, self.neck)).requires_grad_(False)
        
        self.loss_byol_cos = loss_byol_cos
        self.loss_byol_mse = loss_byol_mse
        self.loss_add_cos = loss_add_cos
        if self.loss_byol_cos > 0 or self.loss_byol_mse > 0 or self.loss_add_cos > 0:
            self.head_type = head.type
            self.loss_module = MODELS.build(head.loss)
            if self.head_type == 'LatentPredictHead':
                self.predictor = MODELS.build(head.predictor)
            else:
                self.predictor = Transformer(
                    dim_in=head.predictor.in_channels,
                    dim_out=head.predictor.out_channels,
                    num_layers=head.predictor.num_layers,
                    num_attention_heads=head.predictor.num_attention_heads,
                    attention_head_dim=head.predictor.attention_head_dim,
                    
                    cat_or_add=head.predictor.cat_or_add,  
                    remove_condition_T=head.predictor.remove_condition_T,
                    
                    cross_attention_dim = head.predictor.cross_attention_dim,
                    only_cross_attention = head.predictor.only_cross_attention,
                    
                    proj_in=head.predictor.proj_in,
                    norm_out=head.predictor.norm_out,
                    proj_to_clip_embeddings=head.predictor.proj_to_clip_embeddings,
                )
        
        self.img_size = img_size
        if img_size!=448:
            self.decoder_layer = decoder_layer
            self.decoder_online = Decoder(img_size=img_size, decoder_layer=decoder_layer)
            self.decoder_target = Decoder(img_size=img_size, decoder_layer=decoder_layer)
        
        self.test_cfg = test_cfg
        self.num_crops = num_crops
        self.multi_crops = False if num_crops==2 or num_crops == [1,1] else True # todo 待验证
        self.OLcrops = OLcrops
        self.noGpL = noGpL
        self.enable_pred_map = enable_pred_map
        self.pred_map = pred_map
        if self.noGpL:
            assert self.OLcrops is True
        
        self.diff_pred = diff_pred
        self.load_test = load_test
        if self.diff_pred:
            self.diff_cfg = ConfigDict(
            ) if diff_cfg is None else diff_cfg
                
            if self.diff_cfg.model.type == 'MLP':
                self.DMmodel = MLP(
                    dim_in=self.diff_cfg.model.in_channels,
                    dim_out=self.diff_cfg.model.out_channels,
                    num_layers=self.diff_cfg.model.num_layers,
                    
                    mlp_dim=self.diff_cfg.model.mlp_params.mlp_dim,
                    bn_on=self.diff_cfg.model.mlp_params.bn_on,
                    bn_sync_num=self.diff_cfg.model.mlp_params.bn_sync_num if self.diff_cfg.model.mlp_params.bn_sync_num else 1,
                    global_sync=(self.diff_cfg.model.mlp_params.bn_sync_num and self.diff_cfg.model.mlp_params.global_sync),
                
                    cat_or_add=self.diff_cfg.cat_or_add,  
                    remove_condition_T=self.diff_cfg.remove_condition_T,
                ) 
            else:
                self.DMmodel = Transformer(
                    dim_in=self.diff_cfg.model.in_channels,
                    dim_out=self.diff_cfg.model.out_channels,
                    num_layers=self.diff_cfg.model.num_layers,
                    num_attention_heads=self.diff_cfg.model.trans_params.num_attention_heads,
                    attention_head_dim=self.diff_cfg.model.trans_params.attention_head_dim,
                    
                    cat_or_add=self.diff_cfg.cat_or_add,  
                    remove_condition_T=self.diff_cfg.remove_condition_T,
                    
                    cross_attention_dim = self.diff_cfg.model.trans_params.cross_attention_dim,
                    only_cross_attention = self.diff_cfg.model.trans_params.only_cross_attention,
                    
                    proj_in=self.diff_cfg.model.trans_params.proj_in,
                    norm_out=self.diff_cfg.model.trans_params.norm_out,
                    proj_to_clip_embeddings=self.diff_cfg.model.trans_params.proj_to_clip_embeddings,

                )
            self.scheduler = DDPMScheduler(num_train_timesteps=self.diff_cfg.scheduler.num_train_timesteps,
                                           clip_sample=False,
                                           beta_schedule="linear",
                                           prediction_type=self.diff_cfg.scheduler.prediction_type,
                                           )
            self.cosineSimilarityFunc = MODELS.build(dict(type='CosineSimilarityLoss'))

    def initialization(self, losses):
        if self.loss_byol_cos > 0:
            assert self.loss_module is not None
            losses['loss_byol_cos'] = 0
        if self.loss_byol_mse > 0:
            losses['loss_byol_mse'] = 0
        if self.loss_add_cos > 0:
            losses['loss_add_cos'] = 0
        
        if self.diff_pred:
            if self.diff_cfg.loss.loss_noise_mse > 0:
                losses['loss_noise_mse'] = 0
            if self.diff_cfg.loss.loss_noise_cos > 0:
                losses['loss_noise_cos'] = 0
            if self.diff_cfg.loss.loss_x0_cos > 0:
                losses['loss_x0_cos'] = 0
            if self.diff_cfg.loss.loss_x0_mse > 0:
                losses['loss_x0_mse'] = 0
            if self.diff_cfg.loss.loss_align_weight > 0:
                losses['loss_align'] = 0
        
        return losses

    def loss(self, inputs: List[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        self.iter_count += 1
        
        losses = dict()
        self.initialization(losses)
        
        assert isinstance(inputs, list)
        views_target = inputs
        if self.OLcrops:
            views_online = inputs
        else:
            views_online = inputs[:2]
        maps_online = []
        maps_target = []
        projs_target = []
        projs_online = []
        for idx, view in enumerate(views_online):
            map_online = self.backbone(view)
            maps_online.append(map_online[0])
            projs_online.append(self.neck(map_online)[0])
        with torch.no_grad():
            if self.base_momentum > 0 :
                self.target_net.update_parameters(
                    nn.Sequential(self.backbone, self.neck))
            for idx, view in enumerate(views_target):
                maps_target.append(self.target_net.module[0](view)[0])
                projs_target.append(self.target_net(view)[0])
                
        if self.noGpL: 
            projs_pairs = make_pairs(projs_target[:2], projs_online[:2]) + \
                          make_pairs(projs_target[2:], projs_online[2:]) + \
                          make_pairs(projs_target[:2], projs_online[2:], False)
        elif self.enable_pred_map:
            GpG = make_pairs(projs_target[:2], projs_online[:2], True) if self.pred_map[0] else []
            GpL = make_pairs(projs_target[:2], projs_online[2:], False) if self.pred_map[1] else []
            LpG = make_pairs(projs_target[2:], projs_online[:2], False) if self.pred_map[2] else []
            LpL = make_pairs(projs_target[2:], projs_online[2:], True) if self.pred_map[3] else []
            projs_pairs = GpG + GpL + LpG + LpL
        else:   # default
            projs_pairs = make_pairs(projs_target, projs_online)
            
        if self.load_test == 1:
            # pretrain
            self.diff_pred = False
            self.diff_cfg.loss.loss_x0_cos = 0
            losses["loss_x0_cos"] = torch.zeros((1,), device='cuda:0')
        elif self.load_test == 2:
            # load
            self.diff_pred = True
            self.loss_byol_cos = 0
            losses["loss_byol_cos"] = torch.zeros((1,), device='cuda:0')
        
        if self.loss_byol_cos > 0 or self.loss_byol_mse > 0 or self.loss_add_cos > 0:
            assert self.predictor is not None                  
            for proj_target, proj_online in projs_pairs:
                if self.head_type == 'LatentPredictHead':
                    pred = self.predictor([proj_online])[0]
                else:
                    pred = self.predictor(noisy_latents=proj_online, condition_feat=torch.zeros((proj_online.shape), device=proj_online.device), 
                                          time_emb=torch.zeros((proj_online.shape), device=proj_online.device))
                target = proj_target.detach()
                if self.loss_byol_cos > 0:
                    losses['loss_byol_cos'] += self.loss_module(pred, target)
                if self.loss_byol_mse > 0:
                    losses['loss_byol_mse'] += F.mse_loss(pred.float(), target.float(), reduction="mean")
                if self.loss_add_cos:
                    pred_part1 = pred
                    
                    latents = proj_target
                    condition_feat = proj_online
                    bsz = latents.shape[0]
                    timesteps = torch.randint(low=0,
                                            high=self.scheduler.config.num_train_timesteps,
                                            size=(bsz,),
                                            device=latents.device).long()
                    noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype)
                    noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                    time_emb = get_timestep_embedding(timesteps, noisy_latents.shape[1])
                    predicted_embedding = self.DMmodel(noisy_latents, condition_feat, time_emb)
                    df_pred_tgt = self.pred_original_sample(predicted_embedding, timesteps, noisy_latents)
                    pred_part2 = df_pred_tgt
                    
                    losses['loss_add_cos'] += self.loss_module(pred_part1+pred_part2, target)
            if self.loss_byol_cos > 0:
                losses['loss_byol_cos'] = 2. * losses['loss_byol_cos'] * self.loss_byol_cos
            if self.loss_byol_mse > 0:
                losses['loss_byol_mse'] = 2. * losses['loss_byol_mse'] * self.loss_byol_mse
            if self.loss_add_cos > 0:
                losses['loss_add_cos'] = 2. * losses['loss_add_cos'] * self.loss_add_cos
            
            
        if self.diff_pred and not self.loss_add_cos:
            if not self.diff_cfg.conditioned_on_tgt:    # default
                latent_condition = projs_pairs  # pairs of projs_target and projs_online
            else:
                if self.noGpL:
                    latent_condition = make_pairs(projs_online[:2], projs_target[:2], True) + \
                                       make_pairs(projs_online[2:], projs_target[2:], True) + \
                                       make_pairs(projs_online[:2], projs_target[2:], False)
                elif self.enable_pred_map:
                    GpG = make_pairs(projs_online[:2], projs_target[:2], True) if self.pred_map[0] else []
                    GpL = make_pairs(projs_online[:2], projs_target[2:], False) if self.pred_map[1] else []
                    LpG = make_pairs(projs_online[2:], projs_target[:2], False) if self.pred_map[2] else []
                    LpL = make_pairs(projs_online[2:], projs_target[2:], True) if self.pred_map[3] else []
                    projs_pairs = GpG + GpL + LpG + LpL
                else:   # default
                    latent_condition = make_pairs(projs_online, projs_target)
            for latents, condition_feat in latent_condition:
                if self.diff_cfg.pred_residual:
                    latents = latents - condition_feat
                bsz = latents.shape[0]
                timesteps = torch.randint(low=0,
                                          high=self.scheduler.config.num_train_timesteps,
                                          size=(bsz,),
                                          device=latents.device).long()
                noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype)
                if self.diff_cfg.diff_prob < 1:  # 令部分noise为0，其对应的timestep为-1 todo 代码正确性待检查
                    num_ones = int(self.diff_cfg.diff_prob * bsz)

                    mask = torch.zeros(bsz, device=latents.device)
                    mask[:num_ones] = 1
                    # mask = mask[torch.randperm(bsz)]
                    scaler_mask = mask.view(-1, 1)
                    noise = noise * scaler_mask

                    mask_off = mask - 1
                    timesteps = timesteps * mask + mask_off
                    timesteps = timesteps.long()

                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                time_emb = get_timestep_embedding(timesteps, noisy_latents.shape[1])
                predicted_embedding = self.DMmodel(noisy_latents, condition_feat, time_emb)

                if self.scheduler.config.prediction_type == "epsilon":  # 预测噪声noise default
                    diff_target = noise
                elif self.scheduler.config.prediction_type == "sample":  # 预测特征x0
                    diff_target = latents

                if self.diff_cfg.loss.loss_noise_mse > 0:
                    loss_noise_mse = F.mse_loss(predicted_embedding.float(), diff_target.float(), reduction="mean")
                    if loss_noise_mse.isnan():
                        losses['loss_noise_mse'] += 0 * loss_noise_mse
                    else:
                        losses['loss_noise_mse'] += self.diff_cfg.loss.loss_noise_mse * loss_noise_mse
                if self.diff_cfg.loss.loss_noise_cos > 0:
                    loss_noise_cos = self.cosineSimilarityFunc(predicted_embedding.float(), diff_target.float())
                    losses['loss_noise_cos'] += 2. * loss_noise_cos * self.diff_cfg.loss.loss_noise_cos
                
                df_pred_tgt = self.pred_original_sample(predicted_embedding, timesteps, noisy_latents)
                cosine_similarity = self.cosineSimilarityFunc(latents, df_pred_tgt.to(latents.device))
                # 若self.scheduler.config.prediction_type == "sample"
                # 则df_pred_tgt == predicted_embedding， latents == diff_target
                if self.diff_cfg.loss.loss_x0_cos > 0:
                    losses['loss_x0_cos'] += 2. * cosine_similarity * self.diff_cfg.loss.loss_x0_cos
                if self.diff_cfg.loss.loss_x0_mse > 0: 
                    loss_x0_mse = F.mse_loss(latents, df_pred_tgt.to(latents.device), reduction="mean")
                    if loss_x0_mse.isnan():
                        losses['loss_x0_mse'] += 0 * loss_x0_mse
                    else:
                        losses['loss_x0_mse'] += self.diff_cfg.loss.loss_x0_mse * loss_x0_mse 
                
                if self.diff_cfg.loss.loss_align_weight > 0:  # default:0
                    assert self.predictor is not None
                    if self.head_type == 'LatentPredictHead':
                        pred = self.predictor([condition_feat])[0]
                    else:
                        pred = self.predictor(noisy_latents=proj_online, condition_feat=torch.zeros((proj_online.shape), device=proj_online.device), 
                                          time_emb=torch.zeros((proj_online.shape), device=proj_online.device))
                    losses['loss_align'] += 2. * self.cosineSimilarityFunc(pred.to(latents.device), df_pred_tgt.to(
                        latents.device)) * self.diff_cfg.loss.loss_align_weight
                    
                if self.test_cfg is not None and self.test_cfg.show_dfcos and self.iter_count % self.test_cfg.show_dfcos_freq == 0:
                    print("average cosine similarity between tgt and tgt': ", -1. * cosine_similarity)
        
        if self.img_size!=448:
            if self.decoder_layer == "neck":
                decoder_online_feat_zip = zip(projs_online[:2], views_online[:2])
                decoder_target_feat_zip = zip(projs_target[:2], views_target[:2])
            elif self.decoder_layer == "backbone":
                decoder_online_feat_zip = zip(maps_online[:2], views_online[:2])
                decoder_target_feat_zip = zip(maps_target[:2], views_target[:2])
                
            losses['loss_decoder_online'] = 0
            for proj_online, img in decoder_online_feat_zip:
                proj_online = proj_online.detach()
                img = img.detach()
                image_online = self.decoder_online(proj_online)
                # print(torch.mean(torch.abs(img)))
                # print(torch.mean(torch.abs(img.float())))
                # print(torch.mean(torch.abs(image_online)))
                # print(torch.mean(torch.abs(image_online.float())))
                losses['loss_decoder_online'] += F.mse_loss(image_online.float(), img.float(), reduction="mean")
            # losses['loss_decoder_online'] /= len(projs_online)
            losses['loss_decoder_target'] = 0
            for proj_target, img in decoder_target_feat_zip:
                proj_target = proj_target.detach()
                img = img.detach()
                image_target = self.decoder_target(proj_target)
                losses['loss_decoder_target'] += F.mse_loss(image_target.float(), img.float(), reduction="mean")
            # losses['loss_decoder_target'] /= len(projs_target)
        
        self.scale_losses(losses, len(projs_pairs) / 2)
        
        # pdb.set_trace()
        return losses

    def scale_losses(self, losses, scale):
        if self.loss_byol_cos > 0:
            losses['loss_byol_cos'] /= scale
        if self.loss_byol_mse > 0:
            losses['loss_byol_mse'] /= scale
            
        if self.diff_pred:
            if self.diff_cfg.loss.loss_noise_mse > 0:
                losses['loss_noise_mse'] /= scale
            if self.diff_cfg.loss.loss_noise_cos > 0:
                losses['loss_noise_cos'] /= scale
            if self.diff_cfg.loss.loss_x0_cos > 0:
                losses['loss_x0_cos'] /= scale
            if self.diff_cfg.loss.loss_x0_mse > 0:
                losses['loss_x0_mse'] /= scale
            if self.diff_cfg.loss.loss_align_weight > 0:
                losses['loss_align'] /= scale
        # return losses

    def extract_feat(self, inputs: List[torch.Tensor], stage='neck', DMtimes=5, DMstep=0):
        inputs = inputs[0]
        outs = []
        if stage == 'backbone':
            outs = self.backbone(inputs)
        elif stage == 'neck':
            outs = self.neck(self.backbone(inputs))
        elif stage == 'target_backbone':
            outs = self.target_net.module[0](inputs)
        elif stage == 'target_neck':
            # feats = self.target_net.module[0](inputs)
            outs = self.target_net(inputs)
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
    
    def distributionAnalysis(
            self,
            features_tuple,
    ):
        count = 0

        for features in features_tuple:  # [bsz, dim]
            count += 1

            bsz = features.shape[0]
            norms = torch.norm(features, dim=1, keepdim=True)
            normalized_features = features / norms
            cos_sim = torch.matmul(normalized_features, normalized_features.t())  # [bsz, bsz]
            cos_sim = cos_sim * (1 - torch.eye(bsz)).to(features.device)
            variance = torch.var(features, dim=0)
            std_deviation = torch.std(features, dim=0)
            coefficient_of_variation = (torch.std(features, dim=0) / torch.mean(features, dim=0))   # 离散系数
            covariance_matrix = torch.mm(features.t(), features) / bsz

            print('average cosine similarity: ', torch.sum(cos_sim) / (bsz * bsz - bsz))
            print('average variance: ', torch.mean(variance))
            print('average std_deviation: ', torch.mean(std_deviation))
            print('coefficient of variation: ', coefficient_of_variation)

            # if self.test_cfg.vis_tgt_cos:
            #     title = self.test_cfg.pic_path.split("/")[-1]
            #     draw_fig(cos_sim, title, title + '_pic' + str(count) + '.png')

    # modified from diffusers.schedulers.scheduling_ddpm的step方法
    def denoise_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
    ):
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(model_output.device)

        t = timestep
        prev_t = t - 1
        predicted_variance = None
        # if t==500 or t==700 or t==900 or t==950 or t==999:
        #     print("\n\n")
        #     print("t:", t)
        #     print("model_output", model_output)
        #     print("sample", sample)
            
        # 1. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        
        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            # if t==500 or t==700 or t==900 or t==950 or t==999:
            #     print("pred_original_sample", pred_original_sample)
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )
        
        # 3. Clip or threshold "predicted x_0"
        if self.scheduler.config.thresholding:     # default: False
            pred_original_sample = self.scheduler._threshold_sample(pred_original_sample)
        elif self.scheduler.config.clip_sample:     # default: True
            pred_original_sample = pred_original_sample.clamp(
                -self.scheduler.config.clip_sample_range, self.scheduler.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        
        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        # if t==500 or t==700 or t==900 or t==950 or t==999:
        #     print("pred_prev_sample", pred_prev_sample)
        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )
            if self.scheduler.variance_type == "fixed_small_log":
                variance = self.scheduler._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.scheduler.variance_type == "learned_range":
                variance = self.scheduler._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (self.scheduler._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance
        # if t==500 or t==700 or t==900 or t==950 or t==999:
        #     print("pred_prev_sample", pred_prev_sample)
        return pred_prev_sample
 
    # modified from diffusers.schedulers.scheduling_ddim的step方法
    # batch个timesteps的original_sample的预测
    def pred_original_sample(
            self,
            model_output: torch.FloatTensor,
            timesteps: torch.FloatTensor,
            sample: torch.FloatTensor,
            eta: float = 0.0,
            use_clipped_model_output: bool = False,
            generator=None,
            variance_noise: Optional[torch.FloatTensor] = None,
    ):
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(model_output.device)
        # 1. get previous step value (=t-1)
        prev_timesteps = timesteps - 1

        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timesteps]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timesteps]

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - torch.einsum("b,bd->bd", beta_prod_t ** (0.5), model_output)) / (
                    alpha_prod_t ** (0.5)).view(-1, 1)
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        return pred_original_sample


def make_pairs(list1, list2, noSamePos=True):
    pairs = []
    for idx1, item1 in enumerate(list1):
        for idx2, item2 in enumerate(list2):
            if noSamePos:
                if idx1 == idx2:    # 如果下标相同，则跳过
                    continue
            pairs.append([item1, item2])
    return pairs


# data_points的shape应为[num_classes, num_samples_per_class, num_points_per_sample, dim]
def draw_dot_chart(data_points, save_path, save_suffix, fit_method='pca', n_components=3):
    save_path = save_path + '_' + fit_method + '_' + str(n_components) + 'D_' + save_suffix +'.png'
    
    assert len(data_points.shape)==4
    data_points = data_points.cpu()
    num_classes, num_samples_per_class, num_points_per_sample, dim = data_points.shape
    
    colors = ['red', 'green', 'blue', 'cyan', 'yellow', 'magenta', 'black']
    if num_classes <= 7:
        colors = colors[0:num_classes]
    else:
        colors = plt.cm.rainbow(range(num_classes))
        
    if fit_method == 'tsne':
        perplexity = min(30, num_classes * num_samples_per_class * num_points_per_sample - 1)
        fit_transform = TSNE(n_components=n_components, perplexity=perplexity)
    elif fit_method == 'pca':
        fit_transform = PCA(n_components=n_components)
    data_points_ = fit_transform.fit_transform(data_points.view(-1, dim).detach().numpy())    # [num_classes * num_samples_per_class * num_points_per_sample, n_components]
          
    if n_components==2:
        for i in range(num_classes):
            num_points_per_class = num_samples_per_class * num_points_per_sample
            start_idx = i * num_points_per_class
            end_idx = (i + 1) * num_points_per_class
            # print(data_points_[start_idx:end_idx, :])
            plt.scatter(data_points_[start_idx:end_idx, 0], data_points_[start_idx:end_idx, 1],
                        label=f'Class {i + 1}', 
                        color=colors[i],
                        s=5)
        plt.title('2D Scatter Plot of Data Points by Class')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
    elif n_components==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(num_classes):
            num_points_per_class = num_samples_per_class * num_points_per_sample
            start_idx = i * num_points_per_class
            end_idx = (i + 1) * num_points_per_class
            # print(data_points_[start_idx:end_idx, :])
            ax.scatter(data_points_[start_idx:end_idx, 0], data_points_[start_idx:end_idx, 1], data_points_[start_idx:end_idx, 2],
                        label=f'Class {i + 1}', 
                        color=colors[i],
                        s=5)
        plt.title('3D Scatter Plot of Data Points by Class')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        ax.legend()
    plt.show()
    plt.savefig(save_path)
    plt.close()

def draw_fig(data, title, path):
    h, w = data.shape
    data = data.cpu().detach().numpy()

    fig = plt.figure(figsize=(h, w))
    plt.imshow(data, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()

    plt.title(title)

    plt.savefig(path)
    plt.close(fig)

class Decoder(nn.Module):
    def __init__(self, img_size=64, input_dim=256, fc_channel=512, fc_size=8, decoder_layer="neck"):
        super(Decoder, self).__init__()
        self.fc_channel = fc_channel
        # self.fc_size = fc_size

        map_size_list = []
        if img_size == 64:
            self.fc_size = 8
            map_size_list = [16, 32, 64]
        elif img_size == 224:
            self.fc_size = 28
            map_size_list = [56, 112, 224]
        elif img_size == 448:
            self.fc_size = 7
            map_size_list = [28, 112, 448]
        else:
            print("error img size", img_size)
        
        self.decoder_layer = decoder_layer
        if self.decoder_layer == "neck":
            self.fc = nn.Linear(input_dim, self.fc_channel * self.fc_size * self.fc_size)
        # elif decoder_layer == "backbone":
        
        self.decoder = nn.Sequential(
            nn.Conv2d(self.fc_channel, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(size=[map_size_list[0], map_size_list[0]], mode='bilinear', align_corners=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(size=[map_size_list[1], map_size_list[1]], mode='bilinear', align_corners=True),
            nn.Conv2d(32, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Upsample(size=[map_size_list[2], map_size_list[2]], mode='bilinear', align_corners=True),
            nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        if self.decoder_layer == "neck":
            features = self.fc(features)
        features = features.view(-1, self.fc_channel, self.fc_size, self.fc_size)
        return self.decoder(features)

class Transformer(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            num_layers: int = 2,
            num_attention_heads: int = 32,
            attention_head_dim: int = 64,
            dropout=0.0,
            
            cat_or_add='cat', 
            remove_condition_T=False,
            
            cross_attention_dim: Optional[int] = None,
            only_cross_attention: bool = False,
            
            proj_in: bool = True,
            norm_out: bool = True,
            proj_to_clip_embeddings: bool = True,
    ):
        super(Transformer, self).__init__()
        if cross_attention_dim is None: # else: cross_attention_dim is not None
            if cat_or_add == 'cat':     # else: cat_or_add == 'add'
                if not remove_condition_T:
                    dim_in = dim_in * 3
                else:
                    dim_in = dim_in * 2            
        
        inner_dim = num_attention_heads * attention_head_dim
        self.cross_attention_dim = cross_attention_dim
        self.cat_or_add = cat_or_add
        self.remove_condition_T = remove_condition_T
        
        if proj_in:
            self.proj_in = nn.Linear(dim_in, inner_dim)
            dim_in = inner_dim
        else:
            self.proj_in = None
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim = dim_in,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn="gelu",
                    attention_bias=True,
                    only_cross_attention=only_cross_attention,
                )
                for d in range(num_layers)
            ]
        )
        
        self.norm_out = nn.LayerNorm(dim_in) if norm_out else None
        self.proj_to_clip_embeddings = nn.Linear(dim_in, dim_out) if proj_to_clip_embeddings else None
        
    def forward(self, noisy_latents, condition_feat, time_emb):  # 前向传播函数
        if self.cross_attention_dim is not None: 
            hidden_states = noisy_latents
            if self.cat_or_add == 'add':
                if not self.remove_condition_T:
                    encoder_hidden_states = condition_feat + time_emb  # [bs, dim] = [bs, dim_in]
                else:
                    encoder_hidden_states = condition_feat
            elif self.cat_or_add == 'cat':
                if not self.remove_condition_T:
                    encoder_hidden_states = torch.cat((condition_feat, time_emb), dim=1)  # [bs, 3*dim] = [bs, dim_in]
                else:
                    encoder_hidden_states = condition_feat  # [bs, 2*dim] = [bs, dim_in]
        else:
            encoder_hidden_states = None
            if self.cat_or_add == 'add':
                if not self.remove_condition_T:
                    hidden_states = noisy_latents + condition_feat + time_emb  # [bs, dim] = [bs, dim_in]
                else:
                    hidden_states = noisy_latents + condition_feat
            elif self.cat_or_add == 'cat':
                if not self.remove_condition_T:
                    hidden_states = torch.cat((noisy_latents, condition_feat, time_emb), dim=1)  # [bs, 3*dim] = [bs, dim_in]
                else:
                    hidden_states = torch.cat((noisy_latents, condition_feat), dim=1)  # [bs, 2*dim] = [bs, dim_in]
        
        if self.proj_in is not None:
            hidden_states = self.proj_in(hidden_states)     # [bs, dim_in] -> [bs, inner_dim]   dim_in的值被替换为inner_dim
            
        hidden_states = hidden_states.unsqueeze(1)  # [bsz, channel=dim_in] -> [bs, 1, channel=dim_in]
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1) if encoder_hidden_states is not None else None
        
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=None)   # [bsz, 1, dim_in]
        
        if self.norm_out is not None:
            hidden_states = self.norm_out(hidden_states)
        if self.proj_to_clip_embeddings is not None:
            hidden_states = self.proj_to_clip_embeddings(hidden_states)
        hidden_states = hidden_states.squeeze(1)
        return hidden_states

class MLP(nn.Module):
    def __init__(
            self,
            dim_in,  # 输入特征的维度
            dim_out,  # 输出特征的维度
            mlp_dim,  # MLP的隐藏层维度
            num_layers,  # MLP的层数
            bn_on=False,  # 是否使用批量归一化（BatchNorm）
            bias=True,  # 线性层是否使用bias（偏置）
            flatten=False,  # 是否在MLP前将输入展平
            xavier_init=True,  # 是否使用Xavier初始化
            bn_sync_num=1,  # 用于同步BatchNorm的设备数量
            global_sync=False,  # 是否全局同步BatchNorm
            
            cat_or_add='cat', 
            remove_condition_T=False,
    ):
        super(MLP, self).__init__()

        if cat_or_add == 'cat':
            if not remove_condition_T:
                dim_in = dim_in * 3
            else:
                dim_in = dim_in * 2
        self.cat_or_add = cat_or_add
        self.remove_condition_T = remove_condition_T
        self.flatten = flatten
        b = False if bn_on else bias  # 根据bn_on参数确定bias参数：若使用BN则bias暂时不使用
        # assert bn_on or bn_sync_num=1  # 断言：如果使用BatchNorm，则bn_sync_num必须为1
        mlp_layers = [nn.Linear(dim_in, mlp_dim, bias=b)]  # 创建第一层线性层
        mlp_layers[-1].xavier_init = xavier_init  # 为线性层设置Xavier初始化
        for i in range(1, num_layers):  # 创建MLP的剩余层
            if bn_on:  # 如果使用BatchNorm
                if global_sync or bn_sync_num > 1:  # 如果全局同步或bn_sync_num大于1
                    mlp_layers.append(
                        NaiveSyncBatchNorm1d(  # 添加同步BatchNorm层
                            num_sync_devices=bn_sync_num,
                            global_sync=global_sync,
                            num_features=mlp_dim,
                        ))
                else:
                    mlp_layers.append(nn.BatchNorm1d(num_features=mlp_dim))  # 添加BatchNorm层
            mlp_layers.append(nn.ReLU(inplace=True))  # 添加ReLU激活函数层
            if i == num_layers - 1:  # 如果是最后一层
                d = dim_out  # 输出维度为dim_out
                b = bias  # 使用bias
            else:
                d = mlp_dim  # 输出维度为mlp_dim
            mlp_layers.append(nn.Linear(mlp_dim, d, bias=b))  # 添加线性层
            mlp_layers[-1].xavier_init = xavier_init  # 为线性层设置Xavier初始化
        self.projection = nn.Sequential(*mlp_layers)  # 将所有层组合成一个Sequential模型

    def forward(self, noisy_latents, condition_feat, time_emb):  # 前向传播函数
        if self.cat_or_add == 'add':
            if not self.remove_condition_T:
                x = noisy_latents + condition_feat + time_emb  # [bs, dim] = [bs, dim_in]
            else:
                x = noisy_latents + condition_feat
        elif self.cat_or_add == 'cat':
            if not self.remove_condition_T:
                x = torch.cat((noisy_latents, condition_feat, time_emb), dim=1)  # [bs, 3*dim] = [bs, dim_in]
            else:
                x = torch.cat((noisy_latents, time_emb), dim=1)  # [bs, 2*dim] = [bs, dim_in]
            
        if x.ndim == 5:  # 如果输入是5维张量（例如，来自卷积神经网络的特征图）
            x = x.permute((0, 2, 3, 4, 1))  # 重排维度顺序，使通道维度在最后
        if self.flatten:  # 如果需要在MLP前将输入展平
            x = x.reshape(-1, x.shape[-1])  # 将输入展平为二维张量

        return self.projection(x)  # 通过MLP进行前向传播并返回结果


def get_timestep_embedding(
        timesteps: torch.Tensor,  # 时间步张量，一维，每个元素代表一个批次中的时间步
        embedding_dim: int,  # 嵌入维度
        flip_sin_to_cos: bool = False,  # 是否将sin转换为cos，默认为False
        downscale_freq_shift: float = 1,  # 下采样频率偏移系数，控制最低频率，默认为1
        scale: float = 1,  # 嵌入的缩放系数，默认为1
        max_period: int = 10000,  # 最大周期数，控制最小频率，默认为10000
):
    """ 正弦时间步嵌入
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.     一个形状为[N]的张量，包含N个索引，每个索引代表批次中的一个时间步。
                      These may be fractional.  可以是分数。
    :param embedding_dim: the dimension of the output.  输出嵌入的维度
    :param max_period: controls the minimum frequency of the embeddings.   控制最小频率的嵌入
    :return: an [N x dim] Tensor of positional embeddings.      一个形状为[N x dim]的张量，包含位置嵌入
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

