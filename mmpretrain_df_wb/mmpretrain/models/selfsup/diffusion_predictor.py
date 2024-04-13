# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.distributions as dist

from pytorchvideo.layers.batch_norm import NaiveSyncBatchNorm1d

from mmengine.dist import all_reduce, get_world_size
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from mmengine.config import ConfigDict

from diffusers import DDPMScheduler
from diffusers.models.attention import BasicTransformerBlock

from .vision_transformer import TransformerPredictor
from .diff_utils import DDIMPipeline, DDIMScheduler, get_timestep_embedding, fill_missing_settings

@MODELS.register_module()
class DiffusionPredictor(nn.Module):
    """
    DiffusionPredictor: Module for Diffusion Predictor.  
    """

    def __init__(self,
                 cfg=None):
        super().__init__()
        
        # Initialize config
        self._initialize_config(cfg)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_config(self, cfg):
        default_cfg = ConfigDict(
            model=dict(
                type='MLP', # MLP, Transformer
                in_channels=256,
                out_channels=256,
                mlp_params=dict(
                    num_layers=2,
                    mlp_dim=4096,
                    bn_on=True,
                    bn_sync_num=False,
                    global_sync=False,),
                trans_params = dict(
                    depth=6,
                    predictor_embed_dim=384,
                    num_attention_heads=12,
                ),
            ),
            scheduler=dict(
                num_train_timesteps=1000,
                prediction_type='epsilon'
            ),
            loss=dict(
                loss_noise_mse=0,
                loss_noise_cos=0,
                loss_x0_cos=3,
                loss_x0_mse=0,
                loss_x0_smooth_l1=0,
                loss_inferx0_cos=0
            ),
            distT_cfg = dict(
                type='uniform',
                ratio=[0.5, 1]
            ),
            cat_or_add='cat',
            pred_residual=False,
            remove_condition_T=False,
            test = None
            ) 
        
        self.cfg = cfg
        fill_missing_settings(self.cfg, default_cfg)
        
        self.diff_pred_cal = False
        if self.cfg.loss.loss_x0_mse > 0 or self.cfg.loss.loss_x0_cos > 0 or self.cfg.loss.loss_x0_smooth_l1 > 0:
            self.diff_pred_cal = True
        
    def _initialize_components(self):
        self._initialize_diff_model()
        self._initialize_diff_scheduler()
        self._initialize_loss_func()
        if self.cfg.loss.loss_inferx0_cos > 0:
            self.DDIMpipeline = DDIMPipeline(self.DMmodel, self.DDIMscheduler)
        
    def _initialize_diff_model(self):
        if self.cfg.model.type ==  'MLP':
            self.DMmodel = MLP(
                dim_in=self.cfg.model.in_channels,
                dim_out=self.cfg.model.out_channels,
                
                num_layers=self.cfg.model.mlp_params.num_layers,
                mlp_dim=self.cfg.model.mlp_params.mlp_dim,
                bn_on=self.cfg.model.mlp_params.bn_on,
                bn_sync_num=self.cfg.model.mlp_params.bn_sync_num if self.cfg.model.mlp_params.bn_sync_num else 1,
                global_sync=(self.cfg.model.mlp_params.bn_sync_num and self.cfg.model.mlp_params.global_sync),
            
                cat_or_add=self.cfg.cat_or_add,  
                remove_condition_T=self.cfg.remove_condition_T,
            ) 
        else:
            self.DMmodel = TransformerPredictor(
                embed_dim=self.cfg.model.in_channels,
                
                predictor_embed_dim=self.cfg.model.trans_params.predictor_embed_dim,
                depth=self.cfg.model.trans_params.depth,
                num_heads=self.cfg.model.trans_params.num_attention_heads,
                
                cat_or_add=self.cfg.cat_or_add,  
                remove_condition_T=self.cfg.remove_condition_T,
            )
        
    def _initialize_diff_scheduler(self):
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.cfg.scheduler.num_train_timesteps,
            clip_sample=False,
            beta_schedule="linear",
            prediction_type=self.cfg.scheduler.prediction_type,
            )
        if self.cfg.loss.loss_inferx0_cos > 0:
            self.DDIMscheduler = DDIMScheduler(
                num_train_timesteps=self.cfg.scheduler.num_train_timesteps,
                clip_sample=False,
                beta_schedule="linear",
                prediction_type=self.cfg.scheduler.prediction_type,
                ) 
        
    def _initialize_loss_func(self):
        if self.cfg.loss.loss_noise_mse > 0 or self.cfg.loss.loss_x0_mse > 0:
            self.mse_loss_fn = torch.nn.MSELoss()
        if self.cfg.loss.loss_noise_cos > 0 or self.cfg.loss.loss_x0_cos > 0 or self.cfg.loss.loss_inferx0_cos > 0:
            self.cos_sim_loss_fn = MODELS.build(dict(type='CosineSimilarityLoss'))  # or torch.nn.CosineSimilarity
        if self.cfg.loss.loss_x0_smooth_l1 > 0:
            self.smooth_l1_loss_fn = torch.nn.SmoothL1Loss()
        
    def _reset_losses(self, losses):
        if self.cfg.loss.loss_noise_mse > 0:
            losses['loss_noise_mse'] = 0
        if self.cfg.loss.loss_noise_cos > 0:
            losses['loss_noise_cos'] = 0
        if self.cfg.loss.loss_x0_mse > 0:
            losses['loss_x0_mse'] = 0
        if self.cfg.loss.loss_x0_cos > 0:
            losses['loss_x0_cos'] = 0
        if self.cfg.loss.loss_x0_smooth_l1 > 0:
            losses['loss_x0_smooth_l1'] = 0
        if self.cfg.loss.loss_inferx0_cos > 0:
            losses['loss_inferx0_cos'] = 0
    
    def _scale_losses(self, losses, scale):
        if self.cfg.loss.loss_noise_mse > 0:
            losses['loss_noise_mse'] /= scale
        if self.cfg.loss.loss_noise_cos > 0:
            losses['loss_noise_cos'] /= scale
        if self.cfg.loss.loss_x0_cos > 0:
            losses['loss_x0_cos'] /= scale
        if self.cfg.loss.loss_x0_mse > 0:
            losses['loss_x0_mse'] /= scale
        if self.cfg.loss.loss_x0_smooth_l1 > 0:
            losses['loss_x0_smooth_l1'] /= scale
        if self.cfg.loss.loss_inferx0_cos > 0:
            losses['loss_inferx0_cos'] /= scale
    
    def sample_timesteps(self, bsz, device):
        if self.cfg.distT_cfg.type == 'gaussian':
            Tupper = self.scheduler.config.num_train_timesteps
            Tmin, Tmax = [int(x * Tupper) for x in self.cfg.distT_cfg.ratio]
            mu = (Tmin + Tmax) / 2
            gaussian_dist = dist.Normal(mu, 1)
            timesteps = torch.clamp(torch.round(gaussian_dist.sample(sample_shape=(bsz,))), Tmin, Tmax).long().to(device)
        elif self.cfg.distT_cfg.type == 'inc':
            Tupper = self.scheduler.config.num_train_timesteps
            Tmin, Tmax = [int(x * Tupper) for x in self.cfg.distT_cfg.ratio]
            probs = torch.arange(Tmin, Tmax)
            linear_inc_dist = dist.Categorical(probs)
            timesteps = linear_inc_dist.sample(sample_shape=(bsz,)).to(device)
        elif self.cfg.distT_cfg.type == 'dec':
            Tupper = self.scheduler.config.num_train_timesteps
            Tmin, Tmax = [int(x * Tupper) for x in self.cfg.distT_cfg.ratio]
            probs = torch.arange(Tmax, Tmin, -1)
            linear_inc_dist = dist.Categorical(probs)
            timesteps = linear_inc_dist.sample(sample_shape=(bsz,)).to(device)
        elif self.cfg.distT_cfg.type == 'uniform':
            Tupper = self.scheduler.config.num_train_timesteps
            Tmin, Tmax = [int(x * Tupper) for x in self.cfg.distT_cfg.ratio]
            timesteps = torch.randint(low=Tmin, high=Tmax, size=(bsz,), device=device).long()
        return timesteps

    # modified from diffusers.schedulers.scheduling_ddim的step方法
    # batch个timesteps的original_sample的预测
    def pred_original_sample(
            self,
            model_output: torch.FloatTensor,
            timesteps: torch.FloatTensor,
            sample: torch.FloatTensor,
            
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
    
    # output_pred和output_loss至少有一个是true
    def forward(self, latent_condition_pairs, output_both=False, output_pred=False, output_loss=True):
        """Forward head.

        Args:
            

        Returns:

        """
        
        # reset losses
        losses = dict()
        self._reset_losses(losses)
        middle_embedding_list, diff_pred_cal_list, diff_pred_infer_list = [], [], []
        
        for latents, condition_feat in latent_condition_pairs:
            if self.cfg.pred_residual:
                latents = latents - condition_feat
            bsz, dim = latents.shape
            
            timesteps = self.sample_timesteps(bsz, latents.device)
            noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype)
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
            time_emb = get_timestep_embedding(timesteps, dim)
            
            predicted_embedding, middle_embedding = self.DMmodel(noisy_latents, condition_feat, time_emb)
            middle_embedding_list.append(middle_embedding)
            
            if self.scheduler.config.prediction_type == "epsilon":
                diff_target = noise
            elif self.scheduler.config.prediction_type == "sample":
                diff_target = latents

            if self.cfg.loss.loss_noise_mse > 0:
                loss_noise_mse = self.mse_loss_fn(predicted_embedding.float(), diff_target.float())
                if loss_noise_mse.isnan():
                    losses['loss_noise_mse'] += torch.tensor(0, dtype=torch.float32, device=latents.device)
                else:
                    losses['loss_noise_mse'] += self.cfg.loss.loss_noise_mse * loss_noise_mse
            if self.cfg.loss.loss_noise_cos > 0:
                loss_noise_cos = self.cos_sim_loss_fn(predicted_embedding.float(), diff_target.float())
                losses['loss_noise_cos'] += 2. * loss_noise_cos * self.cfg.loss.loss_noise_cos

            if self.diff_pred_cal:
                diff_pred_cal = self.pred_original_sample(predicted_embedding, timesteps, noisy_latents)
                diff_pred_cal_list.append(diff_pred_cal)
                if self.cfg.loss.loss_x0_mse > 0:
                    loss_x0_mse = self.mse_loss_fn(latents, diff_pred_cal.to(latents.device))
                    if loss_x0_mse.isnan():
                        losses['loss_x0_mse'] += torch.tensor(0, dtype=torch.float32, device=latents.device)
                    else:
                        losses['loss_x0_mse'] += self.cfg.loss.loss_x0_mse * loss_x0_mse
                if self.cfg.loss.loss_x0_cos > 0:
                    x0_cos_sim = self.cos_sim_loss_fn(latents, diff_pred_cal.to(latents.device))
                    if x0_cos_sim.isnan():
                        losses['loss_x0_cos'] += torch.tensor(0, dtype=torch.float32, device=latents.device)
                    else:
                        losses['loss_x0_cos'] += 2. * x0_cos_sim * self.cfg.loss.loss_x0_cos
                if self.cfg.loss.loss_x0_smooth_l1 > 0:
                    loss_x0_smooth_l1 = self.smooth_l1_loss_fn(latents, diff_pred_cal.to(latents.device))
                    if loss_x0_smooth_l1.isnan():
                        losses['loss_x0_smooth_l1'] += torch.tensor(0, dtype=torch.float32, device=latents.device)
                    else:
                        losses['loss_x0_smooth_l1'] += self.cfg.loss.loss_x0_smooth_l1 * loss_x0_smooth_l1
            
            if self.cfg.loss.loss_inferx0_cos > 0:
                x_T = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype)
                x_0 = self.DDIMpipeline(
                    batch_size=latents.shape[0],
                    device=latents.device,
                    dtype=latents.dtype,
                    shape=latents.shape[1:],
                    feat=x_T,
                    condition=condition_feat,
                    num_inference_steps=self.cfg.scheduler.num_inference_timesteps,
                )
                diff_pred_infer_list.append(x_0)
                losses['loss_inferx0_cos'] += self.cfg.loss.loss_inferx0_cos * self.cos_sim_loss_fn(latents, x_0.to(latents.device))
            
        self._scale_losses(losses, len(latent_condition_pairs) / 2)
        
        output = {
            'losses': losses,
            'middle_embedding_list': middle_embedding_list,
            'diff_pred_cal_list': diff_pred_cal_list,
            'diff_pred_infer_list': diff_pred_infer_list
        }
        return output

        # if output_both:
        #     return diff_pred_list, losses
        # elif output_pred:
        #     return diff_pred_list
        # elif output_loss:
        #     return losses

class Transformer(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            num_layers: int = 2,
            num_attention_heads: int = 32,
            attention_head_dim: int = 64,
            dropout=0.0,
            
            cross_attention_dim: Optional[int] = None,
            only_cross_attention: bool = False,

            proj_in: bool = True,
            norm_out: bool = True,
            proj_to_clip_embeddings: bool = True,
            
            cat_or_add='cat',
            remove_condition_T=False,
    ):
        super(Transformer, self).__init__()
        if cross_attention_dim is None:  # else: cross_attention_dim is not None
            if cat_or_add == 'cat':  # else: cat_or_add == 'add'
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
                    dim=dim_in,
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

    def forward(self, noisy_latents, condition_feat, time_emb):
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
        else:   # 只有self-attention
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
            hidden_states = self.proj_in(hidden_states)  # [bs, dim_in] -> [bs, inner_dim]   dim_in的值被替换为inner_dim

        hidden_states = hidden_states.unsqueeze(1)  # [bsz, channel=dim_in] -> [bs, 1, channel=dim_in]
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1) if encoder_hidden_states is not None else None

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                                  attention_mask=None)  # [bsz, 1, dim_in]

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
        # assert bn_on or bn_sync_num=1
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
        
        self.intermediate_output = None
        def hook(module, data_input, data_output):
            self.intermediate_output = data_output
        self.projection[0].register_forward_hook(hook)

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

        return self.projection(x), self.intermediate_output

