# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.layers.batch_norm import NaiveSyncBatchNorm1d

from mmengine.dist import all_gather
from mmengine.model import ExponentialMovingAverage
from mmengine.config import ConfigDict

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from ..utils import batch_shuffle_ddp, batch_unshuffle_ddp
from .base import BaseSelfSupervisor

from diffusers import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor

@MODELS.register_module()
class DiffMoCo(BaseSelfSupervisor):
    """MoCo.

    Implementation of `Momentum Contrast for Unsupervised Visual
    Representation Learning <https://arxiv.org/abs/1911.05722>`_.
    Part of the code is borrowed from:
    `<https://github.com/facebookresearch/moco/blob/master/moco/builder.py>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact feature
            vectors.
        head (dict): Config dict for module of head functions.
        queue_len (int): Number of negative keys maintained in the
            queue. Defaults to 65536.
        feat_dim (int): Dimension of compact feature vectors.
            Defaults to 128.
        momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.001.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing
            input data. If None or no specified type, it will use
            "SelfSupDataPreprocessor" as type.
            See :class:`SelfSupDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (Union[List[dict], dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 queue_len: int = 65536,
                 feat_dim: int = 128,
                 momentum: float = 0.001,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None,
                 
                 loss_infoNCE_weight: float = 1.0,
                 
                 diff_pred: bool = True,
                 diff_cfg: Optional[ConfigDict] = None,
                 
                 mlp_pred: bool = False,
                 ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # create momentum model
        self.encoder_k = ExponentialMovingAverage(
            nn.Sequential(self.backbone, self.neck), momentum)

        # create the queue
        self.queue_len = queue_len
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.loss_infoNCE_weight = loss_infoNCE_weight
        self.mlp_pred = mlp_pred
        # diffusion model set
        self.diff_pred = diff_pred
        if self.diff_pred:
            # diffusion model参数初始化
            self.diff_cfg = ConfigDict(
            ) if diff_cfg is None else diff_cfg

            # diffusion model - predict noise
            if self.diff_cfg.cat_or_add == 'cat':
                dim_in = self.diff_cfg.model.in_channels * 3
            else:
                dim_in = self.diff_cfg.model.in_channels
            self.model = MLP(
                dim_in=dim_in,
                dim_out=self.diff_cfg.model.out_channels,
                mlp_dim=self.diff_cfg.model.mlp_dim,
                num_layers=self.diff_cfg.model.num_mlp_layers,
                bn_on=self.diff_cfg.model.bn_on,
                bn_sync_num=self.diff_cfg.model.bn_sync_num if self.diff_cfg.model.bn_sync_num else 1,
                global_sync=(self.diff_cfg.model.bn_sync_num and self.diff_cfg.model.global_sync)
            )
            self.scheduler = DDPMScheduler(num_train_timesteps=self.diff_cfg.scheduler.num_train_timesteps,
                                           clip_sample=False,
                                           beta_schedule="linear",
                                           prediction_type=self.diff_cfg.scheduler.prediction_type,
                                           )
            self.cosineSimilarityFunc = MODELS.build(dict(type='CosineSimilarityLoss'))
        
        if self.mlp_pred:
            dim_in = self.diff_cfg.model.in_channels
            self.model = MLP(
                dim_in=dim_in,
                dim_out=self.diff_cfg.model.out_channels,
                mlp_dim=self.diff_cfg.model.mlp_dim,
                num_layers=self.diff_cfg.model.num_mlp_layers,
                bn_on=self.diff_cfg.model.bn_on,
                bn_sync_num=self.diff_cfg.model.bn_sync_num if self.diff_cfg.model.bn_sync_num else 1,
                global_sync=(self.diff_cfg.model.bn_sync_num and self.diff_cfg.model.global_sync)
            )
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Update queue."""
        # gather keys before updating queue
        keys = torch.cat(all_gather(keys), dim=0)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    def loss(self, inputs: List[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        losses = dict()
        assert isinstance(inputs, list)
        im_q = inputs[0]
        im_k = inputs[1]
        # compute query features from encoder_q
        q = self.neck(self.backbone(im_q))[0]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self.encoder_k.update_parameters(
                nn.Sequential(self.backbone, self.neck))

            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)[0]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = batch_unshuffle_ddp(k, idx_unshuffle)

        if self.diff_pred:
            assert self.diff_cfg.loss.loss_df_weight > 0
            losses['loss_df'] = 0
            latents = k
            condition_feat = q
            
            bsz, dim = latents.shape
            timesteps = torch.randint(low=0,
                                          high=self.scheduler.config.num_train_timesteps,
                                          size=(bsz,),
                                          device=latents.device).long()
            noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype)
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
            predicted_embedding = self.model(noisy_latents, condition_feat, timesteps, self.diff_cfg.cat_or_add)
            if self.scheduler.config.prediction_type == "epsilon":  # 预测噪声 default
                diff_target = noise
            elif self.scheduler.config.prediction_type == "sample":  # 预测特征
                diff_target = latents

            loss_df = F.mse_loss(predicted_embedding.float(), diff_target.float(), reduction="mean")
            if loss_df.isnan():
                losses['loss_df'] += 0 * loss_df
            else:
                losses['loss_df'] += self.diff_cfg.loss.loss_df_weight * loss_df
            
            df_pred_k = self.pred_original_sample(predicted_embedding, timesteps, noisy_latents)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [df_pred_k, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [df_pred_k, self.queue.clone().detach()])

            losses['infoNCE'] = self.loss_infoNCE_weight * self.head.loss(l_pos, l_neg)
            # update the queue
            self._dequeue_and_enqueue(k)

        
        return losses

    def extract_feat(self, inputs: List[torch.Tensor], stage='backbone', DMtimes=5, DMstep=0):
        inputs = inputs[0]
        outs = []
        feats = self.backbone(inputs)
        if stage == 'neck':
            feats = self.neck(feats)
            outs = feats
        elif stage == 'DM' and DMtimes!=0:
            condition_feat = self.neck(feats)[0]
            self.scheduler.set_timesteps(self.diff_cfg.scheduler.num_train_timesteps)
            bsz, dim = condition_feat.shape
            for i in range(DMtimes):  # 对所有数据进行DMtimes次denoise loop
                latent = torch.randn(condition_feat.shape, device=condition_feat.device, dtype=condition_feat.dtype)
                for t in self.scheduler.timesteps:  # denoise loop
                    t_ = t.view(-1).to(condition_feat.device)
                    model_output = self.model(latent, condition_feat, t_.repeat(bsz), self.diff_cfg.cat_or_add)
                    latent = self.denoise_step(model_output, t, latent)
                    if t == DMstep:
                        break
                outs.append(latent)
                # feats = torch.stack(feat_list)  # [DMtimes, bsz, dim]
                # feats = feats.permute(1, 0, 2).contiguous()  # [bsz, DMtimes, dim]
            outs = tuple(outs)
        elif stage == 'backbone':
            outs = feats
        return outs
    

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
        if self.scheduler.config.thresholding:
            pred_original_sample = self.scheduler._threshold_sample(pred_original_sample)
        elif self.scheduler.config.clip_sample:     # default: False
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

        # 4. Clip or threshold "predicted x_0"
        # if self.scheduler.config.thresholding:
        #     pred_original_sample = self._threshold_sample(pred_original_sample)
        # elif self.scheduler.config.clip_sample:
        #     pred_original_sample = pred_original_sample.clamp(
        #         -self.scheduler.config.clip_sample_range, self.scheduler.config.clip_sample_range
        #     )

        return pred_original_sample


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
    ):
        super(MLP, self).__init__()

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

    def forward(self, noisy_latents, condition_feat, timesteps, cat_or_add):  # 前向传播函数
        dim = noisy_latents.shape[1]
        time_emb = get_timestep_embedding(timesteps, dim)

        if cat_or_add == 'add':
            x = noisy_latents + condition_feat + time_emb  # [bs, dim]
        elif cat_or_add == 'cat':
            x = torch.cat((noisy_latents, condition_feat, time_emb), dim=1)  # [bs, 3*dim]
        
        # for param in self.parameters():  
        #     if(param.shape[1]==768):
        #         print("\nnoisy_latent_part:")
        #         weight_noisylatent_part = param[:, :256]    
        #         print(noisy_latents)
        #         print(torch.einsum("mn,tn->tm", weight_noisylatent_part, noisy_latents))    # (4096, 256) (bsz, 256) -> (bsz, 4096)
                
        #         print("\ncontidion_part:")
        #         weight_contidion_part = param[:, 256:256*2]
        #         print(condition_feat)
        #         print(torch.einsum("mn,tn->tm", weight_contidion_part, condition_feat))
                
        #         print("\ntimestep_part:")
        #         weight_timestep_part = param[:, 256*2:]
        #         print(time_emb)
        #         print(torch.einsum("mn,tn->tm", weight_timestep_part, time_emb))
                
        #         # print(torch.einsum("mn,tn->tm", weight_timestep_part, time_emb))
        #         exit(0)
        
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

