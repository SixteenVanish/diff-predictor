# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union
import os
import datetime
import numpy as np
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import matplotlib.pyplot as plt


from mmengine.dist import all_reduce, get_world_size
from mmengine.model import BaseModule
from mmpretrain.registry import MODELS
from mmpretrain.utils import distributed as du
from pytorchvideo.layers.batch_norm import NaiveSyncBatchNorm1d

class DINOLoss(nn.Module):
    """
    DINOLoss: Cross-entropy loss between softmax outputs of teacher and student networks.
    ref: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/main_dino.py#L363
    """

    def __init__(self,
                 out_dim,
                 teacher_temp=0.01,
                 student_temp=0.1,
                 center_momentum=0.9,
                 sync_center=True):
        """
        Initialize DINOLoss.

        Args:
            out_dim (int): Dimensionality of output.
            teacher_temp (float): Temperature for teacher network.
            student_temp (float): Temperature for student network.
            center_momentum (float): Momentum for center update.
            sync_center (bool): Flag to synchronize center update.
        """
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp = teacher_temp
        self.sync_center = sync_center

    def forward(self, student_output, teacher_output):
        """
        Compute the loss using cross-entropy between softmax outputs.

        Args:
            student_output (Tensor): Output of student network.
            teacher_output (Tensor): Output of teacher network.

        Returns:
            loss (Tensor): Computed loss.
        """
        student_out = student_output / self.student_temp

        teacher_out = F.softmax(
            (teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1),
                         dim=-1).mean()
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update the center used for teacher output.

        Args:
            teacher_output (Tensor): Output of teacher network.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if self.sync_center:
            du.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) *
                                           du.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum)

def save_data(data, path):
    numpy_array = data.cpu().detach().numpy()
    np.savetxt(path, numpy_array)

def draw_line(data, title, path='pic.png'):
    data = data.cpu().detach().numpy()
    # 创建图形
    fig = plt.figure(figsize=(20, 6))
    # 画出向量
    plt.plot(data)
    # 设置图形标题和轴标签
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')
    # 得到图形
    plt.savefig(path)
    plt.close(fig)

def draw_fig(data, title, path='pic.png'):
    h, w = data.shape
    data = data.cpu().detach().numpy()

    data_no_diag = data * (1 - np.eye(h))

    fig = plt.figure(figsize=(h, w))
    plt.imshow(data_no_diag, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()

    plt.title(title)

    plt.savefig(path)
    plt.close(fig)

def visualization(sample_sim_Mp, sim_MM, sample_soft_sim_Mp, path):
    curr_time = datetime.datetime.now()
    timestamp = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    path = os.path.join(path, 'vis_'+timestamp)
    if not os.path.exists(path) and dist.get_rank() == 0:
        print(path)
        os.makedirs(path)
    dist.barrier()
    draw_line(sample_sim_Mp, 'sample_sim_Msrc_Psrc', os.path.join(path, 'sample_sim_Msrc_Psrc.png'))
    draw_line(sample_soft_sim_Mp, 'sample_soft_sim_Msrc_Psrc', os.path.join(path, 'sample_soft_sim_Msrc_Psrc.png'))
    draw_fig(sim_MM, 'sim_Msrc_Msrc', os.path.join(path, 'sim_Msrc_Msrc.png'))
    save_data(sim_MM, os.path.join(path, 'sim_Msrc_Msrc.txt'))

@MODELS.register_module()
class MemReconHead(BaseModule):
    """
       MemoryRecon: Module for memory reconstruction.
       """

    def __init__(self, dim_input=256, dim_output=256, cfg=None):
        """
        Initializes the MemoryRecon module.

        Args:
            dim_input (int): Dimension of the input features.
            dim_output (int): Dimension of the output features.
            cfg (Namespace): Configuration containing memory-related parameters.
        """
        super(MemReconHead, self).__init__()
        self.dim_input = dim_input  # source feature dim
        self.dim_output = dim_output  # target feature dim
        self.cfg = cfg

        # Handle different recon_content scenarios
        # recon的是y_src，因此维度需要转换
        if cfg.MODEL.ARCH == "r3d_shallow" or cfg.RESNET.DEPTH <= 34:
            self.recon_dim = cfg.RESNET.WIDTH_PER_GROUP * 8
            self.recon_mid_dim = cfg.RESNET.WIDTH_PER_GROUP * 8
        else:
            self.recon_dim = cfg.RESNET.WIDTH_PER_GROUP * 32
            self.recon_mid_dim = self.cfg.RESNET.WIDTH_PER_GROUP * 32
        self.pool_feat = nn.AdaptiveAvgPool2d((1, 1))

        # Initialize various attributes based on configuration
        self._initialize_attributes()

        # Initialize components
        self._initialize_components()

        # Initialize softmax
        self.softmax = nn.Softmax(-1)

    def _initialize_attributes(self):
        """
        Initializes attributes based on configuration settings.
        """
        # Extract relevant configuration attributes from memory configuration
        memory_cfg = self.cfg.MEMORY

        # Dimension of the source visual concept dictionary
        self.dim_source_mem = memory_cfg.DIM_SOURCE_MEM
        # Dimension of the target visual concept dictionary
        self.dim_target_mem = memory_cfg.DIM_TARGET_MEM
        # Dimension of the key-value cross memory
        self.dim_cross_mem = memory_cfg.DIM_CROSS_MEM

        # Scaling factor for attention distribution
        self.radius = memory_cfg.RADIUS
        # Number of visual concepts in the dictionaries
        self.n_concept = memory_cfg.N_CONCEPT
        # Number of slots for key and value memories
        self.n_slot = memory_cfg.N_SLOT
        # Number of attention heads of the memory
        self.n_head = memory_cfg.N_HEAD
        # Type of loss function for feature reconstruction using dictionary learning
        self.recon_loss_type = memory_cfg.RECON_LOSS_TYPE
        # Average dimension of the reconstruction loss
        self.average_dim = memory_cfg.AVERAGE_DIM
        # Whether to use input projection before memory
        self.use_input_proj = memory_cfg.USE_INPUT_PROJ
        # Dimension for input projection to cross memory
        self.dim_cross_input_proj = memory_cfg.DIM_CROSS_INPUT_PROJ
        # Whether to use output projection after memory
        self.use_output_proj = memory_cfg.USE_OUTPUT_PROJ
        # Whether to use source feature reconstruction
        self.use_src_recon = memory_cfg.USE_SRC_RECON
        # Whether to use target feature reconstruction
        self.use_tar_recon = memory_cfg.USE_TAR_RECON
        # Whether to use KL divergence loss for alignment between visual concept codes
        self.use_align = memory_cfg.USE_ALIGN
        # Temperature for KL divergence loss
        self.kl_t = memory_cfg.KL_T

        # Whether to use contrastive loss for source memory
        self.use_source_mem_contrastive = memory_cfg.USE_SOURCE_MEM_CONTRASTIVE
        # Whether to use contrastive loss for target memory
        self.use_target_mem_contrastive = memory_cfg.USE_TARGET_MEM_CONTRASTIVE
        # Whether to use contrastive loss for cross memory
        self.use_cross_mem_contrastive = memory_cfg.USE_CROSS_MEM_CONTRASTIVE

        # Type of addressing for memory (e.g., "cosine" or other)
        self.address_type = memory_cfg.ADDRESS_TYPE
        # Whether to predict residuals for memory update
        self.predict_residual = memory_cfg.PREDICT_RESIDUAL

        # Whether to use sparse coding for memory
        self.use_sparse = memory_cfg.USE_SPARSE
        # Whether to use sparse coding before memory (not clear)
        self.use_sparse_before = memory_cfg.USE_SPARSE_BEFORE
        # Regulization loss type for sparse coding
        self.sparse_loss_type = memory_cfg.SPARSE_LOSS_TYPE
        # Top-k for sparse coding
        self.sparse_topk = memory_cfg.SPARSE_TOPK

        self.predict_type = memory_cfg.PREDICT_LOSS_TYPE
        self.dino_temp_predict_t = memory_cfg.DINO_TEMP_PREDICT_T
        self.dino_temp_predict_s = memory_cfg.DINO_TEMP_PREDICT_S

        # Type of the structure for feature reconstruction in the dictionary learning
        self.recon_type = memory_cfg.RECON_TYPE
        # Type of the structure of the predictor, memory for the key-value memory enhanced predictor and mlp for the mlp predictor
        self.cross_branch_type = memory_cfg.CROSS_BRANCH_TYPE
        # Temperature for DINO loss (teacher)
        self.dino_temp_align_t = memory_cfg.DINO_TEMP_ALIGN_T
        # Temperature for DINO loss (student)
        self.dino_temp_align_s = memory_cfg.DINO_TEMP_ALIGN_S
        # Number of MLP layers for the dictionary reconstruction branch
        self.num_mlp_layers = memory_cfg.NUM_MLP_LAYERS
        # Number of MLP layers for the source feature projection branch
        self.num_mlp_layers_in = memory_cfg.NUM_MLP_LAYERS_IN
        # Number of MLP layers for the output feature projection branch
        self.num_mlp_layers_out = memory_cfg.NUM_MLP_LAYERS_OUT

        # Loss weight
        self.loss_src_alpha = memory_cfg.LOSS_SOURCE_RECON
        self.loss_tgt_alpha = memory_cfg.LOSS_TARGET_RECON
        self.loss_beta = memory_cfg.LOSS_KL

    def _initialize_components(self):
        # Initialize source reconstruction components
        if self.use_src_recon:
            self._initialize_source_recon()

        # Initialize target reconstruction components
        if self.use_tar_recon:
            self._initialize_target_recon()

        # Initialize cross-branch components
        if self.cross_branch_type == "memory":
            self._initialize_cross_memory()
        else:
            self._initialize_cross_branch()

        # Initialize input projection components
        if self.use_input_proj:
            self._initialize_input_projection()

        # Initialize output projection components
        if self.use_output_proj:
            self._initialize_output_projection()

        # Initialize KL and loss alignment components
        self._initialize_kl_dino_loss()

    def _initialize_source_recon(self):
        if self.address_type == "cosine":  # 带有权重归一化的线性层
            self.source_key = nn.utils.weight_norm(
                nn.Linear(self.dim_source_mem, self.n_concept, bias=False))
            self.source_key.weight_g.data.fill_(1)  # 权重初始化为1
            self.source_key.weight_g.requires_grad = False  # 不需要梯度
        else:  # 使用普通的线性层
            self.source_key = nn.Linear(self.dim_source_mem,
                                        self.n_concept,
                                        bias=False)

        if self.recon_type == "memory":  # 使用参数张量作为source_mem
            self.source_mem = nn.Parameter(torch.Tensor(
                self.n_concept, self.dim_source_mem),
                requires_grad=True)
            nn.init.trunc_normal_(self.source_mem, std=0.02)  # 使用截断正态分布进行初始化，标准差为0.02
        else:  # 使用MLPHead
            self.source_recon = MLPHead(
                self.n_concept,
                self.recon_dim,
                self.recon_mid_dim,
                self.num_mlp_layers,
                bn_on=self.cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
                if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP
                             and self.cfg.BN.GLOBAL_SYNC),
            )

    def _initialize_target_recon(self):
        if self.address_type == "cosine":
            self.target_key = nn.utils.weight_norm(
                nn.Linear(self.dim_target_mem, self.n_concept, bias=False))
            self.target_key.weight_g.data.fill_(1)
            self.target_key.weight_g.requires_grad = False
        else:
            self.target_key = nn.Linear(self.dim_target_mem,
                                        self.n_concept,
                                        bias=False)

        if self.recon_type == "memory":
            self.target_mem = nn.Parameter(torch.Tensor(
                self.n_concept, self.dim_target_mem),
                requires_grad=True)
            nn.init.trunc_normal_(self.target_mem, std=0.02)
        else:
            self.target_recon = MLPHead(
                self.n_concept,
                self.recon_dim,
                self.recon_mid_dim,
                self.num_mlp_layers,
                bn_on=self.cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
                if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP
                             and self.cfg.BN.GLOBAL_SYNC),
            )

    def _initialize_cross_memory(self):  # predictor为key-value memory enhanced predictor
        self.cross_key = nn.utils.weight_norm(  # 使用权重归一化的线性层作为cross key，将输入投影到slot空间
            nn.Linear(self.dim_cross_input_proj, self.n_slot, bias=False))
        self.cross_key.weight_g.data.fill_(1)

        self.cross_key.weight_g.requires_grad = False
        self.cross_mem = nn.Parameter(torch.Tensor(self.n_slot,
                                                   self.dim_cross_mem),
                                      requires_grad=True)
        nn.init.trunc_normal_(self.cross_mem, std=0.02)  # 使用截断正态分布进行初始化，标准差为0.02

    def _initialize_cross_branch(self):  # predictor为MLP
        self.cross_branch = MLPHead(
            self.dim_input,
            self.dim_output,
            self.cfg.CONTRASTIVE.MLP_DIM,
            2,
            bn_on=self.cfg.CONTRASTIVE.BN_MLP,
            bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
            if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
            global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP and self.cfg.BN.GLOBAL_SYNC),
        )

    def _initialize_input_projection(self):
        if self.cross_branch_type == "memory":
            self.in_proj_cross = MLPHead(
                self.dim_input,
                self.n_head * self.dim_cross_input_proj,
                self.n_head * self.dim_cross_input_proj
                if self.num_mlp_layers_in == 1 else self.dim_cross_input_proj,
                self.num_mlp_layers_in,
                bn_on=self.cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
                if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP
                             and self.cfg.BN.GLOBAL_SYNC),
            )

        if self.use_src_recon and self.recon_type == "memory":
            self.in_proj_src = MLPHead(
                self.dim_input,
                self.dim_source_mem,
                self.dim_input,
                self.num_mlp_layers_in,
                bn_on=self.cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
                if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP
                             and self.cfg.BN.GLOBAL_SYNC),
            )

        if self.use_tar_recon and self.recon_type == "memory":
            self.in_proj_tar = MLPHead(
                self.dim_output,
                self.dim_target_mem,
                self.dim_output,
                self.num_mlp_layers_in,
                bn_on=self.cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
                if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP
                             and self.cfg.BN.GLOBAL_SYNC),
            )

    def _initialize_output_projection(self):
        self.out_proj_cross = MLPHead(
            self.n_head * self.dim_cross_mem,
            self.dim_output,
            self.dim_output,
            self.num_mlp_layers_out,
            bn_on=self.cfg.CONTRASTIVE.BN_MLP,
            bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
            if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
            global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP and self.cfg.BN.GLOBAL_SYNC),
        )

        if self.use_src_recon and self.recon_type == "memory":
            self.out_proj_source = MLPHead(
                self.dim_source_mem,
                self.recon_dim,
                self.recon_mid_dim,
                self.num_mlp_layers,
                bn_on=self.cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
                if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP
                             and self.cfg.BN.GLOBAL_SYNC),
            )

        if self.use_tar_recon and self.recon_type == "memory":
            self.out_proj_tar = MLPHead(
                self.dim_target_mem,
                self.recon_dim,
                self.recon_mid_dim,
                self.num_mlp_layers,
                bn_on=self.cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
                if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP
                             and self.cfg.BN.GLOBAL_SYNC),
            )

    def _initialize_kl_dino_loss(self):
        self.dino_loss_align = DINOLoss(
            out_dim=self.n_concept,
            student_temp=self.dino_temp_align_s,
            teacher_temp=self.dino_temp_align_t,
            sync_center=True if self.cfg.NUM_GPUS > 1 else False).cuda()

        self.dino_loss_predict = DINOLoss(
            out_dim=self.dim_output,
            student_temp=self.dino_temp_predict_s,
            teacher_temp=self.dino_temp_predict_t,
            sync_center=True if self.cfg.NUM_GPUS > 1 else False).cuda()

    @staticmethod
    def regularization_loss(input, cfg, topk=-1):
        try:
            # Try to extract dimensions from the input tensor
            BS, n_head, n_concept = input.shape
        except:
            # If the input has only two dimensions, assume n_head = 1
            BS, n_concept = input.shape
            n_head = 1

        if cfg.sparse_loss_type == 'l1':
            # L1-norm regularization loss
            loss = torch.norm(input, p=1) / BS
        elif cfg.sparse_loss_type == 'neglog':
            # Negative log-likelihood regularization loss
            loss = torch.mean(-input * torch.log(input))
        elif cfg.sparse_loss_type == 'max_margin':
            # Maximum margin regularization loss
            loss = torch.mean(
                torch.max(
                    torch.min(input, dim=0)[0] - torch.max(input, dim=0)[0] +
                    0.1,
                    torch.zeros_like(torch.min(input, dim=0)[0]),
                ))
        elif cfg.sparse_loss_type == 'max_margin_neglog':
            # Combination of max margin and negative log-likelihood losses
            topk_ = 10
            loss = 0.01 * torch.sum(
                torch.max(
                    torch.min(input, dim=0)[0] - torch.max(input, dim=0)[0] +
                    0.1,
                    torch.zeros_like(torch.min(input, dim=0)[0]),
                )) + 0.1 * torch.sum(1 - torch.sum(
                torch.topk(input, topk_, dim=-1)[0], dim=-1)) / (BS *
                                                                 n_head)
        else:
            # Top-k regularization loss
            loss = torch.sum(1 - torch.sum(torch.topk(input, topk, dim=-1)[0],
                                           dim=-1)) / (BS * n_head)
        return loss

    @staticmethod  # 静态方法，该方法属于类本身，而不是类的实例
    def reconstruction_loss(pred, tar, loss_type="cosine", average_dim=0):
        assert pred.size() == tar.size() and tar.numel() > 0  # 维度相同且元素数量>0
        if loss_type == "l2":
            loss = torch.sum(torch.pow(pred - tar, 2))
        elif loss_type == "cosine":
            loss = torch.abs(1 - F.cosine_similarity(pred, tar, 1)).sum()
        else:
            raise RuntimeError(f"Loss type {loss_type} is not supported.")

        if average_dim == -1:
            loss /= tar.numel()
        else:
            loss /= tar.shape[average_dim]
        return loss

    @staticmethod
    def contrastive_loss(self, input):
        """
        Contrastive loss to encourage distinctiveness of visual features stored in different memory slots.

        Args:
            input (torch.Tensor): A 2D tensor representing the features stored in value memories.

        Returns:
            torch.Tensor: Contrastive loss value.
        """
        assert len(input.shape) == 2

        # Compute the confusion matrix as described in the contrastive learning method
        confusion_matrix = torch.abs(
            torch.eye(input.shape[0]).cuda() - torch.matmul(
                F.normalize(input, dim=-1),
                F.normalize(input, dim=-1).transpose(0, 1),
            ))

        # Compute the separate loss by summing up the values in the confusion matrix and dividing by the number of slots
        separate_loss = torch.sum(confusion_matrix) / input.shape[0]

        return separate_loss

    @staticmethod
    def mempred_loss(pred, tar):
        # Lmempred: 负的余弦相似度
        # the negative cosine similarity between the predicted target representation and the target representation
        bs = pred.shape[0]
        cos_sim = F.cosine_similarity(pred, tar)
        avg_cos_sim = torch.sum(cos_sim, dim=0) / bs
        return avg_cos_sim


    def loss(self, source, infos=None):
        """
        Forward pass of MemoryRecon.

        Args:
            source (Tensor): Input source data.
            infos (list): List of additional information.

        Returns:
            output (dict): Output containing various losses and predictions.
        """

        # Ensure the length of 'infos' is 3
        assert len(infos) == 3

        # 'infos'中的z_tgt
        if infos[0] is not None:
            target = infos[0].detach()  # todo 加或者不加.detach()有影响吗
            # if len(infos[0]) > 1:  # 如果'infos[0]'的长度大于1，则将其元素连接成一个张量
            #     target = torch.cat(infos[0]).detach()
            # else:  # 如果'infos[0]'的长度不大于1，则取其第一个元素（也是唯一元素）
            #     target = infos[0][0].detach()
        else:
            # 克隆z_src以用于计算params和flops
            target = source.clone().detach()

        # Initialize variables for features 初始化特征变量
        feat, feat_target = None, None
        if infos[1] is not None:  # 如果提供了'infos[1]'，则取其第一个元素作为feat
            # feat = infos[1][0]
            feat = infos[1]
        else:  # 如果没有提供'infos[1]'，则生成一个形状与源数据相同，数据类型与源数据相同的随机张量作为feat
            feat = torch.randn(source.shape[0], self.recon_dim, 1, 1,
                               1).type_as(source)
        if infos[2] is not None:
            feat_target = infos[2]
            # if len(infos[2]) > 1:  # 如果'infos[2]'的长度大于1，则将其元素连接成一个张量作为feat_target
            #     feat_target = torch.cat([feat_tar[0]
            #                              for feat_tar in infos[2]]).detach()
            # else:  # 如果'infos[2]'的长度不大于1，则取其第一个元素的第一个元素作为目标特征
            #     feat_target = infos[2][0][0]
        else:  # 如果没有提供'infos[2]'，则生成一个形状与源数据相同，数据类型与源数据相同的随机张量作为feat_target
            feat_target = torch.randn(source.shape[0], self.recon_dim, 1, 1,
                                      1).type_as(source)

        # Get batch size and feature dimension
        B, C = source.size()

        # Initialize output dictionary
        output = {}

        # Calculate projection of source data for cross memory mechanism
        if self.use_input_proj and self.cross_branch_type == "memory":
            source_cross_proj = self.in_proj_cross(source)  # 𝑝_src = 𝜙 (𝑧_src)
        else:
            source_cross_proj = source

        # Calculate cross memory mechanism and address
        if self.cross_branch_type == "memory":  # key-value memory enhanced predictor
            # 𝑠𝑖𝑚(𝑴_src, 𝑝_src)=𝑴_src, 𝑝_src之间的余弦相似度
            # bs个(h, d)与1个(h, s, d) -> bs个(h, s)
            # 在每个head中， d 与 (s, d) -> (s), s为每个head的slot数
            # 即，在每个head中， 该head中对应的每个d维向量（共bs个），与s个该head对应的slot对应的d维向量，点积
            # 最后得到，(bs, h=head数, s=slot数/head)， 其中每个元素表示𝑝_src的每个头(head)的每个slot与对应key的slot向量的相似度
            cross_mem_sim = torch.einsum(
                "bhd,hsd->bhs",
                F.normalize(
                    source_cross_proj.view(B, self.n_head,
                                           self.dim_cross_input_proj),
                    dim=2,
                ),
                F.normalize(
                    self.cross_key.weight_v.view(
                        self.n_head,
                        self.n_slot // self.n_head,
                        self.dim_cross_input_proj,
                    ),
                    dim=2,
                ),
            )
            if self.cfg.TEST.VIS_MIDDLE:    # default: False
                output["vis_cross_mem_sim"] = cross_mem_sim
            cross_mem_address = self.softmax(self.radius * cross_mem_sim)  # 公式(2) the knowledge relevance score
            # bs个(h, s)与1个(h, s, d) -> bs个(h, d)
            # 在每个head中， s 与 (s, d) -> (d), s为每个head的slot数
            # 即，bs个数据，对每个head中的每个slot有一个relevance score，与memory中的每个head中的每个slot对应的d维向量，算加权和
            # 得到一个d维的𝑧_tgt′。得到，(bs, h=head数, d)， 其中每个元素表示每个数据的每个头(head)所得的对应𝑧_tgt′
            # 再整理成(bs, head数 * d)
            cross_recon = torch.einsum(
                "bhs,hsd->bhd",
                cross_mem_address,
                self.cross_mem.view(
                    self.n_head,
                    self.n_slot // self.n_head,
                    self.dim_cross_mem,
                ),  # 公式(3) 𝑧_tgt′ = 𝐴src · 𝑴tgt
            ).reshape(B, self.n_head * self.dim_cross_mem)
            if self.predict_residual:    # default: False
                cross_recon += source
            if self.use_output_proj:    # default: False
                cross_recon = self.out_proj_cross(cross_recon)
        else:
            cross_recon = self.cross_branch(source)
        output["output_predict"] = cross_recon  # 𝑧_tgt′, (bs, self.n_head * self.dim_cross_mem)

        # Perform calculations related to target reconstruction
        if self.use_tar_recon:
            # Check if input projection and reconstruction type are memory-based
            if self.use_input_proj and self.recon_type == "memory":
                target_proj = self.in_proj_tar(target)
            else:   # default
                target_proj = target

            # Calculate target memory similarity using the specified address type
            if self.address_type == "cosine":
                target_mem_sim = self.target_key(
                    F.normalize(target_proj, dim=-1))
            else:  # 公式(5) 𝑞_tgt = 𝑤_tgt · 𝑧tgt。 忽略bs, 维度变化：self.dim_target_mem -> self.n_concept
                target_mem_sim = self.target_key(target_proj)   # 𝑞_tgt

            # Optionally visualize target memory similarity
            if self.cfg.TEST.VIS_MIDDLE:    # default: False
                output["vis_target_mem_sim"] = target_mem_sim

            # Apply softmax to target memory similarity multiplied by radius
            target_mem_address = self.softmax(self.radius * target_mem_sim)     # 公式(7) P_t

            # Calculate target reconstruction based on reconstruction type
            if self.recon_type == "memory":
                target_recon = torch.matmul(target_mem_address,
                                            self.target_mem)
                target_recon = self.out_proj_tar(target_recon)
            else:  # 得到𝑫tgt (𝑞tgt)。忽略bs, 维度变化： self.n_concept -> self.recon_dim
                target_recon = self.target_recon(target_mem_sim)    # 𝑫tgt (𝑞tgt)

            # Calculate the reconstruction loss for target
            loss_target_recon = self.reconstruction_loss(   # 公式(6) L_recon
                target_recon,   # (bsz, recon_dim)
                self.pool_feat(feat_target).view(target_recon.shape[0],     # y_tgt. feat_target.shape: [128, 2048, 7, 7]
                                                 -1).detach(),
                loss_type=self.recon_loss_type,
                average_dim=self.average_dim,
            )
            output["loss_target_recon"] = loss_target_recon     # L_recon

            # Calculate sparse loss for target memory if required
            if self.use_sparse:     # default: False
                if self.use_sparse_before:
                    # Calculate sparse loss using softmax of target memory similarity
                    loss_target_sparse = self.regularization_loss(
                        self.softmax(target_mem_sim), self.topk)
                else:
                    # Calculate sparse loss using target memory address
                    loss_target_sparse = self.regularization_loss(
                        target_mem_address, self.topk)
                output["loss_target_mem_sparse"] = loss_target_sparse

        # Perform calculations related to source reconstruction
        if self.use_src_recon:
            # Check if input projection and reconstruction type are memory-based
            if self.use_input_proj and self.recon_type == "memory":
                source_proj = self.in_proj_src(source)
            else:
                source_proj = source

            # Calculate source memory similarity using the specified address type
            if self.address_type == "cosine":
                source_mem_sim = self.source_key(
                    F.normalize(source_proj, dim=-1))
            else:
                source_mem_sim = self.source_key(source_proj)

            # Optionally visualize source memory similarity
            if self.cfg.TEST.VIS_MIDDLE:
                output["vis_source_mem_sim"] = source_mem_sim

            # Apply softmax to source memory similarity multiplied by radius
            source_mem_address = self.softmax(self.radius * source_mem_sim)

            # Calculate source reconstruction based on reconstruction type
            if self.recon_type == "memory":
                source_recon = torch.matmul(source_mem_address,
                                            self.source_mem)
                source_recon = self.out_proj_source(source_recon)
            else:
                source_recon = self.source_recon(source_mem_sim)

            # Calculate the reconstruction loss for source
            loss_source_recon = self.reconstruction_loss(
                source_recon,
                self.pool_feat(feat).view(B, -1).detach(),
                loss_type=self.recon_loss_type,
                average_dim=self.average_dim,
            )
            output["loss_source_recon"] = loss_source_recon

            # Calculate sparse loss for source memory if required
            if self.use_sparse:     # default: False
                if self.use_sparse_before:
                    # Calculate sparse loss using softmax of source memory similarity
                    loss_source_mem_sparse = self.regularization_loss(
                        self.softmax(source_mem_sim), self.topk)
                else:
                    # Calculate sparse loss using source memory address
                    loss_source_mem_sparse = self.regularization_loss(
                        source_mem_address, self.topk)
                output["loss_source_mem_sparse"] = loss_source_mem_sparse

        # Calculate contrastive losses if specified memory types are used
        if self.use_target_mem_contrastive:     # default:False
            output["loss_target_mem_contrastive"] = self.contrastive_loss(
                self.target_mem)

        if self.use_source_mem_contrastive:     # default:False
            output["loss_source_mem_contrastive"] = self.contrastive_loss(
                self.source_mem)

        if self.use_cross_mem_contrastive:     # default:False
            output["loss_cross_mem_contrastive"] = self.contrastive_loss(
                self.cross_mem)

        # Calculate visual concept alignment loss using KL divergence loss
        if self.use_align and self.use_src_recon and self.use_tar_recon:
            loss_kl = self.dino_loss_align(source_mem_sim, target_mem_sim)
            output["loss_kl"] = loss_kl

        if self.cfg.TEST.VIS_MIDDLE:  # default: False
            rand_bs = random.randint(0, cross_mem_sim.shape[0])

            # 𝑠𝑖𝑚(𝑴src, 𝑝src)   cross_mem_sim: [bs, head, slot]
            sample_sim_Mp = cross_mem_sim[rand_bs].view(cross_mem_sim.shape[2])

            # 𝑠𝑖𝑚(𝑴src, 𝑴src)
            Msrc = self.cross_key.weight_v     # 𝑴src: [dim, slot]
            sim_MM = torch.matmul(Msrc.transpose(0, 1), Msrc)

            # softmax(radius, 𝑠𝑖𝑚(𝑴src, 𝑝src))    cross_mem_address: [bs, head, slot]
            sample_soft_sim_Mp = cross_mem_address[rand_bs].view(cross_mem_address.shape[2])

            visualization(sample_sim_Mp, sim_MM, sample_soft_sim_Mp, self.cfg.TEST.VIS_PATH)

        if self.cfg.TEST.OUTPUT_SIM:
            # 计算batch内256个特征之间，C(256,2)个相似度，并平均，比较 Byol 和 memory_byol，
            # 看是否后者更高（前者大概约 0.6，后者可能 0.9，若如此则是后者坍塌
            # pred.shape: [128, 256]
            from itertools import combinations
            pred = output['output_predict']
            bsz = pred.shape[0]

            # 计算两两向量之间的余弦相似度
            combinations_list = list(combinations(range(bsz), 2))  # 生成所有索引的组合
            similarities = torch.nn.functional.cosine_similarity(pred[combinations_list[0][0]],
                                                                 pred[combinations_list[0][1]], dim=0).view(1, 1)
            for pair in combinations_list[1:]:
                sim = torch.nn.functional.cosine_similarity(pred[pair[0]], pred[pair[1]], dim=0).view(1, 1)
                similarities = torch.cat((similarities, sim), dim=0)
            avg_similaritiy = torch.sum(similarities, dim=0) / len(combinations_list)
            print("Avg Similaritiy: ", avg_similaritiy)

            output['avg_similaritiy'] = avg_similaritiy

        # Lconcept = alpha * Lrecon + beta * Lalign
        loss = 0
        if self.use_src_recon:
            loss += self.loss_src_alpha * output['loss_source_recon']
        if self.use_tar_recon:
            loss += self.loss_tgt_alpha * output['loss_target_recon']
        if self.use_align and self.use_src_recon and self.use_tar_recon:
            loss += self.loss_beta * output["loss_kl"]

        if self.predict_type == 'cosine':
            loss += self.mempred_loss(output['output_predict'], target)  # 公式(4) Lmempred
        elif self.predict_type == 'dino':
            loss += self.dino_loss_predict(output['output_predict'], target)  # 将Lmempred的计算换成DINO的形式

        return loss

class MLPHead(nn.Module):

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
        super(MLPHead, self).__init__()
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

    def forward(self, x):  # 前向传播函数
        if x.ndim == 5:  # 如果输入是5维张量（例如，来自卷积神经网络的特征图）
            x = x.permute((0, 2, 3, 4, 1))  # 重排维度顺序，使通道维度在最后
        if self.flatten:  # 如果需要在MLP前将输入展平
            x = x.reshape(-1, x.shape[-1])  # 将输入展平为二维张量

        return self.projection(x)  # 通过MLP进行前向传播并返回结果
