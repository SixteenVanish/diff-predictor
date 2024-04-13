import torch
import torch.nn as nn
from mmengine.config import ConfigDict

from .diff_utils import fill_missing_settings

class Decoder(nn.Module):
    def __init__(self, img_size=64, cfg=None):
        super(Decoder, self).__init__()
        
        self._initialize_config(img_size, cfg)
        print('decoder cfg:')
        print(self.cfg)
        
        if self.cfg.decoder_layer in ["neck", "predictor_mid", "predictor"]:
            self.fc = nn.Linear(self.cfg.input_dim, self.cfg.fc_channel * self.cfg.fc_size * self.cfg.fc_size)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.cfg.fc_channel, 32, kernel_size=1, stride=1, padding=0),    # [128, fc_size, fc_size]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(size=[self.cfg.map_size_list[0], self.cfg.map_size_list[0]], mode='bilinear', align_corners=True),  # [128, map_dim0, map_dim0]
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),     # [32, map_dim0, map_dim0]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(size=[self.cfg.map_size_list[1], self.cfg.map_size_list[1]], mode='bilinear', align_corners=True),  # [32, map_dim1, map_dim1]
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),      # [12, map_dim1, map_dim1]
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Upsample(size=[self.cfg.map_size_list[2], self.cfg.map_size_list[2]], mode='bilinear', align_corners=True),   # [12, map_dim2, map_dim2]
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),       # [3, map_dim2=64/224/448, map_dim2=64/224/448]
            nn.Sigmoid(),
        )

    def forward(self, features):        
        if self.cfg.decoder_layer in ["neck", "predictor_mid", "predictor"]:
            features = self.fc(features)
        features = features.view(-1, self.cfg.fc_channel, self.cfg.fc_size, self.cfg.fc_size)
        return self.decoder(features)
    
    def _initialize_default_params(self, img_size):
        map_size_list = []
        if img_size == 64:
            map_size_list = [16, 32, 64]
            fc_size = 8
            fc_channel = 512
        elif img_size == 224:
            map_size_list = [56, 112, 224]
            fc_size = 14
            fc_channel = 64
        elif img_size == 448:   # for CUB
            map_size_list = [28, 112, 448]
            fc_size = 7
            fc_channel = 512
        else:
            print("error img size", img_size)
        return map_size_list, fc_size, fc_channel
    
    def _initialize_config(self, img_size, cfg):
        map_size_list, fc_size, fc_channel = self._initialize_default_params(img_size)
        default_cfg = ConfigDict(
            decoder_layer="predictor_mid",
            input_dim=4096,
            
            map_size_list=map_size_list,
            fc_size=fc_size,
            fc_channel=fc_channel,
        ) 
        
        self.cfg = cfg
        fill_missing_settings(self.cfg, default_cfg)
    

class Decoder_v0(nn.Module):
    def __init__(self, img_size=64, cfg=None):
        super(Decoder, self).__init__()
        
        self._initialize_config(img_size, cfg)
        print('decoder cfg:')
        print(self.cfg)
        
        if self.cfg.decoder_layer in ["neck", "predictor_mid", "predictor"]:
            self.fc = nn.Linear(self.cfg.input_dim, self.cfg.fc_channel * self.cfg.fc_size * self.cfg.fc_size)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.cfg.fc_channel, 128, kernel_size=1, stride=1, padding=0),    # [128, fc_size, fc_size]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(size=[self.cfg.map_size_list[0], self.cfg.map_size_list[0]], mode='bilinear', align_corners=True),  # [128, map_dim0, map_dim0]
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),     # [32, map_dim0, map_dim0]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(size=[self.cfg.map_size_list[1], self.cfg.map_size_list[1]], mode='bilinear', align_corners=True),  # [32, map_dim1, map_dim1]
            nn.Conv2d(32, 12, kernel_size=3, stride=1, padding=1),      # [12, map_dim1, map_dim1]
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Upsample(size=[self.cfg.map_size_list[2], self.cfg.map_size_list[2]], mode='bilinear', align_corners=True),   # [12, map_dim2, map_dim2]
            nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1),       # [3, map_dim2=64/224/448, map_dim2=64/224/448]
            nn.Sigmoid(),
        )

    def forward(self, features):        
        if self.cfg.decoder_layer in ["neck", "predictor_mid", "predictor"]:
            features = self.fc(features)
        features = features.view(-1, self.cfg.fc_channel, self.cfg.fc_size, self.cfg.fc_size)
        return self.decoder(features)
    
    def _initialize_default_params(self, img_size):
        map_size_list = []
        if img_size == 64:
            map_size_list = [16, 32, 64]
            fc_size = 8
            fc_channel = 512
        elif img_size == 224:
            map_size_list = [56, 112, 224]
            fc_size = 14
            fc_channel = 64
        elif img_size == 448:   # for CUB
            map_size_list = [28, 112, 448]
            fc_size = 7
            fc_channel = 512
        else:
            print("error img size", img_size)
        return map_size_list, fc_size, fc_channel
    
    def _initialize_config(self, img_size, cfg):
        map_size_list, fc_size, fc_channel = self._initialize_default_params(img_size)
        default_cfg = ConfigDict(
            decoder_layer="predictor_mid",
            input_dim=4096,
            
            map_size_list=map_size_list,
            fc_size=fc_size,
            fc_channel=fc_channel,
        ) 
        
        self.cfg = cfg
        fill_missing_settings(self.cfg, default_cfg)
    
