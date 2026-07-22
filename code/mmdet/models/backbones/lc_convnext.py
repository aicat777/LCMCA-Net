# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger
from ..builder import BACKBONES

class LayerNorm(nn.Module):
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class LSKA(nn.Module):
    def __init__(self, dim, stage_idx=0):
        super().__init__()
        
       
        if stage_idx == 0:
            k_size = 7
            d_rate = 2
            conv0_kernel = 3
            spatial_kernel = 3
            spatial_padding = 2
            
        elif stage_idx == 1:
            k_size = 7
            d_rate = 2
            conv0_kernel = 3
            spatial_kernel = 3
            spatial_padding = 2
            
        elif stage_idx == 2:
            k_size = 35
            d_rate = 3
            conv0_kernel = 5
            spatial_kernel = 11
            spatial_padding = 15
            
        else:
            k_size = 23
            d_rate = 3
            conv0_kernel = 5
            spatial_kernel = 7
            spatial_padding = 9
        self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, conv0_kernel), 
                                stride=1, padding=(0, (conv0_kernel-1)//2), groups=dim)
        self.conv0v = nn.Conv2d(dim, dim, kernel_size=(conv0_kernel, 1), 
                                stride=1, padding=((conv0_kernel-1)//2, 0), groups=dim)
        self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, spatial_kernel), 
                                        stride=1, padding=(0, spatial_padding), 
                                        groups=dim, dilation=d_rate)
        self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(spatial_kernel, 1), 
                                        stride=1, padding=(spatial_padding, 0), 
                                        groups=dim, dilation=d_rate)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.k_size = k_size
        self.d_rate = d_rate
        self.stage_idx = stage_idx
        self._init_weights()
    
    def _init_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups > 1:
                    nn.init.normal_(m.weight, std=0.03)
                else:
                    nn.init.normal_(m.weight, std=0.06)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn

class LSKA_ConvNeXt_Block(nn.Module):
    
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, stage_idx=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.norm_before_lska = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        self.lska = LSKA(dim, stage_idx)
        self.gamma_lska = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                        requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x_res1 = input + self.drop_path(x)
        x_res1_norm = self.norm_before_lska(x_res1)
        x_lska = self.lska(x_res1_norm)
        if self.gamma_lska is not None:
            x_lska = self.gamma_lska.view(-1, 1, 1) * x_lska
        x_output = x_res1 + self.drop_path(x_lska)
        return x_output

@BACKBONES.register_module()
class LSKAConvNeXtv1(nn.Module):
    
    def __init__(self, 
                 in_chans=3, 
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0., 
                 layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3],  
                 frozen_stages=-1,
                 norm_cfg=None,  
                 pretrained=None,
                 init_cfg=None):  
        super().__init__()
        
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                block = LSKA_ConvNeXt_Block(
                    dim=dims[i], 
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    stage_idx=i
                )
                stage_blocks.append(block)
            
            stage = nn.Sequential(*stage_blocks)
            self.stages.append(stage)
            cur += depths[i]
        
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.out_channels = [dims[i] for i in out_indices]
        self.apply(self._init_weights)
        self._freeze_stages()
        if isinstance(pretrained, str):
            self.init_weights(pretrained)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.downsample_layers[0].eval()
            for param in self.downsample_layers[0].parameters():
                param.requires_grad = False
        
        for i in range(1, min(self.frozen_stages + 1, 4)):
            if i < len(self.downsample_layers):
                self.downsample_layers[i].eval()
                for param in self.downsample_layers[i].parameters():
                    param.requires_grad = False
            
            if i-1 < len(self.stages):
                self.stages[i-1].eval()
                for param in self.stages[i-1].parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        
        outs = []
        
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                outs.append(x)
        
        
        return tuple(outs)