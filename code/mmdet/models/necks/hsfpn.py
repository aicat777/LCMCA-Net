import math
import torch
import numpy as np
import torch.nn as nn
import torch_dct as DCT
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import ConvModule
from mmdet.models.builder import NECKS
from mmcv.runner import BaseModule, auto_fp16

__all__ =['HSFPN']

#------------------------------------------------------------------#
# Spatial Path of HFP
# Only p1&p2 use dct to extract high_frequency response
#------------------------------------------------------------------#
class DctSpatialInteraction(BaseModule):
    def __init__(self,
                in_channels,
                ratio,
                isdct = True,
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')):
        super(DctSpatialInteraction, self).__init__(init_cfg)
        self.ratio = ratio
        self.isdct = isdct # true when in p1&p2 # false when in p3&p4
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
            *[ConvModule(in_channels, 1, kernel_size=1, bias=False)]
        )

    def forward(self, x):
        _, _, h0, w0 = x.size()
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        idct = DCT.dct_2d(x, norm='ortho') 
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        weight = weight.view(1, h0, w0).expand_as(idct)             
        dct = idct * weight # filter out low-frequency features 
        dct_ = DCT.idct_2d(dct, norm='ortho') # generate spatial mask
        return x * dct_

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


#------------------------------------------------------------------#
# Channel Path of HFP
# Only p1&p2 use dct to extract high_frequency response
#------------------------------------------------------------------#
class DctChannelInteraction(BaseModule):
    def __init__(self,
                in_channels, 
                patch,
                ratio,
                isdct=True,
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')
                ):
        super(DctChannelInteraction, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        self.channel1x1 = nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=1, groups=32, bias=False)],
        )
        self.channel2x1 = nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=1, groups=32, bias=False)],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()
        if not self.isdct: # true when in p1&p2 # false when in p3&p4
            amaxp = F.adaptive_max_pool2d(x,  output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x,  output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))
            return x * torch.sigmoid(self.channel2x1(channel))

        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)             
        dct = idct * weight # filter out low-frequency features 
        dct_ = DCT.idct_2d(dct, norm='ortho') 

        amaxp = F.adaptive_max_pool2d(dct_,  output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_,  output_size=(self.h, self.w))       
        amaxp = torch.sum(self.relu(amaxp), dim=[2,3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2,3]).view(n, c, 1, 1)

        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(channel))
        
    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight  


#------------------------------------------------------------------#
# High Frequency Perception Module HFP
#------------------------------------------------------------------#
class HFP(BaseModule):
    def __init__(self, 
                in_channels,
                ratio,
                patch = (8,8),
                isdct = True,
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')):
        super(HFP, self).__init__(init_cfg)
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct = isdct) 
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct = isdct)
        self.out =  nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels)]
            )
    def forward(self, x):
        spatial = self.spatial(x) # output of spatial path
        channel = self.channel(x) # output of channel path
        return self.out(spatial + channel)


#------------------------------------------------------------------#
# Spatial Dependency Perception Module SDP
#------------------------------------------------------------------#
class SDP(BaseModule):
    def __init__(self,
                dim=256,
                inter_dim=None,
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SDP, self).__init__(init_cfg)
        self.inter_dim=inter_dim
        if self.inter_dim == None:
            self.inter_dim = dim
        self.conv_q = nn.Sequential(*[ConvModule(dim, self.inter_dim, 1, padding=0, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.conv_k = nn.Sequential(*[ConvModule(dim, self.inter_dim, 1, padding=0, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x_low, x_high, patch_size):
        b_, _, h_, w_ = x_low.size()
        q = rearrange(self.conv_q(x_low), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
        q = q.transpose(1,2) # 1,4096,128
        k = rearrange(self.conv_k(x_high), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
        attn = torch.matmul(q, k) # 1, 4096, 1024
        attn = attn / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)
        v = k.transpose(1,2)# 1, 1024, 128
        output = torch.matmul(attn,v)# 1, 4096, 128
        output = rearrange(output.transpose(1, 2).contiguous(), '(b h w) c (p1 p2) -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=h_//patch_size[0], w=w_//patch_size[1])
        return output + x_low


#------------------------------------------------------------------#
# HSFPN (适配start_level=1，去掉传统FPN融合)
#------------------------------------------------------------------#
@NECKS.register_module()
class HSFPN(BaseModule):
    def __init__(self,
                in_channels,
                out_channels,
                num_outs,
                ratio = (0.25, 0.25),
                start_level=1,  # 默认从C3开始
                end_level=-1,
                add_extra_convs=False,
                relu_before_extra_convs=False,
                no_norm_on_lateral=False,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None,
                upsample_cfg=dict(mode='nearest'),
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')):
        super(HSFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # 构建侧边连接
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        def interpolate(input):
            up_mode = 'nearest'
            return F.interpolate(input, scale_factor=2, mode='nearest', 
                               align_corners=False if up_mode == 'bilinear' else None)
        self.fpn_upsample = interpolate 
        
        # 根据start_level=1进行适配：输入是C2,C3,C4,C5（4层），但从C3开始处理
        # laterals[0]对应C3, laterals[1]对应C4, laterals[2]对应C5
        # 我们需要3个HFP模块和2个SDP模块
        
        # 创建HFP模块
        self.SelfAttn_p5 = HFP(out_channels, ratio=None, isdct=False)  # 对应C5
        self.SelfAttn_p4 = HFP(out_channels, ratio=None, isdct=False)  # 对应C4
        self.SelfAttn_p3 = HFP(out_channels, ratio=ratio, patch=(8, 8), isdct=True)  # 对应C3
        
        # 创建SDP模块
        self.CrossAtten_p5_p4 = SDP(dim=out_channels)  # C5 -> C4
        self.CrossAtten_p4_p3 = SDP(dim=out_channels)  # C4 -> C3

        # 添加额外的卷积层以满足num_outs=5
        extra_levels = num_outs - (self.backbone_end_level - self.start_level)
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    # 使用最后一个输入特征层（C5）来构建额外层
                    extra_in_channels = self.in_channels[-1]
                else:
                    extra_in_channels = out_channels
                
                extra_fpn_conv = ConvModule(
                    extra_in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        
        # 构建侧边连接特征（从start_level开始）
        laterals = []
        for i in range(self.start_level, self.backbone_end_level):
            lateral_idx = i - self.start_level
            lateral = self.lateral_convs[lateral_idx](inputs[i])
            laterals.append(lateral)
        
        # HS-FPN核心流程（纯HFP+SDP，去掉传统FPN融合）
        # laterals[0]: C3, laterals[1]: C4, laterals[2]: C5
        
        # 获取C4的尺寸用于SDP
        _, _, h, w = laterals[1].size()
        
        # 处理C5（最高层）
        laterals[2] = self.SelfAttn_p5(laterals[2])
        
        # C5 -> C4的SDP
        laterals[1] = self.CrossAtten_p5_p4(
            self.SelfAttn_p4(laterals[1]), 
            self.fpn_upsample(laterals[2]), 
            [h, w]
        )
        
        # C4 -> C3的SDP
        laterals[0] = self.CrossAtten_p4_p3(
            self.SelfAttn_p3(laterals[0]), 
            self.fpn_upsample(laterals[1]), 
            [h, w]
        )
        
        # 直接应用3x3卷积得到输出（去掉传统FPN融合）
        used_backbone_levels = len(laterals)
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        
        # 添加额外输出层以满足num_outs=5
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                # 使用最大池化增加层级
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    # 使用最后一个输入特征层（C5）构建额外层
                    extra_source = inputs[-1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                
                # 第一个额外层
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                
                # 剩余的额外层
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        
        return tuple(outs)