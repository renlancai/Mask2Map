import torch
import torch.nn as nn
import torch.nn.init as init

class LidarConvFeatureExtractor(nn.Module):
    def __init__(self, 
                 in_channels=64, 
                 base_dim=64,
                 output_dim=512,
                 num_blocks=[2,2,2,2], 
                 use_bn=True,
                 init_type='kaiming'):
        """
        Args:
            in_channels: 输入通道数 (C)
            base_dim: 基础特征维度 
            output_dim: 最终输出维度 (dim)
            num_blocks: 每个stage的卷积块数量 [2,2,2,2]类似ResNet18
            use_bn: 是否使用BatchNorm
            init_type: 参数初始化方式
        """
        super().__init__()
        
        # 网络配置参数
        self.stage_config = [
            {'channels': base_dim,   'stride': 1},    # stage1
            {'channels': base_dim*2, 'stride': 2},    # stage2
            {'channels': base_dim*4, 'stride': 2},    # stage3
            {'channels': base_dim*8, 'stride': 2},    # stage4
        ]
        
        # 初始卷积层
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 
                     kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(base_dim) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        # 构建卷积stage
        self.stages = nn.ModuleList()
        in_ch = base_dim
        for idx, (cfg, n_blocks) in enumerate(zip(self.stage_config, num_blocks)):
            stage = self._make_stage(
                in_channels=in_ch,
                out_channels=cfg['channels'],
                num_blocks=n_blocks,
                stride=cfg['stride'],
                use_bn=use_bn
            )
            self.stages.append(stage)
            in_ch = cfg['channels']
        
        # 最终输出卷积
        self.final_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=output_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # 参数初始化
        self._initialize_weights(init_type)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, use_bn):
        """构建单个stage"""
        blocks = []
        # 首层处理stride变化
        blocks.append(
            ConvBlock(in_channels, out_channels, 
                     stride=stride, use_bn=use_bn)
        )
        # 后续层保持尺寸不变
        for _ in range(1, num_blocks):
            blocks.append(
                ConvBlock(out_channels, out_channels,
                         stride=1, use_bn=use_bn)
            )
        return nn.Sequential(*blocks)
    
    def _initialize_weights(self, init_type):
        """参数初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入形状: (B, C, H, W) = (B, 64, 450, 900)
        x = self.input_conv(x)
        for stage in self.stages:
            x = stage(x)
        x = self.final_conv(x)  # 输出形状: (B, dim, H, W)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_bn=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                     kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)
