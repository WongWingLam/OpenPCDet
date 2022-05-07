import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        #
        # coarse regression
        #
        self.block1 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(
                num_filters[0], num_filters[0], 3, stride=layer_strides[0]),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add_module("temp",
                nn.Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add_module("temp", nn.BatchNorm2d(num_filters[0]))
            self.block1.add_module("temp", nn.ReLU())

        self.deconv1 = nn.Sequential( # up-sampling
            nn.ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            nn.BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(
                num_filters[0],
                num_filters[1],
                3,
                stride=layer_strides[1]),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add_module("temp",
                nn.Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add_module("temp", nn.BatchNorm2d(num_filters[1]))
            self.block2.add_module("temp", nn.ReLU())

        self.deconv2 = nn.Sequential( # up-sampling
            nn.ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            nn.BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add_module("temp",
                nn.Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add_module("temp", nn.BatchNorm2d(num_filters[2]))
            self.block3.add_module("temp", nn.ReLU())

        self.deconv3 = nn.Sequential( # up-sampling
            nn.ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            nn.BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )

        #    
        # fine regression
        #

        # B1_2 and B1_3 are obtained by two downsampling operations performing on B1
        self.block1_dec2x = nn.MaxPool2d(kernel_size=2) # C=64
        self.block1_dec4x = nn.MaxPool2d(kernel_size=4) # C=64

        # an up-sampling and a down-sampling are operated on B2 to obtain {B2_1, B2, B2_3}
        self.block2_dec2x = nn.MaxPool2d(kernel_size=2) # C=128
        self.block2_inc2x = nn.ConvTranspose2d(num_filters[1],num_filters[0]//2,upsample_strides[1],stride=upsample_strides[1]) # C=32

        # two up-sampling operations to obtain {B3_1, B3_2} is ob-tained by tbased on B3
        self.block3_inc2x = nn.ConvTranspose2d(num_filters[2],num_filters[1]//2,upsample_strides[1],stride=upsample_strides[1]) # C=64
        self.block3_inc4x = nn.ConvTranspose2d(num_filters[2],num_filters[0]//2,upsample_strides[2],stride=upsample_strides[2]) # C=32

        # this are perform after the concatenating B1^i , B2^i and B3^i for i = 1, 2, 3, respectively
        self.fusion_block1 = nn.Conv2d(num_filters[0]+num_filters[0]//2+num_filters[0]//2, num_filters[0], 1)
        self.fusion_block2 = nn.Conv2d(num_filters[0]+num_filters[1]+num_filters[1]//2, num_filters[1], 1)
        self.fusion_block3 = nn.Conv2d(num_filters[0]+num_filters[1]+num_filters[2], num_filters[2], 1)
        
        # a series of convolution operations followed by an up- sampling layer are executed
        # , which results in the feature maps UP = {UP1,UP2,UP3},
        C_reduce = 32
        
        # for B3 then refine_up3 to upsampling to obtain UP3 
        self.RF1 = nn.Sequential(  # 3*3
            nn.Conv2d(num_filters[2], C_reduce, kernel_size=1, stride=1),
            nn.BatchNorm2d(C_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_reduce, num_filters[2], kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True),
        )

        # for B2 then refine_up2 to upsampling to obtain UP2 
        self.RF2 = nn.Sequential(  # 5*5
            nn.Conv2d(num_filters[1], C_reduce, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_reduce, num_filters[1], kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True),
        )

        # for B1 then refine_up1 to upsampling to obtain UP1 
        self.RF3 = nn.Sequential(  # 7*7
            nn.Conv2d(num_filters[0], C_reduce, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_reduce, C_reduce, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_reduce, num_filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=True),
        )

        self.refine_up1 = nn.Sequential(
            nn.ConvTranspose2d(num_filters[0],num_upsample_filters[0], upsample_strides[0],stride=upsample_strides[0]),
            nn.BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.refine_up2 = nn.Sequential(
            nn.ConvTranspose2d(num_filters[1],num_upsample_filters[1],upsample_strides[1],stride=upsample_strides[1]),
            nn.BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.refine_up3 = nn.Sequential(
            nn.ConvTranspose2d(num_filters[2],num_upsample_filters[2],upsample_strides[2], stride=upsample_strides[2]),
            nn.BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )

        # the 1 × 1 convolution to transform FC into FB
        self.bottle_conv = nn.Conv2d(sum(num_upsample_filters), sum(num_upsample_filters)//3, 1)

        # convolution layers after element-wise sum of UP and FB
        self.concat_conv1 = nn.Conv2d(num_filters[1], num_filters[1], kernel_size=3, padding=1)
        self.concat_conv2 = nn.Conv2d(num_filters[1], num_filters[1], kernel_size=3, padding=1)
        self.concat_conv3 = nn.Conv2d(num_filters[1], num_filters[1], kernel_size=3, padding=1)
        
        c_in = sum(num_upsample_filters)
        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']

        #
        # coarse regression
        #

        # Block1
        b1 = self.block1(spatial_features)
        upSample1 = self.deconv1(b1)

        # Block2
        b2 = self.block2(b1)
        upSample2 = self.deconv2(b2)

        # Block3
        b3 = self.block3(b2)
        upSample3 = self.deconv3(b3)
        coarse_feat = torch.cat([upSample1, upSample2, upSample3], dim=1)

        #
        # fine regression
        #
        b2_inc2x = self.block2_inc2x(b2) # B2^1
        b3_inc4x = self.block3_inc4x(b3) # B3^1

        b1_dec2x = self.block1_dec2x(b1) # B1^2
        b3_inc2x = self.block3_inc2x(b3) # B3^2

        b1_dec4x = self.block1_dec4x(b1) # B1^3
        b2_dec2x = self.block2_dec2x(b2) # B2^3

        # concat B1^1, B2^1, B3^1
        concat_block1 = torch.cat([b1,b2_inc2x,b3_inc4x], dim=1)
        fusion_block1 = self.fusion_block1(concat_block1)

        # concat B1^2, B2^2, B3^2
        concat_block2 = torch.cat([b1_dec2x,b2,b3_inc2x], dim=1)
        fusion_block2 = self.fusion_block2(concat_block2)

        # concat B1^3, B2^3, B3^3
        concat_block3 = torch.cat([b1_dec4x,b2_dec2x,b3], dim=1)
        fusion_block3 = self.fusion_block3(concat_block3)

        # obtain feature maps UP = {UP1,UP2,UP3}
        refine_up1 = self.RF3(fusion_block1)
        refine_up1 = self.refine_up1(refine_up1)
        refine_up2 = self.RF2(fusion_block2)
        refine_up2 = self.refine_up2(refine_up2)
        refine_up3 = self.RF1(fusion_block3)
        refine_up3 = self.refine_up3(refine_up3)

        # F_C -> F_B
        F_b = self.bottle_conv(coarse_feat)

        # Element-wise Sum of UP and F_B
        branch1_sum_wise = refine_up1 + F_b
        branch2_sum_wise = refine_up2 + F_b
        branch3_sum_wise = refine_up3 + F_b

        concat_conv1 = self.concat_conv1(branch1_sum_wise)
        concat_conv2 = self.concat_conv2(branch2_sum_wise)
        concat_conv3 = self.concat_conv3(branch3_sum_wise)
        fine_feat = torch.cat([concat_conv1,concat_conv2,concat_conv3], dim=1)

        # prediction will be generated in DensedHead by only fine feature
        data_dict['spatial_features_2d'] = fine_feat

        return data_dict
