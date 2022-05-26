import torch
from torch import nn

__all__ = ['NestedUNet']


class VGGBlock_3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class FFO(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #kernel的4放在哪个位置要跑起来看看
        self.conv_3_2 = nn.Conv3d(in_channels, out_channels, kernel_size=(4,1,1))#这里按道理说in_channels=out_channels

    def forward(self, x):
        out = self.conv_3_2(x)
        # out = torch.reshape(out, (out.size(dim=0),out.size(dim=1),1,512/out.size(dim=1)*32,512/out.size(dim=1)*32))

        return torch.squeeze(out, 2)


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        #pooling 1 的位置要跟着上面FFO的4变
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1,2,2))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_3d = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        self.ffo_32 = FFO(nb_filter[0],nb_filter[0])
        self.ffo_64 = FFO(nb_filter[1], nb_filter[1])
        self.ffo_128 = FFO(nb_filter[2], nb_filter[2])
        self.ffo_256 = FFO(nb_filter[3], nb_filter[3])
        self.ffo_512 = FFO(nb_filter[4], nb_filter[4])

        self.conv0_0 = VGGBlock_3D(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock_3D(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock_3D(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock_3D(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock_3D(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock_3D(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock_3D(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock_3D(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock_3D(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock_3D(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock_3D(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up_3d(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up_3d(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up_3d(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up_3d(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up_3d(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up_3d(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        #conv2D
        x3_1 = self.conv3_1(torch.cat([self.ffo_256(x3_0), self.up(self.ffo_512(x4_0))], 1))
        x2_2 = self.conv2_2(torch.cat([self.ffo_128(x2_0), self.ffo_128(x2_1), self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([self.ffo_64(x1_0), self.ffo_64(x1_1), self.ffo_64(x1_2), self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([self.ffo_32(x0_0), self.ffo_32(x0_1), self.ffo_32(x0_2), self.ffo_32(x0_3), self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

