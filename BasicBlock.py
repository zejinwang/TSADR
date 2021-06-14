import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.init as weight_init
import torch
__all__ = ['MultipleBasicBlock','MultipleBasicBlock_4']
def conv3x3(in_planes, out_planes, dilation = 1, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=int(dilation*(3-1)/2), dilation=dilation, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dilation = 1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes,dilation, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # weight_init.xavier_normal()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class UNet(torch.nn.Module):
    def __init__(self, in_channel):
        super(UNet, self).__init__()
        self.in_channel = in_channel

        def Basic(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Upsample(channel):
             return torch.nn.Sequential(
                 torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                 torch.nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
                 torch.nn.ReLU(inplace=False)
             )
        self.moduleConv1 = Basic(in_channel, 128)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(128, 256)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(256, 512)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv3 = Basic(512, 512)
        self.moduleUpsample3 = Upsample(512)

        self.moduleDeconv2 = Basic(512, 256)
        self.moduleUpsample2 = Upsample(256)

        self.moduleDeconv1 = Basic(256, 128)
        self.moduleUpsample1 = Upsample(128)

        self.attn1 = Self_Attn(512)
        self.attn2 = Self_Attn(512)

    def forward(self, rfield):
        tensorJoin = rfield

        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorAttn1  = self.attn1(tensorPool3)
        tensorCombine = tensorPool3 + tensorAttn1

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorCombine = tensorUpsample3 + tensorConv3

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        tensorCombine = tensorUpsample2 + tensorConv2

        tensorDeconv1 = self.moduleDeconv1(tensorCombine)
        tensorUpsample1 = self.moduleUpsample1(tensorDeconv1)

        tensorCombine = tensorUpsample1 + tensorConv1

        return tensorCombine


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim, activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        #self.conv = nn.Sequential(*[
        #                            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
        #                            nn.ReLU()
        #                           ])
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
               x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out


class SparseSelfAttn(nn.Module):
    def __init__(self,in_dim=128,P_h=16, P_w=16):
        super(SparseSelfAttn,self).__init__()

        self.P_h = P_h
        self.P_w = P_w
        self.Self_Attn1 = Self_Attn(in_dim, 'relu')
        self.Self_Attn2 = Self_Attn(in_dim, 'relu')
    def forward(self,x):
        # x: input features with shape [N,C,H,W]
        # P h, P w: Number of partitions along H and W dimension
        N, C, H, W = x.size()
        Q_h, Q_w = H // self.P_h, W // self.P_w
        x = x.reshape(N, C, Q_h, self.P_h, Q_w, self.P_w)

        # Long range Attention
        x = x.permute(0, 3, 5, 1, 2, 4)
        x = x.reshape(N * self.P_h * self.P_w, C, Q_h, Q_w)
        x = self.Self_Attn1(x)
        x = x.reshape(N, self.P_h, self.P_w, C, Q_h, Q_w)

        # Short range Attention
        x = x.permute(0, 4, 5, 3, 1, 2)
        x = x.reshape(N * Q_h * Q_w, C, self.P_h, self.P_w)
        x = self.Self_Attn2(x)
        x = x.reshape(N, Q_h, Q_w, C, self.P_h, self.P_w)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(N, C, H, W)


class HierarStruct(nn.Module):
    def __init__(self,in_dim=128,R_h=8,R_w=8):
        super(HierarStruct,self).__init__()

        self.R_h = R_h
        self.R_w = R_w
        self.series1 = SparseSelfAttn(in_dim,P_h=4,P_w=4)
        self.SA = Self_Attn(in_dim, 'relu')
        #self.series2 = SparseSelfAttn(in_dim,P_h=16,P_w=16)
    def forward(self,x):
        # x: input features with shape [N,C,H,W]
        # P h, P w: Number of partitions along H and W dimension
        N, C, H, W = x.size()
        Q_h, Q_w = H // self.R_h, W // self.R_w
        x = x.reshape(N, C, Q_h, self.R_h, Q_w, self.R_w)

        # Long range Attention
        x = x.permute(0, 3, 5, 1, 2, 4)
        x = x.reshape(N * self.R_h * self.R_w, C, Q_h, Q_w)
        x = self.series1(x)
        x = x.reshape(N, self.R_h, self.R_w, C, Q_h, Q_w)

        # Short range Attention
        x = x.permute(0, 4, 5, 3, 1, 2)
        x = x.reshape(N * Q_h * Q_w, C, self.R_h, self.R_w)
        #x = self.series2(x)
        x = self.SA(x)
        x = x.reshape(N, Q_h, Q_w, C, self.R_h, self.R_w)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(N, C, H, W)


class MultipleBasicBlock(nn.Module):

    def __init__(self,input_feature,
                 block, num_blocks,
                 intermediate_feature = 64, dense = True):
        super(MultipleBasicBlock, self).__init__()
        self.dense = dense
        self.num_block = num_blocks
        self.intermediate_feature = intermediate_feature

        self.block1= nn.Sequential(*[
            nn.Conv2d(input_feature, intermediate_feature,
                      kernel_size=7, stride=1, padding=3, bias=True),
            nn.ReLU(inplace=True)
        ])

        # for i in range(1, num_blocks):
        self.block2 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=2 else None
        self.block3 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=3 else None
        self.block4 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=4 else None
        self.block5 = nn.Sequential(*[nn.Conv2d(intermediate_feature, 3 , (3, 3), 1, (1, 1))])

        #self.UNet = UNet(intermediate_feature)
        #self.attn1 = SparseSelfAttn()
        self.attn1 = HierarStruct()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x) if self.num_block>=2 else x
        x = self.block3(x) if self.num_block>=3 else x
        #x = self.UNet(x)
        x = self.attn1(x)
        x = self.block4(x) if self.num_block== 4 else x
        x = self.block5(x)
        return x

def MultipleBasicBlock_4(input_feature,intermediate_feature = 64):
    model = MultipleBasicBlock(input_feature,
                               BasicBlock,4 ,
                               intermediate_feature)
    return model


if __name__ == '__main__':

    # x= Variable(torch.randn(2,3,224,448))
    # model =    S2DF(BasicBlock,3,True)
    # y = model(x)
    model = MultipleBasicBlock(200, BasicBlock,4)
    model = BasicBlock(64,64,1)
    # y = model(x)
    exit(0)
