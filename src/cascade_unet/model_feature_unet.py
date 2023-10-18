from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from typing import List,Optional,Dict


#个人认为这里有几个点可以进行调整
#1.这里普通的卷积块更换成ResnetBlock会更好，残差块目前用的用的还是很普遍的
#2.BatchNorm建议换成GroupNorm,num_groups=32就行，这个目前在知乎上的看的，就是建议使用GroupNorm
#3.激活函数可以使用Relu，但是目前看人家的代码里用的Silu比较多，所以这里可以改可以不改，无所谓，我感觉这个也无高下之分吧
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

#从代码的可读性来看，你这里使用了一个up_conv,那最好对应的设置相应的下采样块，个人认为会更好。
#从你UNET里的内容来看，你是MaxPool做的下采样，但是我看一般diffusion的代码都是会用卷积做下采样比较多就是卷积核大小3，padding 1 ，stride为2，但是这个不强制
#BatchNorm建议换一下，然后激活函数随意
#核心一点还是建议写一个单独下采样块，这样个人感觉代码可读性会好一点，后续修改可能也简单一点
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

#这个attention 代码有问题，建议修改
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi

#那这里特征pooling，你要注意你要做的事情是将一个CHW的特征图变成C*1的特征向量
#这个有一个明显的代码错误，这里你是想用自适应的全局平均池化，但是你应该用nn.AdaptiveAvgPool2d(),而不是3D
class FeaturePooling(nn.Module):
    def __init__(self, output_size):
        super(FeaturePooling, self).__init__()
        self.adaptive_pooling = nn.AdaptiveAvgPool3d(output_size)

    def forward(self, input_features):
        output_features = self.adaptive_pooling(input_features)
        output_vector = output_features.view(output_features.size(0), -1)
        return output_vector

#1.代码结构需要调整，这种代码写起来简单，但是如果做修改的话，修改起来会很麻烦，做实验的可能东改改西改改，就忘记了之前的设置参数了
#2.

class UNet(nn.Module):

    def __init__(self, img_ch=3, output_ch=3):
        super(UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.featurepooling = FeaturePooling(output_size=(768, 1, 1))

    def forward(self, x):
        # encoding path
        feature_unet=[]
        x1 = self.Conv1(x)
        # print(x1.shape)
        pooling_x1=self.featurepooling(x1)
        feature_unet.append(pooling_x1)
        # print(pooling_x1.shape)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        # print(x2.shape)
        pooling_x2=self.featurepooling(x2)
        feature_unet.append(pooling_x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        # print(x3.shape)
        pooling_x3=self.featurepooling(x3)
        feature_unet.append(pooling_x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        # print(x4.shape)
        pooling_x4=self.featurepooling(x4)
        feature_unet.append(pooling_x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        # print(x5.shape)
        pooling_x5=self.featurepooling(x5)
        feature_unet.append(pooling_x5)
        
        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)   #注意力加的我没太看懂
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        print(d5.shape)
        pooling_d5=self.featurepooling(d5)
        feature_unet.append(pooling_d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        print(d4.shape)
        pooling_d4=self.featurepooling(d4)
        feature_unet.append(pooling_d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        print(d3.shape)
        pooling_d3=self.featurepooling(d3)
        feature_unet.append(pooling_d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        print(d2.shape)
        pooling_d2=self.featurepooling(d2)
        feature_unet.append(pooling_d2)

        
        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)
        print(d1.shape)
        pooling_d1=self.featurepooling(d1)
        feature_unet.append(pooling_d1)

        print(len(feature_unet))

        return d1,feature_unet
