from torch import nn
import torch
from typing import Optional,List
from net_block import UpLayer,DownLayer,AttnBlock2D,ResnetBlock
from torchsummary import summary
class DownBlock(nn.Module):
    def __init__(self, in_channels:int,out_channels:int,
                 layer_per_block:int=1,is_final:bool=False) -> None:
        super().__init__()
        
        self.downblock=nn.ModuleList([])
        for _ in range(layer_per_block):
            self.downblock.append(ResnetBlock(in_channels,out_channels))
            in_channels=out_channels
        
        self.downsample=None
        if not is_final:
            self.downsample=DownLayer(in_channels=in_channels,out_channels=out_channels)

        
    def forward(self,hidden_states:torch.FloatTensor):

        output_states=[]
        for i,block in enumerate(self.downblock):
            hidden_states=block(hidden_states)
            output_states.append(hidden_states)  #记录skip_connect 跳跃链接的特征
        
        cascade_feature=hidden_states  #记录用于级联特征
        if self.downsample is not None:
            hidden_states=self.downsample(hidden_states)
        return hidden_states,output_states,cascade_feature

class UpBlock(nn.Module):
    def __init__(self, in_channels:int,out_channels:int,skip_channels:List=[],
                 layer_per_block:int=1,is_final:bool=False) -> None:
        super().__init__()
        
        self.upsample=None
        if not is_final:
            self.upsample=UpLayer(in_channels=in_channels,out_channels=in_channels)

        self.upblock=nn.ModuleList([])
        for i in range(layer_per_block):
            in_channels+=skip_channels[i]
            self.upblock.append(ResnetBlock(in_channels,out_channels))
            in_channels=out_channels
        
    def forward(self,hidden_states:torch.FloatTensor,skip_feature:torch.FloatTensor):
        if self.upsample is not None:
            hidden_states=self.upsample(hidden_states)
        for i,block in enumerate(self.upblock):
            hidden_states=torch.cat([hidden_states,skip_feature[i]],dim=1)
            hidden_states=block(hidden_states)
        
        cascade_feature=hidden_states  #级联用于跳跃链接的特征
        
        return hidden_states,cascade_feature

#增加了交叉注意力的DownBlock
class AttnDownBlock(nn.Module):
    def __init__(self, in_channels:int,out_channels:int,
                 layer_per_block:int=1,is_final:bool=False,has_attn:bool=False,heads_num:int=8,head_dim:int=64) -> None:
        super().__init__()
        
        self.downblock=nn.ModuleList([])
        for _ in range(layer_per_block):
            block=nn.Sequential()
            block.append(ResnetBlock(in_channels,out_channels))
            if has_attn:
                assert heads_num!=0 and head_dim!=0,'heads_num or head_dim must >0'
                block.append(AttnBlock2D(query_dim=out_channels,head_dim=head_dim,heads_num=heads_num))
            self.downblock.append(block)
            in_channels=out_channels
        
        self.downsample=None
        if not is_final:
            self.downsample=DownLayer(in_channels=in_channels,out_channels=out_channels)

        
    def forward(self,hidden_states:torch.FloatTensor):

        output_states=[]
        for i,block in enumerate(self.downblock):
            hidden_states=block(hidden_states)
            output_states.append(hidden_states)  #记录skip_connect 跳跃链接的特征
        
        cascade_feature=hidden_states  #记录用于级联特征
        if self.downsample is not None:
            hidden_states=self.downsample(hidden_states)
        return hidden_states,output_states,cascade_feature


class AttnUpBlock(nn.Module):
    def __init__(self, in_channels:int,out_channels:int,skip_channels:List=[],
                 layer_per_block:int=1,is_final:bool=False,has_attn:bool=False,head_dim:int=64,heads_num:int=8) -> None:
        super().__init__()
        
        self.upsample=None
        if not is_final:
            self.upsample=UpLayer(in_channels=in_channels,out_channels=in_channels)

        self.upblock=nn.ModuleList([])
        for i in range(layer_per_block):
            in_channels+=skip_channels[i]
            block=nn.Sequential()
            block.append(ResnetBlock(in_channels,out_channels))
            if has_attn:
                assert heads_num!=0 and head_dim!=0,'heads_num or head_dim must >0'
                block.append(AttnBlock2D(query_dim=out_channels,head_dim=head_dim,heads_num=heads_num))
            self.upblock.append(block)
            in_channels=out_channels
        
    def forward(self,hidden_states:torch.FloatTensor,skip_feature:torch.FloatTensor):
        if self.upsample is not None:
            hidden_states=self.upsample(hidden_states)
        for i,block in enumerate(self.upblock):
            hidden_states=torch.cat([hidden_states,skip_feature[i]],dim=1)
            hidden_states=block(hidden_states)
        
        cascade_feature=hidden_states  #级联用于跳跃链接的特征
        
        return hidden_states,cascade_feature

class Unet(nn.Module):
    def __init__(self,
                in_channels=[64,128,256,512,1024],  #DownBlockd的输入通道数
                layer_per_block:int=1,
                mid_attention:bool=False,  #是否在mid_block中添加注意力机制
                mid_heads_num:int=8,
                mid_head_dim:int=64,
                cascade_index:int=2,  #从那一层开始记录级联特征
                ) -> None:
        
        super().__init__()
     

        self.cascade_index=cascade_index


        self.conv_in=nn.Conv2d(3,out_channels=in_channels[0],kernel_size=3,padding=1)

        #Down
        self.Down=nn.ModuleList([])
        self.skip_ch=[]
        for i in range(len(in_channels)):
            in_ch=in_channels[i]
            out_ch=in_channels[min(i+1,len(in_channels)-1)]
            self.skip_ch.append([out_ch]*layer_per_block)
            self.Down.append(DownBlock(in_channels=in_ch,out_channels=out_ch,layer_per_block=layer_per_block))

        #Mid
        self.Mid=nn.Sequential()
        self.Mid.append(ResnetBlock(in_channels=in_channels[-1],out_channels=in_channels[-1]))
        #如果需要在MidBlock中加attention
        if mid_attention:
            self.Mid.append(AttnBlock2D(query_dim=in_channels[-1],heads_num=mid_heads_num,head_dim=mid_head_dim))
        self.Mid.append(ResnetBlock(in_channels=in_channels[-1],out_channels=in_channels[-1]))

        #Up
        self.Up=nn.ModuleList([])
        in_channels=list(reversed(in_channels))
        self.skip_ch=list(reversed(self.skip_ch))
        for i in range(len(in_channels)):
            in_ch=in_channels[max(i-1,0)]
            out_ch=in_channels[i]
            self.Up.append(UpBlock(in_channels=in_ch,out_channels=out_ch,layer_per_block=layer_per_block,skip_channels=self.skip_ch[i]))
        
        self.conv_out=nn.Conv2d(in_channels[-1],3,kernel_size=3,stride=1,padding=1)

    def forward(self,hidden_states):
        self.skip_feature=[]   #保存用于跳跃链接的特征
        self.cascade_features=[]  #保存用于级联的特征

        hidden_states=self.conv_in(hidden_states)
        for i,block in enumerate(self.Down):
            hidden_states,output_states,cascade_feature=block(hidden_states)
            self.skip_feature.append(output_states)

            if i>=self.cascade_index:
                self.cascade_features.append(cascade_feature)
        
        hidden_states=self.Mid(hidden_states)
        self.cascade_features.append(hidden_states)

        self.skip_feature=list(reversed(self.skip_feature))
        for i,block in enumerate(self.Up):
            skip_feature=self.skip_feature[i]
            hidden_states,cascade_feature=block(hidden_states,skip_feature)

            if i<len(self.Up)-self.cascade_index:
                self.cascade_features.append(cascade_feature)
        
        hidden_states=self.conv_out(hidden_states)
        return hidden_states,self.cascade_features



class Attn_Unet(nn.Module):
    def __init__(self,
                in_channels=[64,128,256,512,1024],  #DownBlockd的输入通道数
                layer_per_block:int=1,
                down_attn:List[bool]=[False,False,True,True,True],   #定义在那些downblock中增加注意力机制，
                down_attn_head_dim:List[bool]=[0,0,64,64,64],        #定义注意力头的维度
                down_attn_heads_num:List[bool]=[0,0,8,8,8],        #定义注意力头数目
                mid_attention:bool=False,  #是否在mid_block中添加注意力机制
                mid_heads_num:int=8,
                mid_head_dim:int=64,
                up_attn:List[bool]=None,   #默认值None表示和Down部分保持一阵
                up_attn_head_dim:List[bool]=None,        #默认值None表示和Down部分保持一阵
                up_attn_heads_num:List[bool]=None,       #默认值None表示和Down部分保持一阵
                cascade_index:int=2,  #从那一层开始记录级联特征
                ) -> None:
        
        super().__init__()
     

        self.cascade_index=cascade_index


        self.conv_in=nn.Conv2d(3,out_channels=in_channels[0],kernel_size=3,padding=1)

        #Down
        self.Down=nn.ModuleList([])
        self.skip_ch=[]
        for i in range(len(in_channels)):
            in_ch=in_channels[i]
            out_ch=in_channels[min(i+1,len(in_channels)-1)]
            has_attn=down_attn[i]
            head_dim=down_attn_head_dim[i]
            heads_num=down_attn_heads_num[i]

            self.skip_ch.append([out_ch]*layer_per_block)
            self.Down.append(AttnDownBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    layer_per_block=layer_per_block,
                    has_attn=has_attn,
                    heads_num=heads_num,
                    head_dim=head_dim
                    ))

        #Mid
        self.Mid=nn.Sequential()
        self.Mid.append(ResnetBlock(in_channels=in_channels[-1],out_channels=in_channels[-1]))
        #如果需要在MidBlock中加attention
        if mid_attention:
            self.Mid.append(AttnBlock2D(query_dim=in_channels[-1],heads_num=mid_heads_num,head_dim=mid_head_dim))
        self.Mid.append(ResnetBlock(in_channels=in_channels[-1],out_channels=in_channels[-1]))

        #Up
        self.Up=nn.ModuleList([])
        in_channels=list(reversed(in_channels))
        self.skip_ch=list(reversed(self.skip_ch))
        
        if up_attn is None:
            up_attn=list(reversed(down_attn))
            up_attn_heads_num=list(reversed(down_attn_heads_num))
            up_attn_head_dim=list(reversed(down_attn_head_dim))

        for i in range(len(in_channels)):
            in_ch=in_channels[max(i-1,0)]
            out_ch=in_channels[i]
            has_attn=up_attn[i]
            head_dim=up_attn_head_dim[i]
            heads_num=up_attn_heads_num[i]
            self.Up.append(
                AttnUpBlock(in_channels=in_ch,out_channels=out_ch,
                        layer_per_block=layer_per_block,skip_channels=self.skip_ch[i],
                        has_attn=has_attn,head_dim=head_dim,heads_num=heads_num,
                        ))
        
        self.conv_out=nn.Conv2d(in_channels[-1],3,kernel_size=3,stride=1,padding=1)

    def forward(self,hidden_states):
        self.skip_feature=[]   #保存用于跳跃链接的特征
        self.cascade_features=[]  #保存用于级联的特征

        hidden_states=self.conv_in(hidden_states)
        for i,block in enumerate(self.Down):
            hidden_states,output_states,cascade_feature=block(hidden_states)
            self.skip_feature.append(output_states)

            if i>=self.cascade_index:
                self.cascade_features.append(cascade_feature)
        
        hidden_states=self.Mid(hidden_states)
        self.cascade_features.append(hidden_states)

        self.skip_feature=list(reversed(self.skip_feature))
        for i,block in enumerate(self.Up):
            skip_feature=self.skip_feature[i]
            hidden_states,cascade_feature=block(hidden_states,skip_feature)

            if i<len(self.Up)-self.cascade_index:
                self.cascade_features.append(cascade_feature)
        # print([k.shape for k in self.cascade_features])
        hidden_states=self.conv_out(hidden_states)
        return hidden_states,self.cascade_features


if __name__=='__main__':
    # unet=Unet()
    # x=torch.zeros((1,3,256,192))
    # unet(x)   #256*192的衣服图像记录的提取级联特征[torch.Size([1, 512, 64, 48]), torch.Size([1, 1024, 32, 24]), torch.Size([1, 1024, 16, 12]), 
    #                             # torch.Size([1, 1024, 8, 6]), 
    #                             # torch.Size([1, 1024, 16, 12]), torch.Size([1, 512, 32, 24]), torch.Size([1, 256, 64, 48])]
    unet=Attn_Unet()
    x=torch.zeros((1,3,256,192))
    unet(x)