from torch import nn,einsum
from torch.nn import functional as F
from typing import List,Optional,Dict
import torch
from einops import rearrange
import xformers


class ResnetBlock(nn.Module):
    def __init__(self, in_channels:int=None,out_channels:int=None,conv_shortcut:bool=False,dropout:float=0.0,num_groups:int=32) -> None:
        super().__init__()
        
        out_channels=out_channels if out_channels is not None else in_channels
        self.use_conv_shortcut=conv_shortcut

        self.norm1=nn.GroupNorm(num_groups=num_groups,num_channels=in_channels)
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)

        self.norm2=nn.GroupNorm(num_groups=num_groups,num_channels=out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)

        self.conv_shortcut=None
        if in_channels !=out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
            else:
                self.conv_shortcut=nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0,stride=1)
        
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        h=x
        h=self.norm1(h)
        h=F.silu(h)
        h=self.conv1(h)

        h=self.norm2(h)
        h=F.silu(h)
        h=self.dropout(h)
        h=self.conv2(h)

        if self.conv_shortcut is not None:
            x=self.conv_shortcut(x)
        
        return x+h

class DownLayer(nn.Module):
    def __init__(self,in_channels:Optional[int]=None, use_conv:bool=True,out_channels:Optional[int]=None,stride:int=2,padding:Optional[int]=1,) -> None:
        super().__init__()
        
        if use_conv:
            assert in_channels is not None,'if you want to use conv to do downsample,you should give in_channels '
            out_channels=out_channels if out_channels is not None else in_channels
            self.down=nn.Conv2d(in_channels,out_channels,3,stride,padding)
        else:
            self.down=nn.AvgPool2d(kernel_size=stride,padding=padding)
    
    def forward(self,x):
        return self.down(x)

class UpLayer(nn.Module):
    def __init__(self, in_channels:Optional[int]=None,use_conv:bool=False,out_channels:Optional[int]=None) -> None:
        super().__init__()

        self.conv=None
        if use_conv:
            assert in_channels is not None,'if you want to use conv to do upsample,you should give in_channels '
            out_channels=out_channels if out_channels is not None else in_channels
            self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        
    def forward(self,x,scale_factor:int=2.0):
        x=F.interpolate(x,scale_factor=scale_factor,mode='nearest')
        if self.conv is not None:
            x=self.conv(x)
        
        return x

#这个是注意力只能用在BCHW这种二维数据上
class AttnBlock2D(nn.Module):
    def __init__(self, query_dim:int,cross_dim:int=None,heads_num:int=8,head_dim:int=64,dropout:float=0.0) -> None:
        super().__init__() 

        self.heads_num=heads_num
        self.head_dim=head_dim

        inner_dim=heads_num*head_dim
        cross_dim=query_dim if cross_dim is None else cross_dim

        self.to_q=nn.Conv2d(query_dim,inner_dim,kernel_size=1)
        self.to_k=nn.Conv2d(cross_dim,inner_dim,kernel_size=1)
        self.to_v=nn.Conv2d(cross_dim,inner_dim,kernel_size=1)

        self.to_out=nn.Sequential(
            nn.Conv2d(inner_dim,query_dim,kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self,self_feature:torch.Tensor,cross_feature:Optional[torch.Tensor]=None):
        residual=self_feature

        #如果这里的attention是自注意力机制，那这里的cross_feature应该为None
        cross_feature=self_feature if cross_feature is None else cross_feature  

        query=self.to_q(self_feature)
        key=self.to_k(cross_feature)
        value=self.to_v(cross_feature)

        
        query=rearrange(query,'b (heads_num ,heads_dim) h w -> b heads_num  (h w) heads_dim  ',heads_num=self.heads_num)
        key=rearrange(key,'b (heads_num ,heads_dim) h w -> b heads_num (h w) heads_dim  ',heads_num=self.heads_num)
        value=rearrange(value,'b (heads_num ,heads_dim) h w -> b heads_num (h w) heads_dim ',heads_num=self.heads_num)

        attn_map=einsum('b heads_num a c,b heads_num d c -> b heads_num a d',query,key)
        attn_scores=torch.softmax(attn_map/(self.head_dim**0.5),dim=-1)

        attn_result=einsum('b heads_num a d -> b heads_num d c -> b heads_num a c',attn_scores,value)

        attn_result=self.to_out(attn_result)

        return attn_result+residual

        
#使用xformer优化注意力计算效率和显存占用
class xformer_AttnBlock2D(nn.Module):
    def __init__(self, query_dim:int,cross_dim:int=None,heads_num:int=8,head_dim:int=64,dropout:float=0.0) -> None:
        super().__init__() 

        self.heads_num=heads_num
        self.head_dim=head_dim

        inner_dim=heads_num*head_dim
        cross_dim=query_dim if cross_dim is None else cross_dim

        self.to_q=nn.Conv2d(query_dim,inner_dim,kernel_size=1)
        self.to_k=nn.Conv2d(cross_dim,inner_dim,kernel_size=1)
        self.to_v=nn.Conv2d(cross_dim,inner_dim,kernel_size=1)

        self.to_out=nn.Sequential(
            nn.Conv2d(inner_dim,query_dim,kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self,self_feature:torch.Tensor,cross_feature:Optional[torch.Tensor]=None):
        residual=self_feature

        #如果这里的attention是自注意力机制，那这里的cross_feature应该为None
        cross_feature=self_feature if cross_feature is None else cross_feature  

        query=self.to_q(self_feature)
        key=self.to_k(cross_feature)
        value=self.to_v(cross_feature)

        
        query=rearrange(query,'b (heads_num ,heads_dim) h w -> b heads_num (h w) heads_dim  ',heads_num=self.heads_num)
        key=rearrange(key,'b (heads_num ,heads_dim) h w -> b heads_num (h w) heads_dim  ',heads_num=self.heads_num)
        value=rearrange(value,'b (heads_num ,heads_dim) h w -> b heads_num (h w) heads_dim ',heads_num=self.heads_num)

        attn_result=xformers.ops.memory_efficient_attention(query,key,value)
        
        attn_result=self.to_out(attn_result)

        return attn_result+residual



