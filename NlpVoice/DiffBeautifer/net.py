import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


#设置一些超参数
parser = argparse.ArgumentParser("DiffBeautifier")
parser.add_argument('--hidden_size', type=int, default=80)
parser.add_argument('--residual_layers', type=int, default=20)
parser.add_argument('--residual_channels', type=int, default=256)
parser.add_argument('--dilation_cycle_length', type=int, default=1)
args = parser.parse_args()


# 实现Mish激活函数
# used as class:
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))






class AttrDict(dict):
    def __init__(self, *args, **kwargs):  #不确定变量个数的tube和dict
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


#对一维卷积做了一些初始化的工作
def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)

#wavenet中间那块
class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(residual_channels, residual_channels)  #
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        # print("****")
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)  
        # print("*******")
        y = x + diffusion_step
      
        y=y[:,0]  #[B,1,residual_channel,T]->[B,residual_channel,T]

        # print("y",y.shape)
        # print("***",self.dilated_conv(y).shape,"&&&",conditioner.shape)
        y = self.dilated_conv(y) + conditioner  #将三个部分糅合在一起

        gate, filter = torch.chunk(y, 2, dim=1)  #在第一维上拆成两份
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffNet(nn.Module):
    def __init__(self, in_dims=80):
        super().__init__()
        self.params = params = AttrDict(
            # Model params
            encoder_hidden=args.hidden_size,    #256
            residual_layers=args.residual_layers,    #20 
            residual_channels=args.residual_channels,     #256
            dilation_cycle_length=args.dilation_cycle_length,    #1，这个不知道干啥用的
        )
        self.input_projection = Conv1d(in_dims, params.residual_channels, 1)  #x做的卷积
        self.diffusion_embedding = SinusoidalPosEmb(params.residual_channels)  #t做的位置编码
        dim = params.residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),     #Mish()是一个激活函数
            nn.Linear(dim * 4, dim)
        )   # t做的前向连接

        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.encoder_hidden, params.residual_channels, 2 ** (i % params.dilation_cycle_length))
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """

        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec[:, 0]   #[B,M,T]
        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.relu(x)


        diffusion_step = self.diffusion_embedding(diffusion_step)   #传入时间步数t，得到的diffusion_step维度为[B,1,params.residual_channels]
        diffusion_step = self.mlp(diffusion_step)


        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, 80, T]  六
        return x[:, None, :, :]    #[B,1,80,T]由回到了最初的一个输入的x的维度
        return x
