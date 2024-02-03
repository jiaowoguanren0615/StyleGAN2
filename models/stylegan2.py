import os
import sys
import math
import fire
import json

from tqdm import tqdm
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing
from contextlib import contextmanager, ExitStack

import numpy as np

import torch
from torch import nn, einsum
from torch.utils import data
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange, repeat
from kornia.filters import filter2d

import torchvision
from torchvision import transforms
from datasets.data_augment import DiffAugment
from datasets import exists, AugWrapper

from .discriminator_model import Discriminator, leaky_relu
from .generate_model import Generator

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False



class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.reshape(x.shape[0], -1)
#
#
# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#     def forward(self, x):
#         return self.fn(x) + x
#
# class ChanNorm(nn.Module):
#     def __init__(self, dim, eps = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
#         self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))
#
#     def forward(self, x):
#         var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
#         mean = torch.mean(x, dim = 1, keepdim = True)
#         return (x - mean) / (var + self.eps).sqrt() * self.g + self.b
#
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = ChanNorm(dim)
#
#     def forward(self, x):
#         return self.fn(self.norm(x))
#
# class PermuteToFrom(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#     def forward(self, x):
#         x = x.permute(0, 2, 3, 1)
#         out, *_, loss = self.fn(x)
#         out = out.permute(0, 3, 1, 2)
#         return out, loss
#
# class Blur(nn.Module):
#     def __init__(self):
#         super().__init__()
#         f = torch.Tensor([1, 2, 1])
#         self.register_buffer('f', f)
#     def forward(self, x):
#         f = self.f
#         f = f[None, None, :] * f [None, :, None]
#         return filter2d(x, f, normalized=True)
#
# # attention
#
# class DepthWiseConv2d(nn.Module):
#     def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
#             nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
#         )
#     def forward(self, x):
#         return self.net(x)
#
# class LinearAttention(nn.Module):
#     def __init__(self, dim, dim_head = 64, heads = 8):
#         super().__init__()
#         self.scale = dim_head ** -0.5
#         self.heads = heads
#         inner_dim = dim_head * heads
#
#         self.nonlin = nn.GELU()
#         self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
#         self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)
#         self.to_out = nn.Conv2d(inner_dim, dim, 1)
#
#     def forward(self, fmap):
#         h, x, y = self.heads, *fmap.shape[-2:]
#         q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
#         q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))
#
#         q = q.softmax(dim = -1)
#         k = k.softmax(dim = -2)
#
#         q = q * self.scale
#
#         context = einsum('b n d, b n e -> b d e', k, v)
#         out = einsum('b n d, b d e -> b n e', q, context)
#         out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)
#
#         out = self.nonlin(out)
#         return self.to_out(out)
#
# # one layer of self-attention and feedforward, for images
#
# attn_and_ff = lambda chan: nn.Sequential(*[
#     Residual(PreNorm(chan, LinearAttention(chan))),
#     Residual(PreNorm(chan, nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
# ])


class StyleGAN2(nn.Module):
    def __init__(self, image_size, latent_dim = 512, fmap_max = 512, style_depth = 8, network_capacity = 16, transparent = False, fp16 = False, cl_reg = False, steps = 1, lr = 1e-4, ttur_mult = 2, fq_layers = [], fq_dict_size = 256, attn_layers = [], no_const = False, lr_mlp = 0.1, rank = 0):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)

        self.S = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mlp)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const, fmap_max = fmap_max)
        self.D = Discriminator(image_size, network_capacity, fq_layers = fq_layers, fq_dict_size = fq_dict_size, attn_layers = attn_layers, transparent = transparent, fmap_max = fmap_max)

        self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mlp)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const)

        self.D_cl = None

        if cl_reg:
            from contrastive_learner import ContrastiveLearner
            # experimental contrastive loss discriminator regularization
            assert not transparent, 'contrastive loss regularization does not work with transparent images yet'
            self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten')

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size)

        # turn off grad for exponential moving averages
        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        # init optimizers
        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = Adam(generator_params, lr = self.lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr = self.lr * ttur_mult, betas=(0.5, 0.9))

        # init weights
        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda(rank)

        # startup apex mixed precision
        self.fp16 = fp16

        if fp16:
            (self.S, self.G, self.D, self.SE, self.GE), (self.G_opt, self.D_opt) = amp.initialize([self.S, self.G, self.D, self.SE, self.GE], [self.G_opt, self.D_opt], opt_level='O1', num_losses=3)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x
