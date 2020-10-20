#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from einops import rearrange


class Img_patch_embedding(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, channels = 3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.to_cls_token = nn.Identity()

    def forward(self, img, mask = None):
        p = self.patch_size
        print(f'patch size :{p}')
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        print(f'Image patch token is (batch, hxw , patch x patch x channel):{x.size()}')
        x = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        print(f'make the cls_token:{cls_tokens.size()}')
        x = torch.cat((cls_tokens, x), dim=1)
        print(f' concatenate the cls_tokens:{x.size()}')      
        x += self.pos_embedding
        print(f'position add to each token :{x.size()}')
        return x


if __name__ == "__main__":

    img_path = glob("/home/ubuntu/byol_/CXR_BYOL/vit_github/examples/3ch/train/*.jpg")

    img = Image.open(img_path[1])
    img = ToTensor()(img).unsqueeze(0)
    print(img.size())

    enc = Img_patch_embedding(
        image_size = 256,
        patch_size = 32,
        dim = 1024,
    )

    # mask = torch.ones(1, 8, 8).bool() # optional mask, designating which patch to attend to

    # enc = ViT()
    out = enc(img)
    print('shape of out :', out.shape)