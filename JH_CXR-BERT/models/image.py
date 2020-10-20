#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torchvision

"Imgae encoder: generalize the final pooling layer to yield not one single output vector, but N seperate image embeddings, unlike in a regular cnn."
"using ResNet-152 with avg pooling over K x M grids in the image, yielding N = KM output vectors of 2048 dimensions each, for every image."
"Input images are resized, center-cropped at 224x224 and normalized, as is the standard for these networks"
"Summary: Resnet 152 -> final layer is constructed with AVG pooling which is (KxM) x 2048 from each image. " 

"The reason for this process: Within the multimodal bitransformer architecture, however, we can handle arbitrary lengths and are not committed to a particular number of inputs. "
"Thus, we generalize the final pooling layer to yield not one single output vector, but N separate image embeddings, unlike in a regular convolutional neural network."
"So, need to fix it to make the imgae embedding styles like ViT(patch wise image cutting) or Pixel bert(extract random samples in intermediate feature map).  "

class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args
        model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048


class ImageClf(nn.Module):
    def __init__(self, args):
        super(ImageClf, self).__init__()
        self.args = args
        self.img_encoder = ImageEncoder(args)
        self.clf = nn.Linear(args.img_hidden_sz * args.num_image_embeds, args.n_classes)

    def forward(self, x):
        x = self.img_encoder(x)
        x = torch.flatten(x, start_dim=1)
        out = self.clf(x)
        return out
