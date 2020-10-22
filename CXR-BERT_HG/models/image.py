import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor

from PIL import Image


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        self.pool_func = (
            nn.AdaptiveMaxPool2d
            if args.img_embed_pool_type == 'max'
            else nn.AdaptiveAvgPool2d
        )

    def forward(self, x):
        # B x 3 x W x H -> B x 2048 x M x M -> B x 2048 x N -> B x N x 2048
        out = self.model(x)
        model_out = out.size()[-2:]

        W = int(model_out[0] / 2)
        H = int(model_out[1] / 2)
        pool = self.pool_func((W, H))
        out = pool(out)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()  # B x N x 2048
        #print('out_size:', out.size())  # torch.Size([32, 9, 2048])

        # random pixel sampling
        # TODO: At each iteration, randomly sample pixels
        num_range = out.size()[1]
        random_sampling = torch.randperm(num_range)[:self.args.num_image_embeds]
        random_sampling, _ = torch.sort(random_sampling)
        #print('random_sampling:', random_sampling)
        random_sample = out[:, random_sampling]
        #print('random_sample_size:', random_sample.size())
        return random_sample



class ImageEncoder_test():
    def __init__(self):
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        self.pool_func = (
            nn.AdaptiveMaxPool2d
        )

    def forward(self, x, num_image_embeds):
        # B x 3 x W x H -> B x 2048 x M x M -> B x 2048 x N -> B x N x 2048
        out = self.model(x)
        print('model_out:', out)
        model_out = out.size()[-2:]
        print(f'size :{model_out}')
        W = int(model_out[0]/2)
        H = int(model_out[1]/2)
        print(f'W: {W}, H:{H}')
        pool = self.pool_func((W,H))
        out = pool(out)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()  # B x N x 2048
        print(f'out : {out}')

        # random pixel sampling
        # TODO: At each iteration, randomly sample pixels
        num_range = out.size()[1]
        random_sampling = torch.randperm(num_range)[:num_image_embeds]
        random_sampling, _ = torch.sort(random_sampling)
        random_sample = out[:, random_sampling]
        print(f'random_sample: {random_sample}')
        return random_sample

# img = Image.open('C:\\Users\\HG_LEE\\PycharmProjects\\CXR-BERT2\\s50010747.jpg').convert("RGB")
# img = ToTensor()(img).unsqueeze(0)
# print(img.size())  # torch.Size([1, 3, 256, 256])
#

# img = torch.Tensor(1,3,256,256)
# enc = ImageEncoder_test()
# out = enc.forward(img, 10)
# print('sampled:', out.size())  # torch.Size([1, 10, 2048])
