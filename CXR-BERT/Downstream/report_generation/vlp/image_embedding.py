import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from glob import glob
# import torchxrayvision as xrv


class fully_sampling(nn.Module):
    def __init__(self):
        super(fully_sampling, self).__init__()
        # model = xrv.models.DenseNet(weights="mimic_ch")
        # model = torchvision.models.resnext50_32x4d(pretrained=True)
        model = torchvision.models.resnet50(pretrained=True)

        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
        )
        self.pool = pool_func((3, 1))

    def forward(self, x):
        # print("Size of x: ", x.size())
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.model(x)
        # print("out", out.size())
        # out = self.pool(self.model(x)) #out torch.Size([100, 2048, 3, 1])
        out = torch.flatten(out, start_dim=2) #out torch.Size([100, 2048, 3])
        out = out.transpose(1, 2).contiguous() #out torch.Size([100, 3, 2048])
        # print("out", out.size())
        # input("STop!!!")

        return out  # BxNx2048

class random_sampling(nn.Module):
    def __init__(self):
        super(random_sampling, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        # B x 3 x W x H -> B x 2048 x M x M -> B x 2048 x N -> B x N x 2048
        out = self.model(x)  # 512x512: torch.Size([16, 2048, 16, 16])
        # print('out_size:', out.size())  # torch.Size([32, 9, 2048])

        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()  # B x N x 2048

        # random pixel sampling
        # TODO: At each iteration, randomly sample pixels
        num_range = out.size()[1]
        random_sampling = torch.randperm(num_range)[:100]
        random_sampling, _ = torch.sort(random_sampling)
        # print('random_sampling:', random_sampling)
        random_sample = out[:, random_sampling]
        # print('random_sample_size:', random_sample.size())

        return random_sample

class Img_patch_embedding(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, channels = 3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

    def forward(self, img, mask = None):
        img_size = img.size()
        # print("\n")
        # print(f'img_size :{img_size}')
        p = self.patch_size
        # print(f'patch size :{p}')
        out = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        # print(f'Image patch token is (batch, hxw , patch x patch x channel) : {out.size()}')
        out = self.patch_to_embedding(out)
        # print(f'patch_to_embedding to each token : {out.size()}')
        return out

if __name__ == "__main__":

    img_path = glob("/home/ubuntu/image_preprocessing/re_512_3ch/Train/*.jpg")

    img = Image.open(img_path[1])
    img = ToTensor()(img).unsqueeze(0)
    print(img.size())


    enc = ImageEncoder()
    out = enc(img)
    print('shape of out :', out.shape)

    # enc = Img_patch_embedding(
    #     image_size = 512,
    #     patch_size = 32,
    #     dim = 1024,
    # )
    # out = enc(img)
    # print('shape of out :', out.shape)