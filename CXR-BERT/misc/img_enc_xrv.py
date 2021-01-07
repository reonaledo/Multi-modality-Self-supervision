import torchxrayvision as xrv
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor
from PIL import Image


class ImageEncoder_pool(nn.Module):
    def __init__(self, args):
        super(ImageEncoder_pool, self).__init__()
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
        print('random_sample_size:', random_sample.size())

        return random_sample


mimic_dataset = xrv.datasets.MIMIC_Dataset
# print(mimic_dataset)

xrv_model = xrv.models.DenseNet(weights="mimic_ch")
weights = xrv_model.state_dict()
print(weights)

tv_resnet = torchvision.models.resnet50(pretrained=True)
modules = list(xrv_model.children())[:-2]
model = nn.Sequential(*modules)
print(model)

img = Image.open('C:\\Users\\subae\\Desktop\\CXR-BERT\\misc\\s50010747.jpg')
img = ToTensor()(img).unsqueeze(0)
print('model_input_size:', img.size())  # torch.Size([1, 1, 256, 256])
out = model(img)
print('model_output_size:', out.size())  # torch.Size([1, 1, 256, 256])

prob = xrv_model(img)
label = xrv.datasets.default_pathologies
map_dict = dict(zip(label, prob[0]))


"""
Atelectasis  0
Cardiomegaly
Consolidation 0
Edema 0 
Enlarged Cardiomediastinum 0 
Fracture 0
Lung Lesion 0 
Lung Opacity 0
Pneumonia 0 
Pneumothorax 0 

Pleural Effusion  -> Effusion
Pleural Other
Support Devices
No Finding

-----NOT EXIST---------
Infiltration
Emphysema
Fibrosis
Pleural_Thickening
Nodule
Mass
Hernia
"""