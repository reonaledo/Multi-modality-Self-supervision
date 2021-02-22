"""
dataset pre-processing

"""
import os
import sys
import torchvision.transforms as transforms

def get_transforms(args):

    if args.img_size == 224:
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # referred from ChexNet
            ]
        )

    elif args.img_size == 512:
        return transforms.Compose(
            [
                # transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # referred from ChexNet
            ]
        )
