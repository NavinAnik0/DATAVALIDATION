import torchvision
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch.cuda.amp as amp
import torch.nn as nn
from torchvision.transforms.transforms import Compose, Normalize, RandomResizedCrop, RandomAffine, RandomPerspective, \
    RandomApply, ToTensor, RandomHorizontalFlip, Resize
from torch.utils.data import DataLoader
import torch
from nfnets import NFNet, SGD_AGC, pretrained_nfnet, MulticlassClassifier, CustomNfNet
import time
from torch.utils.tensorboard import SummaryWriter
import boto3
from torch.profiler import profile, ProfilerActivity


def main():
    s3 = boto3.client('s3')
    pretrained_model_object = s3.get_object(Bucket='dev-model-training', Key='pretrained_model/F0_haiku.npz')[
        'Body'].read()
    pretrained = pretrained_nfnet(
        model_object=pretrained_model_object,
        stochdepth_rate=0.25,
        alpha=0.2,
        activation='gelu'
    )

    # for name, param in pretrained.named_parameters():
    #     param.requires_grad = False
    # pretrained = torchvision.models.efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    # pretrained = torchvision.models.resnet34(weights='IMAGENET1K_V1')
    stem = pretrained.stem
    for name, param in stem.named_parameters():
        param.requires_grad = False
    body = pretrained.body[0:6]
    for name, param in body.named_parameters():
        param.requires_grad = False
    model = CustomNfNet(stem, body, stochdepth_rate=0.25, alpha=0.2, activation='gelu')
    # model.to('cuda:0')
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total paramerter  Ram Requirements {pytorch_total_params / 1000000} MB")
    torch.save(model, 'test.pth')
    input = torch.rand(4, 3, 1080, 1920)
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        output = model(input)
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))

main()