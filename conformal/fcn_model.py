from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.segmentation import model_urls, load_state_dict_from_url
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
import random
import os


class FCNHead(nn.Sequential):
    def __init__(self, img_shape, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]
        super(FCNHead, self).__init__(*layers)
        self.img_shape = img_shape
        self.in_channels, self.channels = in_channels, channels
        self.num_classes = channels
        self.feature_shape = (2048, self.img_shape[0] // 8, self.img_shape[1] // 8)

    def forward(self, inputs):
        inputs = inputs.view(inputs.shape[0], *self.feature_shape)
        output = super(FCNHead, self).forward(inputs)
        return output


class FCN(nn.Module):
    def __init__(self, img_shape, backbone, classifier):
        super(FCN, self).__init__()
        self.img_shape = img_shape
        # remove the first dimension "unlabeled"
        self.out_shape = (classifier.num_classes - 1) * img_shape[0] * img_shape[1]
        self.encoder = backbone
        self.g = classifier

    @staticmethod
    def output_post_process(output, img_shape):
        output = F.interpolate(output, size=img_shape, mode='bilinear', align_corners=False)
        # remove the first dimension "unlabeled"
        output = output[:, 1:, ...]
        output = output.view(output.shape[0], -1)
        return output

    def forward(self, x):
        features = self.encoder(x)
        output = self.g(features)
        output = self.output_post_process(output, img_shape=self.img_shape)
        return output


class MyIntermediateLayerGetter(IntermediateLayerGetter):
    def __init__(self, model, return_layers: str):
        super(MyIntermediateLayerGetter, self).__init__(model, {return_layers: 'out'})
        self.return_layers = return_layers

    def forward(self, x):
        for name, module in self.items():
            x = module(x)
            if name == self.return_layers:
                out = x
        out = out.view(out.shape[0], -1)
        return out


def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:

        w_class = 1 / (ln(c + p_class)),

    where c is usually 1.02 and p_class is the propensity score of that
    class:

        propensity_score = freq_class / total_pixels.

    References: https://arxiv.org/abs/1606.02147

    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.

    """
    class_count = 0
    total = 0
    for _, label, _ in dataloader:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


def _segm_resnet(backbone_name, img_shape, num_classes, pretrained_backbone=True):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    return_layers = 'layer4'
    backbone = MyIntermediateLayerGetter(backbone, return_layers=return_layers)

    inplanes = 2048
    classifier = FCNHead(img_shape, inplanes, num_classes)

    model = FCN(img_shape, backbone, classifier)
    return model


def make_fcn(pretrained=False, progress=True, img_shape=(512, 1024), num_classes=21, **kwargs):
    model = _segm_resnet('resnet50', img_shape, num_classes, **kwargs)
    if pretrained:
        arch = 'fcn_resnet50_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            del state_dict['classifier.4.weight']
            del state_dict['classifier.4.bias']
            model.load_state_dict(state_dict, strict=False)
    return model
