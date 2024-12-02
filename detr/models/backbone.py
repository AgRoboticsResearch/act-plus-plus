# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import math
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from models.position_encoding import build_position_encoding
from models.dinov2.dinov2 import DINOv2

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        # print("xs: ", xs)
        # print("xs[0]: ", xs['0'].shape)

        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # print("Joiner x: ", x.shape)

            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos

class DinoJoiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        
        out: List[NestedTensor] = []
        pos = []
        x = xs
        out.append(x)
        # position encoding
        pos.append(self[1](x).to(x.dtype))
        # print("DinoJoiner x: ", x.shape)

        return out, pos

class DinoBackbone(nn.Module):
    def __init__(self, model_name='vits'):
        super().__init__()

        self.intermediate_layer_idx = {
        'vits': [2, 5, 8, 11],
        'vitb': [2, 5, 8, 11], 
        'vitl': [4, 11, 17, 23], 
        'vitg': [9, 19, 29, 39]
        }
        self.model_name = model_name
        self.model_backbone = DINOv2(model_name=model_name)
        self.num_channels = 384

        if model_name == 'vits':
            self.num_channels = 384
            self.num_features = 1369
        train_backbone = False
        if not train_backbone:
            print("[DinoBackbone]: Freezing the backbone")
            for name, parameter in self.model_backbone.named_parameters():
                # the dino v2 backbone is not going to be trained
                parameter.requires_grad_(False)
        else:
            print("[DinoBackbone]: Training the backbone")
        self.linear = nn.Linear(self.num_features, 15*25)

    def forward(self, x):
        xs = self.model_backbone(x)
        x = xs['x_norm_patchtokens']
        x = x.permute(0, 2, 1)
        # print("DinoBackbone x: ", x.shape)
        x = self.linear(x)
        # print("DinoBackbone x: ", x.shape)
        x = x.reshape(-1, self.num_channels, 15, 25)
        # print("DinoBackbone x: ", x.shape)

        return x

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def build_dino_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    args.backbone = 'resnet18'
    encoder='vits'


    dino_backbone = DinoBackbone(model_name=encoder)


    # backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    # model = Joiner(backbone, position_embedding)
    # model.num_channels = backbone.num_channels

    model = DinoJoiner(dino_backbone, position_embedding)
    model.num_channels = dino_backbone.num_channels

    return model
