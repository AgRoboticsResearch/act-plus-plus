# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae
from .detr_vae import build_cnnmlp as build_cnnmlp
from .detr_vae import build_epact as build_epact
from .detr_vae import build_dinoact as build_dinoact

def build_ACT_model(args):
    return build_vae(args)

def build_CNNMLP_model(args):
    return build_cnnmlp(args)

def build_EPACT_model(args):
    return build_epact(args)

def build_DINOACT_model(args):
    return build_dinoact(args)