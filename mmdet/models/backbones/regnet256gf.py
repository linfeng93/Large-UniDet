# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from classy_vision.models import RegNet as ClassyRegNet, build_model

from ..builder import BACKBONES
from mmcv.runner import BaseModule, load_checkpoint


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


@BACKBONES.register_module()
class RegNet256gf(BaseModule):

    _model_config = {
        "model_scale": "256gf",
        "depth": 27,  
        "w_0": 640,
        "w_a": 230.83,
        "w_m": 2.53,
        "group_width": 373,
    }

    def __init__(self,
                 pretrain,
                 freeze_at=5,
                 norm='SyncBN'):
        super(RegNet256gf, self).__init__()
        model_config = self._model_config
        self.ckpt_path = pretrain
        self.frozen_stages = freeze_at
        self._norm = norm

        if "name" in model_config:
            name = model_config["name"]
            model = build_model({"name": name})
        else:
            model = ClassyRegNet.from_config(model_config)
        
        if self._norm == "SyncBN":
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        feature_blocks: List[Tuple[str, nn.Module]] = []

        # - get the stem
        feature_blocks.append(("conv1", model.stem))

        # - get all the feature blocks
        for k, v in model.trunk_output.named_children():
            assert k.startswith("block"), f"Unexpected layer name {k}"
            block_index = len(feature_blocks) + 1
            feature_blocks.append((f"res{block_index}", v))

        self._feature_blocks = nn.ModuleDict(feature_blocks)

        if self.ckpt_path is not None and os.path.exists(self.ckpt_path):
            load_checkpoint(self, self.ckpt_path, map_location='cpu')

        self._freeze_stages()

    def _freeze_stages(self):
        assert len(self._feature_blocks) == 5
        if self.frozen_stages > 0:
            for idx, block in enumerate(self._feature_blocks):
                if self.frozen_stages > idx:
                    if idx == 0:
                        self._feature_blocks[block].eval()
                        for param in self._feature_blocks[block].parameters():
                            param.requires_grad = False
                            self._feature_blocks[block] = FrozenBatchNorm2d.convert_frozen_batchnorm(self._feature_blocks[block])
                    else:
                        self._freeze_layers(self._feature_blocks[block])

    def _freeze_layers(self, layer):
        for _, module in layer.named_modules():
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(module)

    def forward(self, x, out_features=[1, 2, 3, 4]):
        outs = []
        for i, block in enumerate(self._feature_blocks):  # i = 0 ~ 4
            x = self._feature_blocks[block](x)
            if i in out_features:
                outs.append(x)
        return tuple(outs)