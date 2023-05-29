from __future__ import annotations
import numpy as np
import json
import torch


class RegersionClass(torch.utils.data.Dataset):
    def __init__(self, angio, clip , geometrics_transforms=None, pixel_transforms=None):
        self.angio = angio
        self.bf = clip
        self.geometrics_transforms = geometrics_transforms

    def __len__(self):

        return 1

    def __getitem__(self,idx):
        img = self.angio/255
        bf = self.bf
        bf [0] =bf [0]/255
        bf [1] =bf [1]/255

        tensor_y = torch.from_numpy(np.array(bf))
        tensor_x = torch.from_numpy(img).unsqueeze(dim=0)
        tensor_x = torch.concat([tensor_x, tensor_x, tensor_x], dim=0)
        return tensor_x.float(), tensor_y.float() 
