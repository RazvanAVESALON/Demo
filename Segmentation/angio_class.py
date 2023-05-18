
import numpy as np
import yaml
import cv2
import torch

config = None
with open('config.yaml') as f:  # reads .yml/.yaml files
    config = yaml.safe_load(f)

class AngioClass(torch.utils.data.Dataset):
    def __init__(self, angio, clipping_point ,geometrics_transforms=None,pixel_transforms=None):
        self.angio = angio
        self.clipping_point = clipping_point
        
    def __len__(self):

        return 1
      
    def __getitem__(self, idx):
        img =self.angio
        clipping_points= self.clipping_point

        target = np.zeros(img.shape, dtype=np.uint8)
        target = cv2.circle(target, [clipping_points[1], clipping_points[0]], 8, [255, 255, 255], -1)

        target = target/255
        new_img = img/255
         
        x = np.expand_dims(new_img, axis=0)
        y = np.expand_dims(target, axis=0)
        
        tensor_y = torch.from_numpy(y)
        tensor_x = torch.from_numpy(x)

        return tensor_x.float(), tensor_y.int()
    
    
    

