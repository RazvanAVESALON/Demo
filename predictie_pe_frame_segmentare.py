
import numpy as np
import json
import torch
from Segmentation.angio_class import AngioClass
from Segmentation.blob_detector import blob_detector
import torch.nn.functional as F
from Segmentation.distances import calcuate_distance, pixels2mm
class DEMO_PredicitionClass_seg:
    
    def __init__(self, angio  , clipping_point , metadata ):
        self.angio = angio
        self.metadata =metadata
        self.clipping_point=clipping_point
        self.network = torch.load(r"E:\__RCA_bif_detection\Arhive xeperimete vechi\Expetimente_Segmentare\Experiment_Dice_index12212022_2049\Weights\my_model12222022_1931_e150.pt")
 
    def __predict__(self):

        
        test_ds=AngioClass(self.angio,self.clipping_point) 

        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)
  
        for batch_index, batch in enumerate(test_loader):
            x, y = iter(batch)

            self.network.eval()
            x = x.type(torch.cuda.FloatTensor)

            y_pred = self.network(x.to(device='cuda:0'))


            for step, (input, gt, pred) in enumerate(zip(x, y, y_pred)):
                pred = F.softmax(pred, dim=0)[1].detach().cpu().numpy()

                pred[pred > 0.5] = 1
                pred[pred <= 0.5] = 0
                np_gt = gt.cpu().detach().numpy()
                
                 
                clipping_points_list=blob_detector(pred)
                copie_pred=clipping_points_list
  
                gt=blob_detector(np_gt[0])
                pred_mm=pixels2mm(copie_pred,self.metadata['MagnificationFactor'], self.metadata['ImageSpacing'] )
                gt_mm=pixels2mm(gt,self.metadata['MagnificationFactor'], self.metadata['ImageSpacing'] )
                distance=calcuate_distance(gt_mm, pred_mm)
                print("LIST",clipping_points_list) 

               

        return clipping_points_list , distance
        




def main():

    img = np.load(r"E:\__RCA_bif_detection\data\00cca518a10d41adb9476aefc38a0b69\40118090\frame_extractor_frames.npz")['arr_0']
    with open(r"E:\__RCA_bif_detection\data\00cca518a10d41adb9476aefc38a0b69\40118090\clipping_points.json") as f:
        clipping_points = json.load(f)

    with open(r"E:\__RCA_bif_detection\data\00cca518a10d41adb9476aefc38a0b69\40118090\angio_loader_header.json") as f:
        angio_leader= json.load(f)
    
    print (img[3].shape)
    print (clipping_points[str(3)])
    predictie= DEMO_PredicitionClass_seg(angio=img[3],clipping_point=clipping_points[str(3)],metadata=angio_leader).__predict__() 
    print("Predictie:",predictie)
 

if __name__ == "__main__":
    main()

