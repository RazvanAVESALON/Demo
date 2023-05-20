
import numpy as np
import json
import torch
from Regression.regresie import RegersionClass
import albumentations as A
from Regression.distances_regression import calculate_distance , pixels2mm


class DEMO_PredicitionClass:
    
    def __init__(self, angio  , clipping_point , metadata ):
        self.angio = angio
        self.metadata =metadata
        self.clipping_point=clipping_point
        self.network = torch.load(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\experiments\exp 21.03\Experiment_MSE03232023_0048\Weights\my_model03232023_1823_e350.pt")
        
        
    def __predict__(self):
        
        
        print ("tema ") 
        resize = A.Compose([
            A.Resize(height=512,
                    width=512)
        ], keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))

        
        test_ds=RegersionClass(self.angio,self.clipping_point) 

        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)
        
        print ("LEN",len(test_loader))
        for  batch_index,batch in enumerate(test_loader):
            x, y= iter(batch)

            print(x.shape,y.shape) 
            self.network.eval()
            x = x.type(torch.cuda.FloatTensor)

            y_pred = self.network(x.to(device='cuda:0'))

            for step, (input, gt, pred) in enumerate(zip(x, y, y_pred)):

                gt = gt.cpu().detach().numpy()*255

                pred = pred.cpu().detach().numpy()*255
                
                
                print (pred, gt ) 
                pred_mm=pixels2mm(pred,self.metadata['MagnificationFactor'], self.metadata['ImageSpacing'] )
                gt_mm=pixels2mm(gt,self.metadata['MagnificationFactor'], self.metadata['ImageSpacing'] )
                distance=calculate_distance(gt_mm, pred_mm)
                print (gt_mm,pred_mm)
                print (distance)
                clipping_points_prediction=pred 
                    
                


        return clipping_points_prediction, distance
        




def main():

    img = np.load(r"E:\__RCA_bif_detection\data\00cca518a10d41adb9476aefc38a0b69\40118090\frame_extractor_frames.npz")['arr_0']
    with open(r"E:\__RCA_bif_detection\data\00cca518a10d41adb9476aefc38a0b69\40118090\clipping_points.json") as f:
        clipping_points = json.load(f)

    with open(r"E:\__RCA_bif_detection\data\00cca518a10d41adb9476aefc38a0b69\40118090\angio_loader_header.json") as f:
        angio_leader= json.load(f)
    
    print (img[3].shape)
    print (clipping_points[str(3)])
    predictie , distance = DEMO_PredicitionClass(angio=img[3],clipping_point=clipping_points[str(3)],metadata=angio_leader).__predict__() 
    print("Predictie:",predictie,"Distance",distance)
 

if __name__ == "__main__":
    main()

