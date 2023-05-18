
import numpy as np
import json
import torch
from regresie import RegersionClass
import albumentations as A
import tqdm
from torchmetrics import MeanSquaredError
import pandas as pd 
from distances_regression import calculate_distance , pixels2mm

class DEMO_PredicitionClass:
    
    def __init__(self, angio  , clipping_point , metadata ):
        self.angio = angio
        self.metadata =metadata
        self.clipping_point=clipping_point
        self.network = torch.load(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\experiments\exp 21.03\Experiment_MSE03232023_0048\Weights\my_model03232023_1823_e350.pt")

    def test(network, test_loader, dataframe, thresh=0.5):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Starting testing on device {device} ...")

        metric = MeanSquaredError()

        
        logs_dict = {'MSE': [], 'Patient': [],
                    'Acquistion': [], 'Distance': []}
        
        frame_dict={}
        network.eval()
        with tqdm(desc='test', unit=' batch', total=len(test_loader.dataset)) as pbar:

            #print (test_loader.dataset[100])
            for data in test_loader:

                ins, tgs, index = data
                network.to(device)
                ins = ins.to(device)
                metric = metric.to(device)
                tgs = tgs.to(device)
                output = network(ins)

                for batch_idx, (frame_pred, target_pred) in enumerate(zip(output, tgs)):

                    MSE_score = metric(frame_pred, target_pred)

                    patient, acquisition, frame, header, annotations = test_loader.dataset.csvdata(
                        (index[batch_idx].numpy()))
                    # print(header)

                    with open(annotations) as g:
                        ann = json.load(g)

                    with open(header) as f:
                        angio_loader = json.load(f)

                    frame_pred = frame_pred.cpu().detach().numpy()
                    target_pred = target_pred.cpu().detach().numpy()

                    gt_coords_mm = pixels2mm(
                        target_pred, angio_loader['MagnificationFactor'], angio_loader['ImageSpacing'])
                    pred_cord_mm = pixels2mm(
                        frame_pred, angio_loader['MagnificationFactor'], angio_loader['ImageSpacing'])

                    if pred_cord_mm == [] and pred_cord_mm == []:
                        logs_dict['Distance'].append(
                            str("Can't calculate Distance For this frame ( No prediction )"))
                    else:
                        distance = calculate_distance(gt_coords_mm, pred_cord_mm)

                    dict 
                    frame_dict[f'{frame}'].append(frame_pred)
                    logs_dict['Distance'].append(distance)
                    logs_dict['MSE'].append(MSE_score.cpu().detach().numpy())
                    logs_dict['Patient'].append(patient)
                    logs_dict['Acquistion'].append(acquisition)
                    logs_dict['Frame'].append(frame)

                pbar.update(ins.shape[0])

            MSE_score = metric.compute()
            print(MSE_score)

            print(f'[INFO] MSE score is {MSE_score:.2f} %')
            return logs_dict
        
        
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

                np_input = input.cpu().detach().numpy()*255
                gt = gt.cpu().detach().numpy()*255

                pred = pred.cpu().detach().numpy()*255

                clipping_points_prediction=pred     

        return clipping_points_prediction
        




def main():

    img = np.load(r"E:\__RCA_bif_detection\data\00cca518a10d41adb9476aefc38a0b69\40118090\frame_extractor_frames.npz")['arr_0']
    with open(r"E:\__RCA_bif_detection\data\00cca518a10d41adb9476aefc38a0b69\40118090\clipping_points.json") as f:
        clipping_points = json.load(f)

    with open(r"E:\__RCA_bif_detection\data\00cca518a10d41adb9476aefc38a0b69\40118090\angio_loader_header.json") as f:
        angio_leader= json.load(f)
    
    print (img[3].shape)
    print (clipping_points[str(3)])
    predictie= DEMO_PredicitionClass(angio=img[3],clipping_point=clipping_points[str(3)],metadata=angio_leader).__predict__() 
    print("Predictie:",predictie)
 

if __name__ == "__main__":
    main()

