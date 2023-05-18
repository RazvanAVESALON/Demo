import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib as pt
import yaml
import cv2
import torch
from tqdm import tqdm
from datetime import datetime
import os
import cv2
import glob
from torchmetrics import MeanSquaredError
import json
from regresie import RegersionClass
from distances_regression import calculate_distance, mm2pixels, pixels2mm
import albumentations as A

def create_dataset_csv(path_construct):
    path_list = {"images_path": [], "annotations_path": [], "frames": [
    ], "patient": [], "acquisition": [], "angio_loader_header": []}
    # frame_list={"frames"}
    head, tail = os.path.split(path_construct)
    
    for acquisiton in path_construct:
        img = os.path.join(acquisiton, "frame_extractor_frames.npz")
        annotations = os.path.join(acquisiton, "clipping_points.json")
        angio_leader = os.path.join(acquisiton, "angio_loader_header.json")
        with open(annotations) as f:
            clipping_points = json.load(f)

        for frame in clipping_points:
            frame_int = int(frame)

            path_list['images_path'].append(img)
            path_list['annotations_path'].append(annotations)
            path_list['frames'].append(frame_int)
            path_list['patient'].append(os.path.basename(head))
            path_list['acquisition'].append(os.path.basename(acquisiton))
            path_list['angio_loader_header'].append(angio_leader)

    return path_list



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


def main(path):

    config = None
    with open('config.yaml') as f:  # reads .yml/.yaml files
        config = yaml.safe_load(f)

    yml_data = yaml.dump(config)
    directory = f"Test{datetime.now().strftime('%m%d%Y_%H%M')}"
    

    resize = A.Compose([
        A.Resize(height=config['data']['img_size'][0],
                 width=config['data']['img_size'][1])
    ], keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))

    
    
    path_construct = glob.glob((path))
    path_list = create_dataset_csv(path_construct)
    test_df = pd.DataFrame(path_list)
    test_ds = RegersionClass(test_df, img_size=config["data"]["img_size"], geometrics_transforms=resize)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config["train"]["bs"], shuffle=False)

    network = torch.load(
        r"/media/cuda/HDD 1TB  - DATE/AvesalonRazvanDate , Experimente/Experimente/Experiment_MSE04012023_1445/Weights/my_model04022023_0726_e250.pt")

    test_set_CSV = test(network, test_loader, test_df,thresh=config['test']['threshold'])
    dataf = pd.DataFrame(test_set_CSV)
    clipping_points_prediction={}
    for batch_index, batch in enumerate(test_loader):
        x, y, index = iter(batch)

        index = index.numpy()

        network.eval()
        x = x.type(torch.cuda.FloatTensor)

        y_pred = network(x.to(device='cuda:0'))

        for step, (input, gt, pred) in enumerate(zip(x, y, y_pred)):

            np_input = input.cpu().detach().numpy()*255
            gt = gt.cpu().detach().numpy()*config['data']['img_size'][0]

            pred = pred.cpu().detach().numpy()*config['data']['img_size'][1]
            clipping_points_prediction.update({str(dataf['Frame']):pred})      

    return clipping_points_prediction
            





if __name__ == "__main__":
    main()
