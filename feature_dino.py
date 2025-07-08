import torch
from torch import nn
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import math
import time
from utils import load_image,VitExtractor
import torchvision.transforms as T
from PIL import Image

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image_path = "./data"   

    config={
        "num":888,
        "h":196,
        "w":196,
        
        "facet":"tokens",
        "stride":14,
        "layer":None,
        "model_name":"dinov2_vitl14"
        }
    
    sc=5
    
    dino_extractor = VitExtractor(model_name=config['model_name'], device=device, stride=config['stride'])
    dino_extractor = dino_extractor.eval().to(device)

    @torch.no_grad()
    def get_ebd_image(image, model_name=config['model_name'], facet='tokens', layer=None, device: str = 'cuda:0'):
        
        imagenet_norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ph = dino_extractor.get_height_patch_num(image[[0]].shape)
        pw = dino_extractor.get_width_patch_num(image[[0]].shape)
        dino_embedding_dim = dino_extractor.get_embedding_dim(model_name)
        n_layers = dino_extractor.get_n_layers()
        layers = [n_layers - 9] if layer is None else [layer]

        dino_features_video = torch.zeros(size=(image.shape[0], dino_embedding_dim, ph, pw), device='cpu')
        for i in range(image.shape[0]):
            dino_input = imagenet_norm(image[[i]]).to(device)
            if facet == "keys":
                features = dino_extractor.get_keys_from_input(dino_input, layers=layers)
            elif facet == "queries":
                features = dino_extractor.get_queries_from_input(dino_input, layers=layers)
            elif facet == "values":
                features = dino_extractor.get_values_from_input(dino_input, layers=layers)
            elif facet == "tokens":
                features = dino_extractor.get_feature_from_input(dino_input, layers=layers) # T (HxW + 1) x C
            else:
                raise ValueError(f"facet {facet} not supported")
            features = rearrange(features[:, 1:, :], "heads (ph pw) ch -> (ch heads) ph pw", ph=ph, pw=pw)
            dino_features_video[i] = features.cpu()
        
        return dino_features_video.to(device)
                    
    with tqdm(total=config['num'],position=0) as pbar:
            pbar.set_description('Processing:')
            for index in range(config['num']):
                
                pth=os.path.join(image_path,f"{index}")
                moving=load_image(f"{pth}/moving.jpg",device,config['h'],config['w'])
                fixed=load_image(f"{pth}/fixed.jpg",device,config['h'],config['w'])
                temp=load_image(f"{pth}/temp.jpg",device,config['h'],config['w'])
            
                _moving=load_image(f"{pth}/moving.jpg",device,config['h']*sc,config['w']*sc)
                _fixed=load_image(f"{pth}/fixed.jpg",device,config['h']*sc,config['w']*sc)
            
                dino_ebd_f=get_ebd_image(_fixed)
                dino_ebd_m=get_ebd_image(_moving)
                
                dino_ebd_pth=os.path.join(pth,"dino_ebd")
                os.makedirs(dino_ebd_pth, exist_ok=True)
                torch.save(dino_ebd_f.cpu(),dino_ebd_pth+"/fixed_ebd.pt")
                print(f"Saved {dino_ebd_pth}, shape: {dino_ebd_f.shape}")
                torch.save(dino_ebd_m.cpu(),dino_ebd_pth+"/moving_ebd.pt")
                print(f"Saved {dino_ebd_pth}, shape: {dino_ebd_m.shape}")
                
                pbar.update(1)
