import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import math
import time
from PIL import Image
from base import SpatialTransformer,ResizeTransform,pca_lowrank_transform
from utils import load_image,VitExtractor
import torchvision.transforms as T
from einops import rearrange
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms.functional import gaussian_blur
        
def get_transform(img_size=(224, 224)):

    return T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class ImageRegistrationDataset(Dataset):
    def __init__(self, data_root, transform=None, flag=1, limit=888):

        self.data_root = data_root
        self.transform = transform if transform is not None else T.ToTensor()
        
        self.limit=limit
        
        if flag!=0:
            self.sample_folders = sorted(
                [d for d in os.listdir(data_root) if d.startswith('train_')],
                key=lambda x: int(x.split('_')[1])
            )        
        else:
            self.sample_folders = sorted(
                [d for d in os.listdir(data_root) if d.startswith('val_')],
                key=lambda x: int(x.split('_')[1])
            )
        
    def __len__(self):
    
        return min(self.limit, len(self.sample_folders))
    
    def __getitem__(self, idx):
    
        folder_name = self.sample_folders[idx]
        folder_path = os.path.join(self.data_root, folder_name)
        
        moving_img = Image.open(os.path.join(folder_path, 'moving.jpg'))
        fixed_img = Image.open(os.path.join(folder_path, 'fixed.jpg'))
        
        temp=Image.open(os.path.join(folder_path, 'temp.jpg'))
        
        if self.transform:
            moving_img = self.transform(moving_img)
            fixed_img = self.transform(fixed_img)
            
            temp = self.transform(temp)
        
        feature_path = os.path.join(folder_path, 'dino_ebd')
        moving_feat = torch.load(os.path.join(feature_path, 'moving_ebd.pt')).squeeze()
        fixed_feat = torch.load(os.path.join(feature_path, 'fixed_ebd.pt')).squeeze()
        
        return {
            'moving_img': moving_img,
            'fixed_img': fixed_img,
            'moving_feat': moving_feat,
            'fixed_feat': fixed_feat,
            'folder_name': folder_name,
            
            'temp':temp  
        }

def create_dataloaders(val_root, batch_size=32, num_workers=4, img_size=(224, 224), limit=888):

    transform = get_transform(img_size)
    
    val_dataset = ImageRegistrationDataset(val_root, transform=transform, flag=0, limit=limit)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,    
        #prefetch_factor=20,
        #multiprocessing_context=multiprocessing.get_context('spawn')
    )
    
    return val_loader
    
def plot_deformed_grid(flow_field, grid_step=20, index=0):

    print(flow_field.shape)

    h, w, _ = flow_field.shape
    grid_x, grid_y = np.meshgrid(np.linspace(0, w, w//grid_step), 
                                np.linspace(0, h, h//grid_step))
    
    deformed_x = grid_x + flow_field[::grid_step, ::grid_step, 0]
    deformed_y = grid_y + flow_field[::grid_step, ::grid_step, 1]
    
    plt.figure(figsize=(10,10))
    plt.plot(grid_x, grid_y, 'k-', linewidth=0.5)
    plt.plot(grid_x.T, grid_y.T, 'k-', linewidth=0.5)
    plt.plot(deformed_x, deformed_y, 'r-', linewidth=1)
    plt.plot(deformed_x.T, deformed_y.T, 'r-', linewidth=1)
    plt.gca().invert_yaxis()
    plt.title("Grid Deformation Visualization")
    
    plt.savefig(f"./disp_{index}.png")
    plt.show()
    
    plt.close()
    
def create_identity_grid(height, width):
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing='ij'
    )
    grid = torch.stack((x, y), dim=2)  
    return grid.unsqueeze(0)
    
def spatial_cosine_similarity(tensor1, tensor2, eps=1e-8, eta=0):

    tensor1_norm = F.normalize(tensor1, p=2, dim=1)  
    tensor2_norm = F.normalize(tensor2, p=2, dim=1)  
    
    similarity = (tensor1_norm * tensor2_norm).sum(dim=1, keepdim=True)  
    
    similarity = torch.clamp(similarity, -1.0 + eps, 1.0 - eps)
    
    binary_mask = (similarity > eta).float()
    
    return binary_mask#similarity  
        
if __name__ == '__main__':
    image_path = "./data"
    out_path = "./output"

    configs={
        "num":888,
        "h":196,
        "w":196,

        'smooth_weight' : 35, 
        'lr' : 3,
        'num_iter' : 500
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    loss_func=torch.nn.MSELoss()
        
    loss_val=[]
    loss_test=[]
        
    val_loader=create_dataloaders(image_path, batch_size=1, num_workers=0, img_size=(configs['h'], configs['w']), limit=configs['num'])
        
    for index, batch in enumerate(tqdm(val_loader,total=len(val_loader),position=0)):
    
        moving_imgs = batch['moving_img'].to(device)
        fixed_imgs = batch['fixed_img'].to(device)
        dino_ebd_m = batch['moving_feat'].to(device)
        dino_ebd_f = batch['fixed_feat'].to(device)
        folder_name = batch['folder_name']
        
        temp = batch['temp'].to(device)
        
        me_in=rearrange(dino_ebd_m,"a b c d -> (a c d) b")
        fe_in=rearrange(dino_ebd_f,"a b c d -> (a c d) b")
        pca_in=torch.concat([fe_in,me_in],dim=0)
        
        n_components=256
        size=dino_ebd_f.shape[-1]
        sc=configs['h']/size

        reduced_patches, eigenvalues = pca_lowrank_transform(pca_in, n_components)

        _moving_ebd = reduced_patches[size*size:, :]
        _fixed_ebd = reduced_patches[:size*size, :]

        F_moving = _moving_ebd.reshape([1,size,size,n_components]).permute(0,3,1,2)
        F_fixed = _fixed_ebd.reshape([1,size,size,n_components]).permute(0,3,1,2)
        print(dino_ebd_m.shape)
        
        lm=torch.load(f"data/{index}/mask.pt")
            
        _F_fixed=lm["mask1"].to(torch.float).to(device)
        _F_moving=lm["mask2"].to(torch.float).to(device)
        
        print(_F_fixed.shape)
        
        _F_fixed=F.interpolate(_F_fixed, align_corners=True, size=(size,size), mode="bilinear")
        _F_moving=F.interpolate(_F_moving, align_corners=True, size=(size,size), mode="bilinear")
        _bbox=lm["bbox"].to(torch.float).to(device)
        
        field = torch.randn(1, 2, size, size, device=device, requires_grad=True)
        
        optimizer = torch.optim.Adam([field], lr=configs['lr'])
        
        grid_identity = create_identity_grid(size, size)

        for i in range(configs['num_iter']):
            optimizer.zero_grad()
        
            displacement_norm = torch.stack([
                field[:, 0, :, :] / (size / 2),
                field[:, 1, :, :] / (size / 2)
            ], dim=-1)  
        
            sampling_grid = grid_identity + displacement_norm  
        
            warped_moving = F.grid_sample(F_moving, sampling_grid, align_corners=True, mode='bilinear', padding_mode='border')
            
            _warped_moving = F.grid_sample(_F_moving, sampling_grid, align_corners=True, mode='bilinear', padding_mode='border')
            
            sim_map = spatial_cosine_similarity(warped_moving, F_fixed)
            
            diff=(warped_moving- F_fixed)*sim_map
            
            loss_f = F.mse_loss(diff, torch.zeros_like(diff))
            loss_m = F.mse_loss(_warped_moving*_bbox, _F_fixed)
        
            loss = 1*loss_f+Grad().loss(field)*configs['smooth_weight']+1*loss_m
        
            loss.backward()
            optimizer.step()
        
            if (i+1) % 500 == 0:
                print(f"Iter {i+1}, Loss: {loss.item():.6f}")
        
        resize = ResizeTransform(1/sc,2)
        resize = resize.to(device)
        
        _disp=resize(field)
        
        disp = torch.stack([
            gaussian_blur(_disp[:, 0], kernel_size=5, sigma=3.0),
            gaussian_blur(_disp[:, 1], kernel_size=5, sigma=3.0)
        ], dim=1)
        
        resize1 = ResizeTransform(sc,2)
        resize1 = resize1.to(device)
        
        grid_identity = create_identity_grid(configs['h'], configs['w'])
        
        displacement_norm = torch.stack([
                disp[:, 0, :, :] / (configs['h'] / 2),
                disp[:, 1, :, :] / (configs['w'] / 2)
            ], dim=-1)  
    
        sampling_grid = grid_identity + displacement_norm  
    
        image_pred = F.grid_sample(moving_imgs, sampling_grid, align_corners=True, mode='bicubic', padding_mode='border')
        
        image_np=image_pred[0].permute(1,2,0).detach().to("cpu").numpy()
        image_np*=255
        image_np=np.clip(image_np,0,255)

        _loss=loss_func(image_pred,temp)

        loss_test.append(_loss.detach().to("cpu").item())
        loss_val.append(loss.detach().to("cpu").item())

        image_final=Image.fromarray(np.uint8(image_np))
        image_final.save(f"{out_path}/{index}.jpg")