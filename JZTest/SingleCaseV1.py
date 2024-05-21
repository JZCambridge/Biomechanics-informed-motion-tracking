# !/usr/bin/env python
# -- coding: utf-8 --
# @Author : JZCambridge
# Version: 1.0
# @Project : $Biomechanics-informed-motion-tracking
# License: MIT

import os
import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import ndimage

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import necessary modules from parent directory
from network import Registration_Net, MotionVAE2D
from util import *

# Dataset class to load and process data
class ExampleDataset(Dataset):
    def __init__(self, data_path, filename, z):
        super(ExampleDataset, self).__init__()
        self.data_path = data_path
        self.filename = filename
        self.z = z
        
        # Load 4D image
        nim = nib.load(os.path.join(data_path, filename, filename + '_4d.nii.gz'))
        image = nim.get_fdata()
        self.img_4D = np.array(image, dtype='float32')

    def __getitem__(self, index):
        input, target, mask = load_data(self.data_path, 
                                        self.filename, 
                                        size=96, 
                                        rand_t=index,
                                        rand_z=self.z,
                                        img_4d=self.filename+'_4d.nii.gz', 
                                        img_ed=self.filename+'_frame01.nii.gz', 
                                        seg_ed=self.filename+'_frame01_gt.nii.gz')

        image = input[:1]
        image_pred = input[1:]

        return image, image_pred, target, mask

    def __len__(self):
        return self.img_4D.shape[3]

# Function to load data and preprocess it
def load_data(data_path, filename, size, rand_t=None, rand_z=None, img_4d='sa.nii.gz', img_ed='sa_ED.nii.gz', seg_ed='label_sa_ED.nii.gz', debug=False):
    if debug: print(f'rand_t: {rand_t}, rand_z: {rand_z}')
    
    # Load 4D image
    if debug: print(f'Load 4D: {os.path.join(data_path, filename, img_4d)}')
    nim = nib.load(os.path.join(data_path, filename, img_4d))
    image = nim.get_fdata()
    image = np.array(image, dtype='float32')
    
    image_max = np.max(np.abs(image))
    image /= image_max
    image_sa = image[..., rand_z, rand_t]
    image_sa = image_sa[np.newaxis]

    # Load ED image
    if debug: print(f'Load ED: {os.path.join(data_path, filename, img_ed)}')
    nim = nib.load(os.path.join(data_path, filename, img_ed))
    image = nim.get_fdata()
    image = np.array(image, dtype='float32')

    # Load ED segmentation
    if debug: print(f'Load ED Segmentation: {os.path.join(data_path, filename, seg_ed)}')
    nim_seg = nib.load(os.path.join(data_path, filename, seg_ed))
    seg = nim_seg.get_fdata()

    image_ED = image[..., rand_z]
    image_ED /= image_max
    seg_ED = seg[..., rand_z]
    
    if debug: visualize_image(seg_ED, title='seg_ED')

    slice = (seg[..., image.shape[2]//2] == 2).astype(np.uint8)
    centre = ndimage.center_of_mass(slice)
    centre = np.round(centre).astype(np.uint8)
    
    if debug: visualize_image(slice, title='slice')

    image_ED = image_ED[np.newaxis]
    seg_ED = seg_ED[np.newaxis]

    image_bank = np.concatenate((image_sa, image_ED), axis=0)
    image_bank = centre_crop(image_bank, size, centre)
    seg_ED = centre_crop(seg_ED, size, centre)
    image_bank = np.transpose(image_bank, (0, 2, 1))
    seg_ED = np.transpose(seg_ED, (0, 2, 1))
    image_bank = np.array(image_bank, dtype='float32')
    seg_ED = np.array(seg_ED, dtype='int16')
    
    if debug: visualize_image(seg_ED[0, :, :], title='seg_ED')

    mask_deform = (seg_ED == 2).astype(np.uint8)
    if debug: visualize_image(mask_deform[0, :, :], title='mask_deform')
    kernel = np.ones((3, 3), np.uint8)
    mask = ndimage.binary_dilation(mask_deform[0], iterations=3)
    mask = np.array(mask[np.newaxis], dtype='int16')
    
    if debug: visualize_image(mask[0, :, :], title='maskD3')

    return image_bank, mask_deform, mask

# Function to visualize an image
def visualize_image(image, title=''):
    plt.imshow(np.array(image), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to plot a 2x2 grid of images with optional masks
def plot_four_grids(arrays, titles, masks=None, filename=''):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(arrays[i], cmap='gray')
        if i >= 2 and masks is not None and len(masks) >= i - 1:
            mask = masks[i - 2]
            rgba_mask = np.zeros((*mask.shape, 4), dtype=np.float32)
            rgba_mask[..., 0] = 1.0  # Red channel
            rgba_mask[..., 3] = 0.5 * (mask != 0)  # Alpha channel: 50% transparency where mask is 1
            ax.imshow(rgba_mask, cmap='Reds')
        ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()

# Function to plot a histogram of a 2D matrix
def plot_histogram(matrix, bins=30, title='Histogram', xlabel='Value', ylabel='Frequency'):
    values = matrix.flatten()
    plt.hist(values, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

# Configuration for GPU
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Load pre-trained models
model_load_path = './models/registration_model_pretrained_0.001_32.pth'
VAE_model_load_path = './models/VAE_recon_model_pretrained.pth'

model = Registration_Net().to(device)
model.load_state_dict(torch.load(model_load_path))
model.eval()  # Set the model to evaluation mode

VAE_model = MotionVAE2D(img_size=96, z_dim=32).to(device)
VAE_model.load_state_dict(torch.load(VAE_model_load_path))
VAE_model.eval()  # Set the VAE model to evaluation mode

# Load the dataset
data_path = '/media/yingmohuanzhou/Data/ACDC/database/testing'
filename = 'patient126'
z = 4
example_set = ExampleDataset(data_path, filename, z)
example_loader = DataLoader(dataset=example_set, batch_size=1, shuffle=False)

# Output folder for saved images
out_folder = '/media/yingmohuanzhou/Work/OneDrive - University of Cambridge/Uni/PhD4/CMR_Motion/Biomechanics/Original'

# Perform inference on the dataset
for i, (source_image, target_image, mask_deform, mask) in enumerate(example_loader):
    source_image = source_image.to(device)
    target_image = target_image.to(device)
    mask = mask.to(device)
    mask_deform = mask_deform.to(device)

    with torch.no_grad():
        net = model(source_image, target_image, source_image)
        
        max_norm = 0.1
        df_gradient = compute_gradient(net['out']).to(device)
        recon, mu, logvar = VAE_model(df_gradient, mask, max_norm)
    
    # Extract the predicted deformed image
    recon_mask = recon.cpu().numpy()[0, 0, :, :]
    predicted_deformed_image = net['fr_st'].cpu().numpy()[0, 0, :, :]
    predicted_deformed_mask_2d = net['reverse_mask'][0, 0, :, :].cpu().numpy()
    mask_ED_2d = mask_deform[0, 0, :, :].cpu().numpy()
    img_ED_2d = target_image[0, 0, :, :].cpu().numpy()
    origninal_image_2d = source_image[0, 0, :, :].cpu().numpy()

    plot_four_grids(arrays=[img_ED_2d, predicted_deformed_image, img_ED_2d, origninal_image_2d], 
                    titles=[f'Image ED Time={i}', f'Registered Image Time={i}', 'ED Image & Mask', 'Image & Deformed Mask'],
                    masks=[mask_ED_2d, predicted_deformed_mask_2d],
                    filename=os.path.join(out_folder, f'ACDC_{filename}_Z{z}_T{i}.png'))
