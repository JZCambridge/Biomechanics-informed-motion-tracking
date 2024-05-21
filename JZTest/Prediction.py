import os
import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch.utils.data as data
# import cv2
from scipy import ndimage

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Now you can import the necessary modules
from network import Registration_Net, MotionVAE2D
from util import *

def get_folders(path):
    # List all directories in the specified path
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    return folders

# specific dataloader
class ValDataset(data.Dataset):
    def __init__(self, data_path):
        super(ValDataset, self).__init__()
        self.data_path = data_path
        self.folders = get_folders(self.data_path)

    def __getitem__(self, index):
        input, target, mask = load_data(self.data_path, 
                                        self.folders[index], 
                                        size=96, 
                                        rand_frame=index % 30,
                                        img_4d=self.folders[index] + '_4d.nii.gz', 
                                        img_ed=self.folders[index] + '_frame01.nii.gz', 
                                        seg_ed=self.folders[index] + '_frame01_gt.nii.gz')

        image = input[:1]
        image_pred = input[1:]

        return image, image_pred, target, mask

    def __len__(self):
        return len(self.folders)


def load_data(data_path, filename, size, rand_frame=None, img_4d='sa.nii.gz', img_ed='sa_ED.nii.gz', seg_ed='label_sa_ED.nii.gz', debug=True):
    # Load images and labels
    if debug: print(f'Load 4d: {os.path.join(data_path, filename, img_4d)}')
    nim = nib.load(os.path.join(data_path, filename, img_4d))
    image = nim.get_fdata()[:, :, :, :]
    image = np.array(image, dtype='float32')

    # generate random index for t and z dimension
    if rand_frame is not None:
        rand_t = rand_frame
        rand_z = rand_frame % (image.shape[2]-1)+1
        print(f'rand_frame: {rand_frame}, rand_t:{rand_t}, rand_z:{rand_z}')
    else:
        rand_t = np.random.randint(0, image.shape[3])
        rand_z = np.random.randint(1, image.shape[2]-1)

    # preprocessing
    image_max = np.max(np.abs(image))
    image /= image_max
    image_sa = image[..., rand_z, rand_t]
    image_sa = image_sa[np.newaxis]

    if debug: print(f'Load ED: {os.path.join(data_path, filename, img_ed)}')
    nim = nib.load(os.path.join(data_path, filename, img_ed))
    image = nim.get_fdata()[:, :, :]
    image = np.array(image, dtype='float32')

    if debug: print(f'Load ED Segmentation: {os.path.join(data_path, filename, seg_ed)}')
    nim_seg = nib.load(os.path.join(data_path, filename, seg_ed))
    seg = nim_seg.get_fdata()[:, :, :]

    image_ED = image[..., rand_z]
    image_ED /= image_max
    seg_ED = seg[..., rand_z]
    
    # Visualize the 2D predicted deformed image
    if debug: visualize_image(seg_ED,title='seg_ED')

    slice = (seg[..., image.shape[2]//2] == 2).astype(np.uint8)
    centre = ndimage.center_of_mass(slice) #  Use scipy.ndimage instead of scipy.ndimage.measurements.
    centre = np.round(centre).astype(np.uint8)
    
    # Visualize the 2D predicted deformed image
    if debug: 
        print(f'image.shape[2]:{image.shape[2]}')
        visualize_image(slice,title='slice')

    image_ED = image_ED[np.newaxis]
    seg_ED = seg_ED[np.newaxis]

    image_bank = np.concatenate((image_sa, image_ED), axis=0)

    image_bank = centre_crop(image_bank, size, centre)
    seg_ED = centre_crop(seg_ED, size, centre)
    image_bank = np.transpose(image_bank, (0, 2, 1))
    seg_ED = np.transpose(seg_ED, (0, 2, 1))
    image_bank = np.array(image_bank, dtype='float32')
    seg_ED = np.array(seg_ED, dtype='int16')
    
    # Visualize the 2D predicted deformed image
    if debug: visualize_image(seg_ED[0,:,:],title='seg_ED')

    mask = (seg_ED == 2).astype(np.uint8)
    # mask = centre_crop(mask, size, centre) TODO Why centre crop again??
    # Visualize the 2D predicted deformed image
    if debug: visualize_image(mask[0,:,:],title='mask0')
    
    kernel = np.ones((3, 3), np.uint8)
    mask = ndimage.binary_dilation(mask[0], kernel, iterations=3)
    # mask = cv2.dilate(mask[0], kernel, iterations=3)
    mask = np.array(mask[np.newaxis], dtype='int16')
    
    # Visualize the 2D predicted deformed image
    if debug: visualize_image(mask[0,:,:],title='mask')

    return image_bank, seg_ED, mask

# Function to visualize the predicted deformed image
def visualize_image(image, title=''):
    plt.imshow(np.array(image), cmap='gray')
    plt.title(title)
    plt.show()

# Configuration for GPU
gpu_id = 0
device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Path to the pre-trained model
model_load_path = './models/registration_model_pretrained_0.001_32.pth'
VAE_model_load_path = './models/VAE_recon_model_pretrained.pth'

# Instantiate the model
model = Registration_Net()
model.load_state_dict(torch.load(model_load_path))
model = model.to(device)  # Move the model to GPU if available
VAE_model = MotionVAE2D(img_size=96, z_dim=32)
VAE_model.load_state_dict(torch.load(VAE_model_load_path))
VAE_model = VAE_model.to(device)
# VAE_model = VAE_model.cuda()

model.eval()  # Set the model to evaluation mode

# Load the dataset
data_path = '/media/yingmohuanzhou/Data/ACDC/database/testing'
val_set = ValDataset(data_path)
val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)

# Perform inference on the validation set
for i, (source_image, target_image, _, mask) in enumerate(val_loader):
    source_image = source_image.to(device)
    target_image = target_image.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        net = model(source_image, target_image, source_image)
        
        max_norm = 0.1
        df_gradient = compute_gradient(net['out']).to(device)
        recon, mu, logvar = VAE_model(df_gradient, mask, max_norm)
    
    # =======debug=========
    # Print the type and shape
    print(f'Type of source_image: {type(source_image)}')
    print(f'Shape of source_image: {source_image.shape}')

    # Visualize the 2D predicted deformed image
    visualize_image(source_image[0, 0, :, :].cpu().numpy())
    
    print(f'Type of target_image: {type(target_image)}')
    print(f'Shape of target_image: {target_image.shape}')
    
    # Visualize the 2D predicted deformed image
    visualize_image(target_image[0, 0, :, :].cpu().numpy())
    
    print(f'Type of mask: {type(mask)}')
    print(f'Shape of mask: {mask.shape}')
    
    # Visualize the 2D predicted deformed image
    visualize_image(mask[0, 0, :, :].cpu().numpy())
    
    # check mask
    nim_seg = nib.load('/media/yingmohuanzhou/Data/ACDC/database/testing/patient130/patient130_frame01_gt.nii.gz')
    seg = nim_seg.get_fdata()[:, :, :]
    
    print(f'Type of seg: {type(seg)}')
    print(f'Shape of seg: {seg.shape}')
    
    # Visualize the 2D predicted deformed image
    visualize_image(seg[:, :, 1])
    
    
        
    
    # Extract the predicted deformed image
    recon_mask = recon.cpu().numpy()
    
    # Print the type and shape
    print(f'Type of predicted_deformed_image: {type(recon_mask)}')
    print(f'Shape of predicted_deformed_image: {recon_mask.shape}')
    
    # Extract the 2D image from the 4D tensor
    recon_mask = recon_mask[0, 0, :, :]
    print(recon_mask)
    # Visualize the 2D predicted deformed image
    visualize_image(recon_mask, title='recon_mask')

    # Extract the predicted deformed image
    predicted_deformed_image = net['fr_st'].cpu().numpy()
    # predicted_deformed_image = net['out'].cpu().numpy()
    
    # Print the type and shape of the predicted deformed image
    print(f'Type of predicted_deformed_image: {type(predicted_deformed_image)}')
    print(f'Shape of predicted_deformed_image: {predicted_deformed_image.shape}')
    
    # Extract the 2D image from the 4D tensor
    deformed_image_2d = predicted_deformed_image[0, 0, :, :]
    print(deformed_image_2d)
    # Visualize the 2D predicted deformed image
    visualize_image(deformed_image_2d)

    # For this example, only visualize the first batch
    break