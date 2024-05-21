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

'''
Example case generation random selection:
Dataset: ACDC Testing
Patient: 126
Short Axis Depth/Slice: 1
'''

# specific dataloader
class ExampleDataset(data.Dataset):
    def __init__(self, data_path, filename, z):
        super(ExampleDataset, self).__init__()
        self.data_path = data_path
        self.filename = filename
        self.z = z
        
        # load 4D image
        nim = nib.load(os.path.join(data_path, filename, filename + '_4d.nii.gz'))
        image = nim.get_fdata()[:, :, :, :]
        self.img_4D = np.array(image, dtype='float32')
        print(self.img_4D.shape[3])

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


def load_data(data_path, filename, size, rand_t=None, rand_z=None, img_4d='sa.nii.gz', img_ed='sa_ED.nii.gz', seg_ed='label_sa_ED.nii.gz', debug=False):
    # print status
    if debug: print(f'rand_t:{rand_t}, rand_z:{rand_z}')
    
    # Load images and labels
    if debug: print(f'Load 4d: {os.path.join(data_path, filename, img_4d)}')
    nim = nib.load(os.path.join(data_path, filename, img_4d))
    image = nim.get_fdata()[:, :, :, :]
    image = np.array(image, dtype='float32')

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
    
    # Visualize
    if debug: visualize_image(seg_ED,title='seg_ED')

    slice = (seg[..., image.shape[2]//2] == 2).astype(np.uint8)
    centre = ndimage.center_of_mass(slice) #  Use scipy.ndimage instead of scipy.ndimage.measurements.
    centre = np.round(centre).astype(np.uint8)
    
    # Visualize the 2D predicted deformed image
    if debug: visualize_image(slice,title='slice')

    image_ED = image_ED[np.newaxis]
    seg_ED = seg_ED[np.newaxis]

    image_bank = np.concatenate((image_sa, image_ED), axis=0)

    image_bank = centre_crop(image_bank, size, centre)
    seg_ED = centre_crop(seg_ED, size, centre)
    image_bank = np.transpose(image_bank, (0, 2, 1))
    seg_ED = np.transpose(seg_ED, (0, 2, 1))
    image_bank = np.array(image_bank, dtype='float32')
    seg_ED = np.array(seg_ED, dtype='int16')
    
    # Visualize
    if debug: visualize_image(seg_ED[0,:,:],title='seg_ED')

    mask_deform = (seg_ED == 2).astype(np.uint8)
    # mask = centre_crop(mask, size, centre) TODO Why centre crop again??
    # Visualize the 2D predicted deformed image
    if debug: visualize_image(mask_deform[0,:,:],title='mask_deform')
    kernel = np.ones((3, 3), np.uint8)
    mask = ndimage.binary_dilation(mask_deform[0], kernel, iterations=3)
    mask = np.array(mask[np.newaxis], dtype='int16')
    
    # Visualize the 2D predicted deformed image
    if debug: visualize_image(mask[0,:,:],title='maskD3')

    return image_bank, mask_deform, mask #seg_ED, mask

# Function to visualize the predicted deformed image
def visualize_image(image, title=''):
    plt.imshow(np.array(image), cmap='gray')
    plt.title(title)
    plt.show()
    
def plot_four_grids(arrays, titles, masks=None, filename=''):
    """
    Plot four 2D arrays in a 2x2 grid with titles and optional masks for the bottom two images.

    Parameters:
    arrays (list of np.ndarray): List of four 2D arrays to be plotted.
    titles (list of str): List of titles for each subplot.
    masks (list of np.ndarray): List of two 2D masks for the bottom two images.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(arrays[i], cmap='gray')
        if i >= 2 and masks is not None and len(masks) >= i - 1:
            # Overlay the mask on the bottom two images
            mask = masks[i - 2]
            rgba_mask = np.zeros((*mask.shape, 4), dtype=np.float32)
            rgba_mask[..., 0] = 1.0  # Red channel
            rgba_mask[..., 3] = 0.3 * (mask != 0)  # Alpha channel: 50% transparency where mask is 1
            ax.imshow(rgba_mask, cmap='Reds')
        ax.set_title(titles[i])
        ax.axis('off')  # Hide axes

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    
    

    
    
def plot_histogram(matrix, bins=30, title='Histogram', xlabel='Value', ylabel='Frequency'):
    """
    Plot a histogram of the values in a 2D matrix.

    Parameters:
    matrix (np.ndarray): The 2D matrix.
    bins (int): Number of bins for the histogram.
    title (str): Title of the histogram.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    """
    # Flatten the 2D matrix into a 1D array
    values = matrix.flatten()
    
    # Plot the histogram
    plt.hist(values, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
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
filename = 'patient126'
z = int(4)
example_set = ExampleDataset(data_path, filename, z)
example_loader = DataLoader(dataset=example_set, batch_size=1, shuffle=False)

# save image
out_folder = '/media/yingmohuanzhou/Work/OneDrive - University of Cambridge/Uni/PhD4/CMR_Motion/Biomechanics/Original'

# Perform inference on the validation set
for i, (source_image, target_image, mask_deform, mask) in enumerate(example_loader):
    print(f'Loop: {i}')
    source_image = source_image.to(device)
    target_image = target_image.to(device)
    mask = mask.to(device)
    mask_deform = mask_deform.to(device)

    with torch.no_grad():
        net = model(source_image, target_image, source_image, mask_deform)
        
        max_norm = 0.1
        df_gradient = compute_gradient(net['out']).to(device)
        recon, mu, logvar = VAE_model(df_gradient, mask, max_norm)
    
    # =======debug=========
    # Print the type and shape
    print(f'Type of source_image: {type(source_image)}')
    print(f'Shape of source_image: {source_image.shape}')

    # Visualize the 2D predicted deformed image
    # visualize_image(source_image[0, 0, :, :].cpu().numpy(), title='source_image')
    
    print(f'Type of target_image: {type(target_image)}')
    print(f'Shape of target_image: {target_image.shape}')
    
    # Visualize the 2D predicted deformed image
    # visualize_image(target_image[0, 0, :, :].cpu().numpy(), title='target_image')
    
    print(f'Type of mask: {type(mask)}')
    print(f'Shape of mask: {mask.shape}')
    
    # Visualize the 2D predicted deformed image
    # visualize_image(mask[0, 0, :, :].cpu().numpy(), title='mask')
    
    # # check mask
    # nim_seg = nib.load('/media/yingmohuanzhou/Data/ACDC/database/testing/patient130/patient130_frame01_gt.nii.gz')
    # seg = nim_seg.get_fdata()[:, :, :]
    
    # print(f'Type of seg: {type(seg)}')
    # print(f'Shape of seg: {seg.shape}')
    
    # # Visualize the 2D predicted deformed image
    # visualize_image(seg[:, :, 1])
        
    # Extract the predicted deformed image
    recon_mask = recon.cpu().numpy()
    
    # Print the type and shape
    print(f'Type of predicted_deformed_image: {type(recon_mask)}')
    print(f'Shape of predicted_deformed_image: {recon_mask.shape}')
    
    # Extract the 2D image from the 4D tensor
    recon_mask = recon_mask[0, 0, :, :]
    print(recon_mask)
    # Visualize the 2D predicted deformed image
    # visualize_image(recon_mask, title='recon_mask')
    
    # Print the type and shape of the predicted deformed image
    df_gradient = compute_gradient(net['out'])
    df_gradient_mask=df_gradient*mask

    print(f'Shape of df_gradient: {df_gradient.shape}')
    print(f'Shape of df_gradient_mask: {df_gradient_mask.shape}')

    # Extract the predicted deformed image
    predicted_deformed_image = net['fr_st'].cpu().numpy()

    
    # Print the type and shape of the predicted deformed image
    print(f'Type of predicted_deformed_image: {type(predicted_deformed_image)}')
    print(f'Shape of predicted_deformed_image: {predicted_deformed_image.shape}')
    
    # Extract the 2D image from the 4D tensor
    deformed_image_2d = predicted_deformed_image[0, 0, :, :]
    # print(deformed_image_2d)
    # Visualize the 2D predicted deformed image
    # visualize_image(deformed_image_2d, title='deformed_image_2d')

    # Extract the predicted deformed mask
    predicted_deformed_mask = net['reverse_mask'].cpu().numpy()
    
    # Print the type and shape of the predicted deformed image
    print(f'Type of predicted_deformed_mask: {type(predicted_deformed_mask)}')
    print(f'Shape of predicted_deformed_mask: {predicted_deformed_mask.shape}')
    
    # Extract the 2D image from the 4D tensor
    predicted_deformed_mask_2d = predicted_deformed_mask[0, 0, :, :]
    # Visualize the 2D predicted deformed image
    # visualize_image(predicted_deformed_mask_2d, title='predicted_deformed_mask_2d')
    
    mask_ED_2d = mask_deform[0, 0, :, :].cpu().numpy()
    img_ED_2d = target_image[0, 0, :, :].cpu().numpy()
    predicted_deformed_mask_2d = net['reverse_mask'][0, 0, :, :].cpu().numpy()
    deformed_image_2d = net['fr_st'][0, 0, :, :].cpu().numpy()
    origninal_image_2d = source_image[0, 0, :, :].cpu().numpy()

    print(np.median(mask_ED_2d))
    # plot_histogram(predicted_deformed_mask_2d)
    
    plot_four_grids(arrays=[img_ED_2d, deformed_image_2d, img_ED_2d, origninal_image_2d], 
                    titles=[f'Image ED Time={i}', f'Registered Image Time={i}', 'ED Image & Mask', 'Image & Deformed Mask'],
                    masks=[mask_ED_2d, predicted_deformed_mask_2d, mask_ED_2d, predicted_deformed_mask_2d],
                    filename = out_folder+f'/ACDC_{filename}_Z{z}_T{i}.png')
    
    # For this example, only visualize the first batch
    # if i == 16: break