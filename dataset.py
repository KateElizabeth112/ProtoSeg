# File to contain dataset class
import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class BTCVDataset(Dataset):
    def __init__(self, root_dir, filenames, train=True):
        '''
          root_dir - string - path towards the folder containg the data
        '''
        # Save the root_dir as a class variable
        self.root_dir = root_dir
        self.train_imgs = os.path.join(root_dir, "imagesTr")
        self.train_labels = os.path.join(root_dir, "labelsTr")

        # Save the filenames in the root_dir as a class variable
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Fetch file filename
        img_name = self.filenames[idx]
        label_name = img_name.split('_')[0] + ".nii.gz"

        # Load the nifty image
        img = nib.load(os.path.join(self.train_imgs, img_name))
        lab = nib.load(os.path.join(self.train_labels, label_name))

        # Get the voxel values as a numpy array
        img = np.array(img.get_fdata())
        lab = np.array(lab.get_fdata())

        # Expand the label to 13 channels
        num_channels = 13
        lab_full = np.zeros((lab.shape[0], lab.shape[1], num_channels))

        for c in range(num_channels):
            lab_full[:, :, c][lab[:, :, 0] == c + 1] = 1

        # swap channels to the first dimension as pytorch expects
        img = torch.tensor(np.swapaxes(img, 0, 2)).double()
        lab_full = torch.tensor(np.swapaxes(lab_full, 0, 2)).double()

        return img, lab_full