# File to contain dataset class
import os
import nibabel as nib
import numpy as np
import pickle as pkl
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset


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


def create_dataset(root_dir, data_dir, train_prop, batch_size, num_workers):
    # Create train and test datasets

    # load filenames
    f = open(os.path.join(root_dir, "filenames.pkl"), 'rb')
    filenames = pkl.load(f)
    f.close()

    # create a dataset
    dataset = BTCVDataset(root_dir=data_dir, filenames=filenames, train=True)

    # print the length of the dataset
    ds_len = dataset.__len__()
    print("Length of dataset ", ds_len)

    # Figure out the number of train and test samples
    num_train_samples = int(ds_len * train_prop)
    num_valid_samples = int(ds_len - num_train_samples)

    # create train and test datasets
    train_dataset = Subset(dataset, range(num_train_samples))
    valid_dataset = Subset(dataset, np.arange(num_train_samples, ds_len))

    print("Number of training samples: {}".format(train_dataset.__len__()))
    print("Number of validation samples: {}".format(valid_dataset.__len__()))

    f = open(os.path.join(root_dir, "filenames_ts.pkl"), 'rb')
    filenames_ts = pkl.load(f)
    f.close()

    # create a dataset
    test_dataset = BTCVDataset(root_dir=data_dir, filenames=filenames_ts, train=False)

    # Create a data loader
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1
    )

    # Create the required DataLoaders for training and testing
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True
    )

    return train_loader, valid_loader, test_loader