import os
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import numpy as np
import pickle as pkl
import time
import matplotlib.pyplot as plt

# local imports
from dataset import BTCVDataset
from loss import get_dice_per_class, dice_coeff
from UNet import UNet

# set up variables
NUM_CONV_LAYERS = 7
BATCH_SIZE = 2
NUM_WORKERS = 2
INIT_LEARNING_RATE = 3e-4
TRAIN_PROP = 0.8

# Set up directories and filenames
#root_dir = '/content/drive/My Drive/PhD/Data/nnUNet/'
root_dir = '/Users/katecevora/Documents/PhD/data/btcv'
data_dir = os.path.join(root_dir, 'nnUNet_raw_data_base/nnUNet_raw_data/Task500_BTCV')
images_dir = os.path.join(data_dir, "imagesTr")
labels_dir = os.path.join(data_dir, "labelsTr")
save_path = os.path.join(root_dir, "models")

# Check if we have a GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def create_dataset():
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
    num_train_samples = int(ds_len * TRAIN_PROP)
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
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=True
    )

    return train_loader, valid_loader, test_loader


def train(train_loader, valid_loader, name):
    epochs = 50
    av_train_error = []
    av_train_dice = []
    av_valid_error = []
    av_valid_dice = []
    eps = []

    net = UNet(inChannels=1, outChannels=13).to(device).double()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    optimizer.zero_grad()
    loss_BCE = nn.BCELoss()
    save_path = os.path.join(root_dir, "models")

    for epoch in range(epochs):

        ##########
        # Train
        ##########

        # set network to train prior to training loop
        net.train()  # this will ensure that parameters will be updated during training & that dropout will be used
        t0 = time.time()

        # reset the error log for the batch
        batch_train_error = []
        batch_train_dice = []

        # iterate over the batches in the training set
        for i, (data, label) in enumerate(train_loader):
            if i % 50 == 0:
                print("Epoch {}, batch {}".format(epoch, i))

            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            pred = net(data)

            # calculate loss
            L_dc = - dice_coeff(pred, label)
            L_ce = loss_BCE(pred, label)
            err = L_dc + L_ce
            err.backward()
            optimizer.step()

            # append to the batch errors
            batch_train_error.append(err.item())
            batch_train_dice.append(-L_dc.item())

        # Checkpoint model
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': err,
        }, os.path.join(save_path, '{}.pt'.format(name)))

        # #############
        # # Validation
        # #############
        batch_valid_error = []
        batch_valid_dice = []

        # set network to eval prior to training loop
        print("Running evaluation.....")
        net.eval()
        for i, (data, label) in enumerate(valid_loader):
            data = data.to(device)
            label = label.to(device)
            pred = net(data)

            # calculate validation loss
            dice_per_class = get_dice_per_class(pred, label)
            L_dc = - dice_coeff(pred, label)
            L_ce = loss_BCE(pred, label)
            err = L_dc + L_ce

            batch_valid_error.append(err.item())
            batch_valid_dice.append(-L_dc.item())

        # Calculate the average training and validation error for this epoch and store
        av_train_error.append(np.mean(np.array(batch_train_error)))
        av_train_dice.append(np.mean(np.array(batch_train_dice)))
        av_valid_error.append(np.mean(np.array(batch_valid_error)))
        av_valid_dice.append(np.mean(np.array(batch_valid_dice)))
        eps.append(epoch)

        # Save everything
        f = open(os.path.join(root_dir, "{}_losses.pkl".format(name)), "wb")
        pkl.dump([eps, av_train_error, av_train_dice, av_valid_error, av_valid_dice], f)
        f.close()

        print('Epoch: {0}, train error: {1:.3f}, valid error: {2:.3f}'.format(eps[-1], av_train_error[-1],
                                                                              av_valid_error[-1]))
        #print('Average dice for training batch:')
        #print('Average dice for validation batch:')


def evaluate(test_loader):
    # evaluate model performance on the test dataset
    return 1

def main():
    train_loader, valid_loader, test_loader = create_dataset()

    # Train the network
    model_name = "unet_v1"
    train(train_loader, valid_loader, model_name)

    # Evaluate saved model
    #evaluate(test_loader)


if __name__ == '__main__':
    main()

