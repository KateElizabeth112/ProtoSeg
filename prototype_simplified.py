import os
import torch
import torch.nn as nn
from torch import autograd
import numpy as np
import pickle as pkl
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# local imports
from dataset import create_dataset
from loss import get_dice_per_class, dice_coeff
from UNet import UNet

# NOTATION
# D is the dimensionality of the embedding space
# C is the number of classes
# H is the height of the image in pixels (512)
# W is the width of the image in pixels (512)
# B is the batch size
# N is the total number of pixels in a batch (N = BxHxW)
# Nc is the number of pixels in a batch assigned to a class c

# set up variables
NUM_CONV_LAYERS = 7
BATCH_SIZE = 2
NUM_WORKERS = 2
INIT_LEARNING_RATE = 3e-4
TRAIN_PROP = 0.8
FOLD = 0

NUM_CLASSES = 14      # number of classes
NUM_PROTO = 2           # number of prototypes per class
EMBEDDED_DIMS = 14       # dimensionality of embedding space

# Set up directories and filenames
# root_dir = '/vol/biomedic3/kc2322/data/btcv'
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


def loss_PPC(embed, probs):
    # probs has dimensions (B, C, H, W). These are the pixel-wise class probabilities assigned by the UNet
    # embed has dimensions (B, D, H, W). These are the pixel embeddings taken from the last layer of the UNet
    # we return the class centroids (equivalent to prototypes)

    # convert probabilities to class assignments
    class_assignments = torch.argmax(probs, 1)        # dimensions (B, H, W)
    embed = torch.swapaxes(embed, 0, 1)                # dimensions (D, B, H, W)

    classes_present = torch.unique(class_assignments, sorted=True)

    # for each class present, separate embedded pixels and calculate centroid
    for c in classes_present:
        assigned_pxls = embed[:, class_assignments==c]      # shape (D, Nc)
        Nc = assigned_pxls.size()[1]
        prototype = torch.mean(assigned_pxls, dim=1)        # shape (D)

        # multiply pixels by the prototype and sum
        prototype_tiled = torch.unsqueeze(prototype, dim=1)     # shape (D, 1)
        prototype_tiled = torch.tile(prototype_tiled, (1, Nc))  # shape (D, Nc)

        dot_prod = torch.sum(assigned_pxls * prototype_tiled, dim=1)   # shape (Nc)
        similarity = torch.mean(dot_prod)

    return similarity


def train(train_loader, valid_loader, name, model_path):
    epochs = 1
    av_train_error = []
    av_train_dice = []
    av_valid_error = []
    av_valid_dice = []
    eps = []

    # create a UNet instance and load pre-trained weights
    net = UNet(inChannels=1, outChannels=NUM_CLASSES).to(device).double()
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    net.load_state_dict(checkpoint["model_state_dict"])
    starting_epoch = checkpoint["epoch"]

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

        # reset the error log for the batch
        batch_train_error = []
        batch_train_dice = []

        # iterate over the batches in the training set
        for i, (data, label) in enumerate(train_loader):
            print("Epoch {}, batch {}".format(epoch, i))

            optimizer.zero_grad()
            data = data[:, :, :64, :64].to(device)
            label = label[:, :, :64, :64].to(device)

            # get the embedded representations and the predicted class probabilities
            embed, probs = net(data)

            # calculate the cross-entropy loss between predicted class probabilities and labels
            L_ce = loss_BCE(probs, label)

            # calculate pixel-prototype contrastive loss
            L_ppc = loss_PPC(embed, probs)

            # Add losses and backpropagate to update network params
            err = L_ce + L_ppc
            err.backward()
            optimizer.step()

            # append to the batch errors
            batch_train_error.append(err.item())

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

            # get the embedded representations and the predicted class probabilities
            embed, probs = net(data)

            # calculate the cross-entropy loss between predicted class probabilities and labels
            L_ce = loss_BCE(probs, label)

            # calculate pixel-prototype contrastive loss
            L_ppc = loss_PPC(embed, probs)

            err = L_ppc + L_ce

            batch_valid_error.append(err.item())

        # Calculate the average training and validation error for this epoch and store
        av_train_error.append(np.mean(np.array(batch_train_error)))
        av_valid_error.append(np.mean(np.array(batch_valid_error)))
        eps.append(epoch)

        # Save everything
        f = open(os.path.join(root_dir, "{}_losses.pkl".format(name)), "wb")
        pkl.dump([eps, av_train_error, av_train_dice, av_valid_error, av_valid_dice], f)
        f.close()

        print('Epoch: {0}, train error: {1:.3f}, valid error: {2:.3f}'.format(eps[-1], av_train_error[-1],
                                                                              av_valid_error[-1]))


def main():
    print(device)

    # Train the network
    root_dir = '/Users/katecevora/Documents/PhD/data/btcv'
    data_dir = os.path.join(root_dir, 'nnUNet_raw_data_base/nnUNet_raw_data/Task500_BTCV')
    model_path = os.path.join(root_dir, "models")
    model_name = "prototype_v1"
    base_model_path = os.path.join(model_path, "unet_v4_2.pt")

    train_loader, valid_loader, test_loader = create_dataset(root_dir, data_dir, FOLD, BATCH_SIZE, NUM_WORKERS)

    train(train_loader, valid_loader, model_name, base_model_path)


if __name__ == '__main__':
    main()

