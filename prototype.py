import os
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import time
import matplotlib.pyplot as plt

# local imports
from dataset import create_dataset
from loss import get_dice_per_class, dice_coeff
from UNet import UNet

# set up variables
NUM_CONV_LAYERS = 7
BATCH_SIZE = 2
NUM_WORKERS = 2
INIT_LEARNING_RATE = 3e-4
TRAIN_PROP = 0.8

NUM_CLASSES = 13      # number of classes
NUM_PROTO = 2           # number of prototypes per class
EMBEDDED_DIMS = 2       # dimensionality of embedding space

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


def initialise_prototypes(num_classes, num_prototypes, embedded_dims):
    # randomly initalise the classes and prototypes in the range (0, 1)
    prototypes = np.zeros((num_classes, num_prototypes, embedded_dims))
    sigma1 = 0.01
    sigma2 = 0.001

    for c in range(num_classes):
        class_centre = np.random.normal(0.5, sigma1, size=embedded_dims)

        # initialise prototypes as a small deviation from the class centre
        p = np.random.normal(0, sigma2, size=(num_prototypes, embedded_dims)) + np.tile(class_centre, num_prototypes).reshape((num_prototypes, embedded_dims))

        prototypes[c, :, :] = p

    return prototypes


def assign_classes(x, p):
    # assign classes to each embedded pixel based on nearest prototype
    batch_size = x.shape[0]
    num_dims = x.shape[1]
    num_pixels = batch_size *x.shape[2] * x.shape[3]

    num_classes = p.shape[0]
    num_prototypes = p.shape[1]

    # flatten the pixel dimensions
    x = np.reshape(x, (batch_size, num_dims, -1))

    print(x.shape)

    # tile CK times
    x = np.reshape(np.tile(x, num_prototypes * num_classes), (batch_size, num_dims, num_pixels, -1))

    print(x.shape)

    # tile prototypes to same size as x
    p = p.swapaxes(0, 1)
    p = np.reshape(p, (num_dims, -1))
    p = np.reshape(np.tile(p, num_pixels), (batch_size, num_dims, num_prototypes * num_classes, -1))

    print(p.shape)


def train(train_loader, valid_loader, name):
    epochs = 1
    av_train_error = []
    av_train_dice = []
    av_valid_error = []
    av_valid_dice = []
    eps = []

    prototypes = initialise_prototypes(NUM_CLASSES, NUM_PROTO, EMBEDDED_DIMS)

    net = UNet(inChannels=1, outChannels=EMBEDDED_DIMS).to(device).double()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    optimizer.zero_grad()
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
            if i % 50 == 0:
                print("Epoch {}, batch {}".format(epoch, i))

            optimizer.zero_grad()
            data = data[:, :, :64, :64].to(device)
            label = label[:, :, :64, :64].to(device)
            embed = net(data)

            # look at distribution of embedding space
            embed = embed.detach().numpy()
            print(embed.shape)

            assign_classes(embed, prototypes)

            plt.scatter(embed[0, 0, :, :].flatten(), embed[0, 1, :, :].flatten())
            for c in range(NUM_CLASSES):
                plt.scatter(prototypes[c, :, 0], prototypes[c, :, 1])
            plt.show()

            # Calculate the distance from each pixel to each prototype



def main():
    print(device)

    train_loader, valid_loader, test_loader = create_dataset(root_dir, data_dir, TRAIN_PROP, BATCH_SIZE, NUM_WORKERS)

    # Train the network
    model_name = "prototype_v1"
    train(train_loader, valid_loader, model_name)


if __name__ == '__main__':
    main()

