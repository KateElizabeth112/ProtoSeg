import os
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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


def initialise_classes_and_prototypes(x, num_classes, num_prototypes):
    # initialse the classes and prototypes by randomly dividing data
    batch_size = x.shape[0]
    num_dims = x.shape[1]

    # flatten the batch and pixel dimensions of x
    x = x.reshape((batch_size, num_dims, -1))   # shape (B, D, N)
    num_pixels = x.shape[-1]

    # choose indices for each class
    indices = np.arange(0, num_pixels)
    num_pxls_per_class = int(np.floor(num_pixels / num_classes))

    # assign prototypes
    prototypes = np.zeros((num_classes, num_prototypes, num_dims))

    for c in range(num_classes):
        # select pixels without replacement from list
        idx = np.random.randint(0, high=indices.shape[0], size=num_pxls_per_class)
        pxl_idx = indices[idx]
        indices = np.delete(indices, idx)
        pxls = x[0, :, pxl_idx]

        # now run k-means to find the prototypes
        kmeans = KMeans(n_clusters=2, random_state=0).fit(pxls)
        print(kmeans.cluster_centers_.shape)
        prototypes[c, :, :] = kmeans.cluster_centers_

    return prototypes


def assign_classes(x, p):
    # assign classes to each embedded pixel based on nearest prototype
    batch_size = x.shape[0]
    num_dims = x.shape[1]
    num_pixels = x.shape[2] * x.shape[3]

    num_classes = p.shape[0]
    num_prototypes = p.shape[1]

    # flatten the pixel dimensions
    x = np.reshape(x, (batch_size, num_dims, -1))   # shape (B, D, N)

    # tile CK times
    x = np.repeat(x[:, :, np.newaxis, :], num_classes * num_prototypes, axis=2)     # shape (B, D, C*K, N)

    # tile prototypes to same size as x
    # prototype shape is (C, K, D)
    p = p.swapaxes(0, 2)    # shape (D, C, K)
    p = np.reshape(p, (num_dims, -1))   # shape (D, C*K)
    p = np.repeat(p[:, :, np.newaxis], num_pixels, axis=2)  # shape (D, C*K, N)
    p = np.repeat(p[np.newaxis, :, :, :], batch_size, axis=0)   # shape (B, D, C*K, N)

    # calculate euclidean distance between p and x
    d = np.power(p - x, 2)
    d = np.sum(d, axis=1)
    d = np.power(d, 0.5)        # shape is (B, C*K, N)

    proto_assignments = np.argmin(d, axis=1)
    class_assignments = np.floor(proto_assignments / num_prototypes)

    return class_assignments


def update_prototypes(x, class_assignments, prototypes):
    # split pixels by class assignment and calculate new prototypes


    return prototypes


def train(train_loader, valid_loader, name):
    epochs = 1
    av_train_error = []
    av_train_dice = []
    av_valid_error = []
    av_valid_dice = []
    eps = []


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

            # initalise prototypes on first iteration
            if (epoch == 0) and (i == 0):
                embed = embed.detach().numpy()
                prototypes = initialise_classes_and_prototypes(embed, NUM_CLASSES, NUM_PROTO)

            # look at distribution of embedding space
            print(embed.shape)
            assign_classes(embed, prototypes)

            plt.scatter(embed[0, 0, :, :].flatten(), embed[0, 1, :, :].flatten())
            for c in range(NUM_CLASSES):
                plt.scatter(prototypes[c, :, 0], prototypes[c, :, 1])
            plt.show()

            # Update prototypes based on class assigmnents




def main():
    print(device)

    train_loader, valid_loader, test_loader = create_dataset(root_dir, data_dir, TRAIN_PROP, BATCH_SIZE, NUM_WORKERS)

    # Train the network
    model_name = "prototype_v1"
    train(train_loader, valid_loader, model_name)


if __name__ == '__main__':
    main()

