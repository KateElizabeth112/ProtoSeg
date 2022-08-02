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

# set up variables
NUM_CONV_LAYERS = 7
BATCH_SIZE = 2
NUM_WORKERS = 2
INIT_LEARNING_RATE = 3e-4
TRAIN_PROP = 0.8
FOLD = 0

NUM_CLASSES = 14      # number of classes
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
        idx = np.random.choice(range(indices.shape[0]), num_pxls_per_class, replace=False)
        pxl_idx = indices[idx]
        indices = np.delete(indices, idx)
        pxls = x[0, :, pxl_idx]

        # now run k-means to find the prototypes
        kmeans = KMeans(n_clusters=num_prototypes, random_state=0).fit(pxls)
        prototypes[c, :, :] = kmeans.cluster_centers_

    return prototypes


def assign_classes(x, p):
    # assign classes to each embedded pixel based on nearest prototype
    # x and p are tensors
    batch_size = x.size()[0]
    num_dims = x.size()[1]
    num_pixels = x.size()[2] * x.size()[3]

    num_classes = p.size()[0]
    num_prototypes = p.size()[1]

    # flatten the pixel dimensions
    x = torch.reshape(x, (batch_size, num_dims, -1))   # shape (B, D, N)

    # tile C, K times
    x = torch.unsqueeze(x, dim=2)   # shape (B, D, 1, N)
    x = torch.unsqueeze(x, dim=2)   # shape (B, D, 1, 1, N)
    x = torch.tile(x, (1, 1, num_classes, num_prototypes, 1))     # shape (B, D, C, K, N)

    # tile prototypes to same size as x
    # prototype shape is (C, K, D)
    p = p.moveaxis(2, 0)    # shape (D, C, K)
    p = torch.unsqueeze(p, dim=3)           # shape (D, C, K, 1)
    p = torch.unsqueeze(p, dim=0)           # shape (1, D, C, K, 1)
    p = torch.tile(p, (batch_size, 1, 1, 1, num_pixels))  # shape (B, D, C, K, N)

    # Calculate cosine distance between p and x
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    dist = cos(p, x)        # shape (B, C, K, N)

    # find most similar prototype for each class (will be used to calculate loss)
    (dist_proto_max, dist_proto_argmax) = torch.max(dist, dim=2)     # shape (B, C, N)

    # find most similar class for each pixel
    dist_class_argmax = torch.argmax(dist_proto_max, dim=1)     # shape is (B, N)

    return dist_class_argmax, dist_proto_max


def update_prototypes(x, class_assignments, prototypes):
    # split pixels by class assignment and calculate new prototypes
    batch_size = x.shape[0]
    num_dims = x.shape[1]
    num_classes = prototypes.shape[0]
    num_prototypes = prototypes.shape[1]
    prototypes_update = np.zeros(prototypes.shape)

    # flatten x in the pixel dimensions
    x = np.reshape(x, (batch_size, num_dims, -1))  # shape (B, D, N)
    x = np.swapaxes(x, 0, 1)       # shape (D, B, N)

    # iterate over classes and update prototypes
    for c in range(num_classes):
        # slice x by class c
        x_class = x[:, class_assignments==c]
        x_class = np.swapaxes(x_class, 0, 1)

        # update prototypes
        if np.isnan(x_class.sum()):
            print("Whoops")
        kmeans = KMeans(n_clusters=num_prototypes, random_state=0).fit(x_class)
        prototypes_update[c, :, :] = kmeans.cluster_centers_

    return prototypes_update


def cross_entropy_loss(prototype_similarities, label):
    # prototype similarities have shape (B, C, N)
    # label has shape (B, C, N^0.5, N^0,5)
    batch_size = label.size()[0]
    num_classes = label.size()[1]

    # reshape label
    label = torch.reshape(label, (batch_size, num_classes, -1)) # shape (B, C, N)

    # softmax prototype similarities across classes
    softmax = nn.Softmax(dim=1)
    prototype_similarities = softmax(prototype_similarities)  # shape (B, C, N)

    # calculate cross entropy loss
    loss_BCE = nn.BCELoss()
    L_ce = loss_BCE(label, prototype_similarities)

    return L_ce


def train(train_loader, valid_loader, name):
    epochs = 1
    av_train_error = []
    av_train_dice = []
    av_valid_error = []
    av_valid_dice = []
    eps = []

    net = UNet(inChannels=1, outChannels=EMBEDDED_DIMS).to(device).double()
    #optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-5)
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
            print("Epoch {}, batch {}".format(epoch, i))

            optimizer.zero_grad()
            data = data[:, :, :64, :64].to(device)
            label = label[:, :, :64, :64].to(device)
            embed = net(data)

            # initalise prototypes on first iteration
            if (epoch == 0) and (i == 0):
                embed = embed.detach().numpy()
                prototypes = initialise_classes_and_prototypes(embed, NUM_CLASSES, NUM_PROTO)

                continue

            else:
                # if we already have initialised prototypes, assign classes based on embeddings
                class_assignments, prototype_similarities = assign_classes(embed, torch.from_numpy(prototypes))

                if False:
                    # look at distribution of embedding space
                    plt.scatter(embed[0, 0, :, :].flatten(), embed[0, 1, :, :].flatten())
                    for c in range(NUM_CLASSES):
                        plt.scatter(prototypes[c, :, 0], prototypes[c, :, 1])
                    plt.show()

                # Update prototypes based on class assigmnents
                prototypes = update_prototypes(embed.detach().numpy(), class_assignments, prototypes)

                # Calculate losses
                L_ce = cross_entropy_loss(prototype_similarities, label)
                print(L_ce)
                with autograd.detect_anomaly():
                    L_ce.backward()
                optimizer.step()

                # Check for NaNs
                for name, param in net.named_parameters():
                    if not(torch.isfinite(param.grad).all()):
                        print(name, torch.isfinite(param.grad).all())
                        break






def main():
    print(device)

    train_loader, valid_loader, test_loader = create_dataset(root_dir, data_dir, FOLD, BATCH_SIZE, NUM_WORKERS)

    # Train the network
    model_name = "prototype_v1"
    train(train_loader, valid_loader, model_name)


if __name__ == '__main__':
    main()

