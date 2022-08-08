import os
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from info_nce import InfoNCE
import argparse

# local imports
from dataset import create_dataset
from UNet import UNet

# NOTATION
# D is the dimensionality of the embedding space
# C is the number of classes
# H is the height of the image in pixels (512)
# W is the width of the image in pixels (512)
# B is the batch size
# N is the total number of pixels in a batch (N = BxHxW)
# Nc is the number of pixels in a batch assigned to a class c

# argparse
parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-b", "--batch_size", default=2, help="Size of UNet training batch.")
parser.add_argument("-n", "--num_epochs", default=10, help="Number of training epochs.")
parser.add_argument("-m", "--model_name", default="prototype", help="Name of the model to be saved")
parser.add_argument("-s", "--slurm", default=False, help="Running on SLURM")
parser.add_argument("-f", "--fold", default=0, help="Fold for cross-validation")
args = vars(parser.parse_args())

# set up variables
NUM_CONV_LAYERS = 7
BATCH_SIZE = int(args['batch_size'])
NUM_WORKERS = 2
NUM_EPOCHS = int(args['num_epochs'])
INIT_LEARNING_RATE = 3e-4
TRAIN_PROP = 0.8
MODEL_NAME = args['model_name']
SLURM = args['slurm']
FOLD = int(args['fold'])

# Set up directories and filenames
if SLURM:
    root_dir ='/vol/biomedic3/kc2322/data/btcv'
else:
    root_dir = '/Users/katecevora/Documents/PhD/data/btcv'
data_dir = os.path.join(root_dir, 'nnUNet_raw_data_base/nnUNet_raw_data/Task500_BTCV')
images_dir = os.path.join(data_dir, "imagesTr")
labels_dir = os.path.join(data_dir, "labelsTr")
save_path = os.path.join(root_dir, "models")

NUM_CLASSES = 14      # number of classes
NUM_PROTO = 2           # number of prototypes per class
EMBEDDED_DIMS = 14       # dimensionality of embedding space


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
    D = embed.size()[0]

    # reshape tensors to size (D, N) where N = BxHxW
    embed_flattened = torch.reshape(embed, (D, -1))     # dimensions (D, N)
    class_assignments_flattened = torch.reshape(class_assignments, (-1,))        # dimensions (N)

    # Number of pixels in batch
    N = embed_flattened.size()[1]

    # find out which classes are present in this batch
    classes_present = torch.unique(class_assignments, sorted=True)
    C = classes_present.size()[0]

    # create empty tensors to fill with positive and negative prototypes
    positive_prototypes = torch.empty((D, N), dtype=torch.double).to(device)
    negative_prototypes = torch.empty((D, C-1, N)).to(device)
    prototypes = torch.empty((D, C)).to(device)

    Ncs = []

    # for each class present, separate embedded pixels and calculate centroid
    for i in range(C):
        c = classes_present[i]
        assigned_pxls = embed_flattened[:, class_assignments_flattened==c]      # shape (D, Nc)
        Nc = assigned_pxls.size()[1]
        Ncs.append(Nc)
        prototype = torch.mean(assigned_pxls, dim=1)        # shape (D)
        prototypes[:, i] = prototype

        # multiply pixels by the prototype and sum
        prototype_tiled = torch.unsqueeze(prototype, dim=1)     # shape (D, 1)
        prototype_tiled = torch.tile(prototype_tiled, (1, Nc))  # shape (D, Nc)

        # fill
        positive_prototypes[:, class_assignments_flattened==c] = prototype_tiled

        dot_prod = torch.sum(assigned_pxls * prototype_tiled, dim=1)   # shape (Nc)
        similarity = torch.mean(dot_prod)

    # Now find the set of negative prototypes for each pixel
    for j in range(C):
        c = classes_present[j]
        Nc = Ncs[j]
        neg_proto = torch.cat((prototypes[:, :j], prototypes[:, j+1:]), dim=1)      # shape (D, C-1)

        # tile
        neg_proto_tiled = torch.unsqueeze(neg_proto, dim=2)         # shape (D, C-1, 1)
        neg_proto_tiled = torch.tile(neg_proto_tiled, (1, 1, Nc))   # shape (D, C-1, Nc)

        # fill
        negative_prototypes[:, :, class_assignments_flattened==c] = neg_proto_tiled

    # swap some dimensions around to get ready for infoNCE loss
    negative_prototypes = torch.swapaxes(negative_prototypes, 0, 2)     # shape (N, C-1, D)
    positive_prototypes = torch.swapaxes(positive_prototypes, 0, 1)     # shape (N, D)
    embed_flattened = torch.swapaxes(embed_flattened, 0, 1)             # shape (N, D)

    loss = InfoNCE(negative_mode='paired')

    output = loss(embed_flattened.double(), positive_prototypes.double(), negative_prototypes.double())

    return output


def train(train_loader, valid_loader, name, model_path):
    epochs = 1
    av_train_error = []
    av_train_ppc = []
    av_valid_error = []
    av_valid_ppc = []
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
        batch_train_ppc = []

        # iterate over the batches in the training set
        for i, (data, label) in enumerate(train_loader):
            if i % 50 == 0:
                print("Epoch {}, batch {}".format(epoch, i))

            optimizer.zero_grad()

            data = data.to(device)
            label = label.to(device)

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

            #print("L_ce: {}, L_ppc: {}, L: {}".format(L_ce.item(), L_ppc.item(), err.item()))

            # append to the batch errors
            batch_train_error.append(err.item())
            batch_train_ppc.append(L_ppc.item())

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
        batch_valid_ppc = []

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
            batch_valid_ppc.append(L_ppc.item())

        # Calculate the average training and validation error for this epoch and store
        av_train_error.append(np.mean(np.array(batch_train_error)))
        av_valid_error.append(np.mean(np.array(batch_valid_error)))
        av_train_ppc.append(np.mean(np.array(batch_train_ppc)))
        av_valid_ppc.append(np.mean(np.array(batch_valid_ppc)))
        eps.append(epoch)

        # Save everything
        f = open(os.path.join(root_dir, "{}_losses.pkl".format(name)), "wb")
        pkl.dump([eps, av_train_error, av_train_ppc, av_valid_error, av_valid_ppc], f)
        f.close()

        print('Epoch: {0}, train error: {1:.3f}, valid error: {2:.3f}'.format(eps[-1], av_train_error[-1],
                                                                              av_valid_error[-1]))


def main():
    print("Configuration:")
    print("Device: ", device)
    print("SLURM: {}".format(SLURM))
    print("Model name: {}".format(MODEL_NAME))
    print("Root dir: {}".format(root_dir))
    print("Batch size: {}".format(BATCH_SIZE))
    print("Fold: {}".format(FOLD))
    print("Number of epochs: {}".format(NUM_EPOCHS))

    # Train the network
    data_dir = os.path.join(root_dir, 'nnUNet_raw_data_base/nnUNet_raw_data/Task500_BTCV')
    model_path = os.path.join(root_dir, "models")
    base_model_path = os.path.join(model_path, "unet_v4_2.pt")

    train_loader, valid_loader, test_loader = create_dataset(root_dir, data_dir, FOLD, BATCH_SIZE, NUM_WORKERS)

    train(train_loader, valid_loader, MODEL_NAME, base_model_path)


if __name__ == '__main__':
    main()

