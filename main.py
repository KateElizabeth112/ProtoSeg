import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl

# local imports
from dataset import create_dataset
from loss import get_dice_per_class, dice_coeff
from UNet import UNet

# argparse
parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-b", "--batch_size", default=2, help="Size of UNet training batch.")
parser.add_argument("-n", "--num_epochs", default=10, help="Number of training epochs.")
parser.add_argument("-m", "--model_name", default="unet", help="Name of the model to be saved")
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

# Check if we have a GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def train(train_loader, valid_loader, name):
    av_train_error = []
    av_train_dice = []
    av_valid_error = []
    av_valid_dice = []
    eps = []

    net = UNet(inChannels=1, outChannels=14).to(device).double()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    optimizer.zero_grad()
    loss_BCE = nn.BCELoss()
    save_path = os.path.join(root_dir, "models")

    for epoch in range(NUM_EPOCHS):

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
    # Check if we have a GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Configuration:")
    print("Device: ", device)
    print("SLURM: {}".format(SLURM))
    print("Model name: {}".format(MODEL_NAME))
    print("Root dir: {}".format(root_dir))
    print("Batch size: {}".format(BATCH_SIZE))
    print("Fold: {}".format(FOLD))
    print("Number of epochs: {}".format(NUM_EPOCHS))

    train_loader, valid_loader, test_loader = create_dataset(root_dir, data_dir, FOLD, BATCH_SIZE, NUM_WORKERS)

    # Train the network
    train(train_loader, valid_loader, MODEL_NAME)

    # Evaluate saved model
    #evaluate(test_loader)


if __name__ == '__main__':
    main()

