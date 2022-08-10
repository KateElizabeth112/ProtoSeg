# evaulate model performance on the test set
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import torch
from loss import get_dice_per_class
from UNet import UNet
from dataset import create_test_dataset
from plotting import PlotOverlay2D
from monai.metrics import compute_surface_dice
import nibabel as nib

DEBUG = False
MAKE_PLOTS = True
PROTO = True

root_dir = '/Users/katecevora/Documents/PhD/data/btcv'
if PROTO:
    output_folder = "/Users/katecevora/Documents/PhD/data/btcv/Images/Proto/Test"
else:
    output_folder = "/Users/katecevora/Documents/PhD/data/btcv/Images/UNet/Test"
data_dir = os.path.join(root_dir, 'nnUNet_raw_data_base/nnUNet_raw_data/Task500_BTCV')
model_path = os.path.join(root_dir, "models")
model_name = "prototype_v1_0.pt"
fold = "0"

organs_dict = {0: "background",
               1: "spleen",
               2: "right kidney",
               3: "left kidney",
               4: "gallbladder",
               5: "esophagus",
               6: "liver",
               7: "stomach",
               8: "aorta",
               9: "inferior vena cava",
               10: "portal & splenic vein",
               11: "pancreas",
               12: "right adrenal gland",
               13: "left adrenal gland"}


def get_surface_dice(y_pred, y, class_thresholds):
    # covert predictions to one hot encoding
    max_idx = torch.argmax(y_pred, 1, keepdim=True)
    one_hot = torch.FloatTensor(y_pred.shape)
    one_hot.zero_()
    one_hot.scatter_(1, max_idx, 1)
    res = compute_surface_dice(one_hot, y, class_thresholds, include_background=True, distance_metric='euclidean')

    res = res.numpy()
    res = res.reshape(-1)
    return res


def evaluate(test_loader, model_path, fold):
    # Check if we have a GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load the model
    net = UNet(inChannels=1, outChannels=14).to(device).double()

    checkpoint = torch.load(model_path, map_location=torch.device(device))
    net.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    net.eval()

    # load the filenames of test data
    f = open(os.path.join(root_dir, "filenames_ts.pkl"), 'rb')
    filenames_ts = pkl.load(f)
    f.close()

    # create an empty dictionary to store results
    results_dict = {}
    dice_all = np.zeros(14)
    dice_all.fill(np.nan)
    nsd_all = np.zeros(14)
    nsd_all.fill(np.nan)

    for j, (data, lab) in enumerate(test_loader):
        pred, _ = net(data.to(device))
        dice = get_dice_per_class(pred, lab.to(device)).cpu().detach().numpy()
        nsd = get_surface_dice(pred, lab.to(device), [1.5 for i in range(14)])

        pred = pred.cpu().detach().numpy()
        lab = lab.cpu().detach().numpy()
        img = data.cpu().detach().numpy()

        # Make an output directory if one doesn't exist
        name = filenames_ts[j].split('_')[0]
        path = os.path.join(output_folder, fold, name)
        if not os.path.exists(path):
            os.makedirs(path)

        print("Processing {}".format(name))

        # add dice scores to results dictionary
        res = {"dice": dice,
               "distance": nsd}

        results_dict[name] = res
        dice_all = np.vstack((dice_all, dice))
        nsd_all = np.vstack((nsd_all, nsd))

        # Make some plots
        if MAKE_PLOTS:
            PlotOverlay2D(img[0, :, :, :], pred[0, :, :, :], lab[0, :, :, :], 13, alpha=0.5, c='jet', save=True,
                          path=path)

            # Look at the different channels of the prediction
            for i in range(pred.shape[1]):
                pred_channel = pred[0, i, :, :]
                lab_channel = lab[0, i, :, :]

                plt.figure(figsize=(10, 3))
                plt.suptitle("Channel {}: {}".format(i, organs_dict[i]))
                fontsize = 12
                plt.subplot(1, 2, 1)
                plt.imshow(pred_channel, vmin=0, vmax=1)
                plt.title("Predictions")
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(lab_channel, vmin=0, vmax=1)
                plt.title("Ground Truth")
                plt.axis('off')
                plt.axis('equal')

                # save in specified directory
                if not DEBUG:
                    plt.savefig(os.path.join(path, "channel_{}.png".format(i)))
                else:
                    plt.show()

    # Calculate averages and save results dictionary
    dice_av = np.nanmean(dice_all, axis=0)
    dice_std = np.nanstd(dice_all, axis=0)
    nsd_av = np.nanmean(nsd_all, axis=0)
    nsd_std = np.nanstd(nsd_all, axis=0)

    results_dict["Average Dice"] = (dice_av, dice_std)
    results_dict["Average Distance"] = (nsd_av, nsd_std)

    f = open(os.path.join(output_folder, fold, "results_dict.pkl"), "wb")
    pkl.dump(results_dict, f)
    f.close()

    for k in range(14):
        print("{} dice: {}, nsd: {}".format(organs_dict[k], dice_av[k], nsd_av[k]))


def evaluate_nnUNet():
    # Check if we have a GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load the filenames of test data
    pred_folder = os.path.join(root_dir, "inference/Ts")
    labels_folder = os.path.join(root_dir, "nnUNet_raw_data_base/nnUNet_raw_data/Task500_BTCV/labelsTs")
    images_folder = os.path.join(root_dir, "nnUNet_raw_data_base/nnUNet_raw_data/Task500_BTCV/imagesTs")
    output_folder = '/Users/katecevora/Documents/PhD/data/btcv/Images/nnUNet/Test/'
    files = os.listdir(pred_folder)

    # create an empty dictionary to store results
    results_dict = {}
    dice_all = np.zeros(14)
    dice_all.fill(np.nan)
    nsd_all = np.zeros(14)
    nsd_all.fill(np.nan)

    for i in range(len(files)):
        if files[i].endswith(".gz"):
            if i % 100 == 0:
                print("Processing file {}".format(i))
            # load predictions (shape 512, 512, 1)
            pred = nib.load(os.path.join(pred_folder, files[i])).get_fdata()

            # load ground truth labels (shape 512, 512, 1)
            lab = nib.load(os.path.join(labels_folder, files[i])).get_fdata()

            # extract the filename (shape 512, 512, 1)
            name = files[i].split('.')[0]
            image_name = name + "_0000.nii.gz"
            img = nib.load(os.path.join(images_folder, image_name)).get_fdata()

            # one hot encode y and y_pred
            num_channels = 14
            lab_full = np.zeros((lab.shape[0], lab.shape[1], num_channels))
            pred_full = np.zeros((pred.shape[0], pred.shape[1], num_channels))

            for c in range(num_channels):
                lab_full[:, :, c][lab[:, :, 0] == c] = 1
                pred_full[:, :, c][pred[:, :, 0] == c] = 1

            # swap channels to the first dimension as pytorch expects
            img = np.swapaxes(img, 0, 2)   # (shape 1, 512, 512)
            lab_full = np.swapaxes(lab_full, 0, 2)      # (shape 14, 512, 512)
            pred_full = np.swapaxes(pred_full, 0, 2)    # (shape 14, 512, 512)

            # Calculate Dice and NSD
            lab_tensor = torch.from_numpy(lab_full).to(device).unsqueeze(0)
            pred_tensor = torch.from_numpy(pred_full).to(device).unsqueeze(0)
            dice = get_dice_per_class(pred_tensor, lab_tensor)
            dice = dice.cpu().detach().numpy()
            nsd = get_surface_dice(pred_tensor, lab_tensor, [1.5 for i in range(14)])

            # Make an output directory if one doesn't exist
            path = os.path.join(output_folder, fold, name)
            if not os.path.exists(path):
                os.makedirs(path)

            print("Processing {}".format(name))

            # add dice scores to results dictionary
            res = {"dice": dice,
                   "distance": nsd}

            results_dict[name] = res
            dice_all = np.vstack((dice_all, dice))
            nsd_all = np.vstack((nsd_all, nsd))

            # Make some plots
            if MAKE_PLOTS:
                PlotOverlay2D(img, pred_full, lab_full, 13, alpha=0.5, c='jet', save=True, path=path)

                # Look at the different channels of the prediction
                for i in range(pred_full.shape[0]):
                    pred_channel = pred_full[i, :, :]
                    lab_channel = lab_full[i, :, :]

                    plt.figure(figsize=(10, 3))
                    plt.suptitle("Channel {}: {}".format(i, organs_dict[i]))
                    fontsize = 12
                    plt.subplot(1, 2, 1)
                    plt.imshow(pred_channel, vmin=0, vmax=1)
                    plt.title("Predictions")
                    plt.axis('off')

                    plt.subplot(1, 2, 2)
                    plt.imshow(lab_channel, vmin=0, vmax=1)
                    plt.title("Ground Truth")
                    plt.axis('off')
                    plt.axis('equal')

                    # save in specified directory
                    if not DEBUG:
                        plt.savefig(os.path.join(path, "channel_{}.png".format(i)))
                    else:
                        plt.show()

    # Calculate averages and save results dictionary
    dice_av = np.nanmean(dice_all, axis=0)
    dice_std = np.nanstd(dice_all, axis=0)
    nsd_av = np.nanmean(nsd_all, axis=0)
    nsd_std = np.nanstd(nsd_all, axis=0)

    results_dict["Average Dice"] = (dice_av, dice_std)
    results_dict["Average Distance"] = (nsd_av, nsd_std)

    f = open(os.path.join(output_folder, fold, "results_dict.pkl"), "wb")
    pkl.dump(results_dict, f)
    f.close()

    for k in range(14):
        print("{} dice: {}, nsd: {}".format(organs_dict[k], dice_av[k], nsd_av[k]))


def main():
    #test_loader = create_test_dataset(root_dir, data_dir)
    #evaluate(test_loader, os.path.join(model_path, model_name), fold)

    evaluate_nnUNet()

if __name__ == "__main__":
    main()