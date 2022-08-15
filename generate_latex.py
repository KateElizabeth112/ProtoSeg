# a script  to generate latex snippets to include in the report
import pickle as pkl
import os


unet_res_folder = "/Users/katecevora/Documents/PhD/data/btcv/Images/UNet/Test"
nnunet_res_folder = "/Users/katecevora/Documents/PhD/data/btcv/Images/nnUNet/Test"
proto_res_folder = "/Users/katecevora/Documents/PhD/data/btcv/Images/Proto/Test"

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
               10: "portal \& splenic vein",
               11: "pancreas",
               12: "right adrenal gland",
               13: "left adrenal gland"}


def print_results():
    for fold in range(3):
        f = open(os.path.join(unet_res_folder, str(fold), "results_dict.pkl"), "rb")
        unet_results_dict = pkl.load(f)
        f.close()

        (av_dice, std_dice) = unet_results_dict["Average Dice"]
        (av_nsd, std_nsd) = unet_results_dict["Average Distance"]

    f = open(os.path.join(nnunet_res_folder, str(0), "results_dict.pkl"), "rb")
    nnunet_results_dict = pkl.load(f)
    f.close()

    (nn_av_dice, nn_std_dice) = nnunet_results_dict["Average Dice"]
    (nn_av_nsd, nn_std_nsd) = nnunet_results_dict["Average Distance"]

    f = open(os.path.join(proto_res_folder, str(2), "results_dict.pkl"), "rb")
    proto_results_dict = pkl.load(f)
    f.close()

    (proto_av_dice, _) = proto_results_dict["Average Dice"]
    (proto_av_nsd, _) = proto_results_dict["Average Distance"]

    for i in range(1, 13):
        print("{0} & {1:.2f} & {2:.2f} & {3:.2f} \\\\".format(organs_dict[i], nn_av_dice[i], av_dice[i], proto_av_dice[i]))

    print("\n")

    for i in range(1, 13):
        print("{0} & {1:.2f} & {2:.2f} & {3:.2f} \\\\".format(organs_dict[i], nn_av_nsd[i], av_nsd[i], proto_av_nsd[i]))


    return 0


if __name__ == "__main__":
    print_results()