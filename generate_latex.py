# a script  to generate latex snippets to include in the report
import pickle as pkl
import os


unet_res_folder = "/Users/katecevora/Documents/PhD/data/btcv/Images/UNet/Test"

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


def print_results(fold=0):
    f = open(os.path.join(unet_res_folder, str(fold), "results_dict.pkl"), "rb")
    unet_results_dict = pkl.load(f)
    f.close()

    (av_dice, std_dice) = unet_results_dict["Average Dice"]
    (av_nsd, std_nsd) = unet_results_dict["Average Distance"]

    print(r"\hline")
    print(r" & \multicolumn{2}{c}{nnUNet} & \multicolumn{2}{c}{UNet} & \multicolumn{2}{c}{Proto} \\")
    print(r"\hline")
    print(r" & DSC & NSD & DSC & NSD & DSC & NSD \\")
    print(r"\hline")
    for i in range(1, 13):
        print("{} & & & & & & \\\\".format(organs_dict[i]))
    return 0


if __name__ == "__main__":
    print_results()