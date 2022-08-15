# File to contain plotting scripts
import numpy as np
import os
import matplotlib.pyplot as plt


# Plot segmentations overlaying image
def PlotOverlay2D(img, pred, gt, max, alpha=0.5, c='gray', save=False, path=""):

    # collapse channels in floating
    num_channels = pred.shape[0]
    pred_2d = np.argmax(pred, axis=0)
    gt_2d = np.argmax(gt, axis=0)

    #for i in range(num_channels):
    #    pred_2d[pred[i, :, :]==1] = i
    #    gt_2d[gt[i, :, :] == 1] = i

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img[0, :, :], cmap='gray')
    plt.imshow(pred_2d, cmap=c, alpha=alpha, vmin=0, vmax=max)
    plt.title("Predictions")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img[0, :, :], cmap='gray')
    plt.imshow(gt_2d, cmap=c, alpha=alpha, vmin=0, vmax=max)
    plt.title("Ground Truth")
    plt.axis('off')
    plt.colorbar()
    plt.axis('equal')

    if save:
        plt.savefig(os.path.join(path, "overlay.png"))
    else:
        plt.show()


def main():
    import matplotlib.path as mpath
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection

    cmap = plt.cm.get_cmap('jet')
    num = 4
    max = 14
    rgba = cmap(num / max)

    print(rgba)

    plt.scatter([0, 1], [0, 1], color=rgba)
    plt.show()

if __name__ == "__main__":
    main()