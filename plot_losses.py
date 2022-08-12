import matplotlib.pyplot as plt
import os
import pickle as pkl

def main():
    name = "prototype_v3_0_losses.pkl"

    f = open(os.path.join("/Users/katecevora/Documents/PhD/losses", name), "rb")
    [eps, av_train_error, av_train_dice, av_valid_error, av_valid_dice] = pkl.load(f)
    f.close()

    plt.plot(eps, av_train_error)
    plt.plot(eps, av_valid_error)
    plt.plot(eps, av_train_dice)
    plt.plot(eps, av_valid_dice)
    plt.show()


if __name__ == "__main__":
    main()