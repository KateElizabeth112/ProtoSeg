# Make the indices for 5-fold cross validation
import pickle as pkl
import os
import numpy as np

NUM_FOLDS = 5


def main():
    root_dir = '/Users/katecevora/Documents/PhD/data/btcv'
    data_dir = os.path.join(root_dir, 'nnUNet_raw_data_base/nnUNet_raw_data/Task500_BTCV')

    # load filenames
    f = open(os.path.join(root_dir, "filenames.pkl"), 'rb')
    filenames = pkl.load(f)
    f.close()

    # get training dataset length
    ds_len = len(filenames)
    fold_len = int(np.floor(ds_len / NUM_FOLDS))
    train_len = (NUM_FOLDS - 1) * fold_len

    # randomly assign indices to folds
    indices = np.arange(0, ds_len)
    folds = []
    for f in range(NUM_FOLDS):
        idx = np.random.choice(range(indices.shape[0]), fold_len, replace=False)
        folds.append(indices[idx].astype('int'))
        indices = np.delete(indices, idx)

    # combine folds into training and validation sets
    train_indices = []
    valid_indices = []
    for f in range(NUM_FOLDS):
        valid_indices.append(folds[f])
        train_idx = np.zeros(train_len)
        current_idx = 0
        for k in range(NUM_FOLDS):
            if (f != k):
                train_idx[current_idx:current_idx+fold_len] = folds[k]
                current_idx += fold_len
        train_indices.append(train_idx)

    # save the folds
    f = open("fold.pkl", "wb")
    pkl.dump([train_indices, valid_indices], f)
    f.close()




if __name__ == "__main__":
    main()