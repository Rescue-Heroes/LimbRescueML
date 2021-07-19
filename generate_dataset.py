import itertools
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


def get_split_ids(n, method="designed", labels=None):
    if method == "designed":
        val = set([0, 12, 20, 34, 41, 46])
        test = set([3, 5, 9, 33, 37, 45])
        train = set(range(n)) - val - test
    elif method == "designed2":
        val = set([11, 36, 41, 43, 10, 30])
        test = set([27, 29, 7, 45, 2, 31])
        train = set(range(n)) - val - test
    elif method == "random":
        n_val = int(n * 0.2)
        train = set(range(n))
        val = set(np.random.choice(list(train), size=n_val, replace=False))
        train -= val
        test = set(np.random.choice(list(train), size=n_val, replace=False))
        train -= test
    elif method == "random_balanced":
        if labels is None:
            raise ValueError("data labels must be provided for random balaced split method")
        train = set(range(n))
        count = Counter(labels)
        n_val_class = int(count.most_common()[-1][-1] * 0.2)
        val = []
        test = []
        for classi in list(count):
            class_index = [i for i in range(len(labels)) if labels[i] == classi]
            choice = list(np.random.choice(class_index, size=2 * n_val_class, replace=False))
            val += choice[:n_val_class]
            test += choice[n_val_class:]
        print(test, val)

    else:
        raise ValueError(f"method must be one of 'designed' or 'random' (got '{method}')")
    return sorted(list(train)), sorted(list(val)), sorted(list(test))


def preprocess_single_file(
    file, head_drop=150, n_samples=10, len_sample=300, preprocess="normalized"
):
    df = pd.read_csv(file)
    left, t_left = (
        df.Value[df.Limb == "LEFT_ARM"].to_numpy()[head_drop:],
        df.Time[df.Limb == "LEFT_ARM"].to_numpy()[head_drop:],
    )
    right, t_right = (
        df.Value[df.Limb == "RIGHT_ARM"].to_numpy()[head_drop:],
        df.Time[df.Limb == "RIGHT_ARM"].to_numpy()[head_drop:],
    )
    del df

    # align left and right from head
    length = min(len(left), len(right))
    left, t_left = left[:length], t_left[:length]
    right, t_right = right[:length], t_right[:length]

    if preprocess == "normalized":
        left /= max(left)
        right /= max(right)

    elif preprocess == "first_order":
        left = np.gradient(left, t_left)
        right = np.gradient(right, t_right)

    elif preprocess == "second_order":
        for i in range(2):
            left = np.gradient(left, t_left)
            right = np.gradient(right, t_right)
    else:
        raise ValueError(
            f"method must be one of 'normalized', 'first_order' or 'second_order' (got '{preprocess}')"
        )

    samples = []
    metas = []
    for _ in range(n_samples):
        i = np.random.randint(len(left) - len_sample)
        samples.append(left[i : i + len_sample])
        samples.append(right[i : i + len_sample])
        metas.append(
            {
                "file": file.name,
                "max_length": length,
                "post_start": i,
                "post_end": i + len_sample,
            }
        )
        # start_end.append((length, i, i + len_sample))

    samples = np.concatenate(samples, axis=0).reshape(int(len(samples) / 2), len_sample * 2)
    return samples, metas


def preprocess_files(files, **kwargs):
    xs = []
    ms = []
    for f in files:
        x, m = preprocess_single_file(f, **kwargs)
        xs.append(x)
        ms.append(m)
    xs = np.concatenate(xs, axis=0)
    ms = list(itertools.chain(*ms))
    return xs, ms


def generate_dataset(anno_file, data_dir, save_path=None, *, split="designed", **kwargs):
    anno = pd.read_csv(anno_file)
    files = anno["Filename"].tolist()
    labels = anno["Label"].tolist()

    data_dir = Path(data_dir)
    files = [data_dir.joinpath(f"{f}.csv") for f in files]
    del anno

    dataset = OrderedDict()
    train_ids, val_ids, test_ids = get_split_ids(len(files), method=split, labels=labels)
    for ids, dset in zip([train_ids, val_ids, test_ids], ["train", "val", "test"]):
        _files = [files[i] for i in ids]
        _labels = [labels[i] for i in ids]

        xs, ms = preprocess_files(_files, **kwargs)
        ys = np.array([[lbl] * kwargs["n_samples"] for lbl in _labels]).flatten()
        # fs = np.array([[f.name] * kwargs["n_samples"] for f in _files]).flatten()

        ids = list(range(len(ys)))
        np.random.shuffle(ids)
        xs, ys = xs[ids], ys[ids]
        ms = [ms[i] for i in ids]

        dataset[f"X_{dset}"] = xs
        dataset[f"y_{dset}"] = ys
        dataset[f"meta_{dset}"] = ms
        # dataset[f"start_end_{dset}"] = ses

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True)
        np.savez(save_path, **dataset)
    return dataset


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess data and prepare dataset. ")
    parser.add_argument(
        "--anno-file",
        metavar="PATH",
        default="rawdata/annotations.csv",
        type=Path,
        help="the path of annotation file",
    )
    parser.add_argument(
        "--data-dir",
        metavar="DIR",
        default="rawdata/files",
        type=Path,
        help="the directory of data files",
    )
    parser.add_argument(
        "--save-path",
        metavar="PATH",
        default="data/dataset.npz",
        type=Path,
        help="the path (inclufing file name) to save train, validation, and test datesets in npz",
    )
    parser.add_argument(
        "--split",
        metavar="METHOD",
        default="designed",
        type=str,
        choices=["designed", "random", "designed2", "random_balanced"],
        help="the way to split train, validation, and test datesetsz",
    )
    parser.add_argument(
        "--head-drop",
        metavar="N",
        default=150,
        type=int,
        help="number of points to drop from head for each data file",
    )
    parser.add_argument(
        "--n-samples",
        metavar="N",
        default=10,
        type=int,
        help="number of samples generated from each date file",
    )
    parser.add_argument(
        "--len-sample",
        metavar="N",
        default=300,
        type=int,
        help="number of data points for each sample",
    )
    parser.add_argument(
        "--preprocess",
        metavar="METHOD",
        default="normalized",
        type=str,
        choices=["normalized", "first_order", "second_order"],
        help="preprocessing method to use",
    )
    parser.add_argument(
        "--seed",
        metavar="N",
        default=0,
        type=int,
        help="seed for random",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    print(args)

    np.random.seed(args.seed)
    dataset = generate_dataset(
        args.anno_file,
        args.data_dir,
        args.save_path,
        split=args.split,
        head_drop=args.head_drop,
        n_samples=args.n_samples,
        len_sample=args.len_sample,
        preprocess=args.preprocess,
    )

    print("Done. ")
    print("Dateset Statistics: ")
    for dset in ["train", "val", "test"]:
        n_samples = [(dataset[f"y_{dset}"] == lbl).sum() for lbl in range(1, 4)]
        print("\t{}: {:d} / {:d} / {:d} for label 1 / 2 / 3".format(dset, *n_samples))
