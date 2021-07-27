import argparse
import itertools
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from limbresml.utils import preprocess_single_file, setup_logger

logger = setup_logger("generate_dataset")


def get_split_ids(n, method="random_balanced", labels=[]):
    if method == "all_test":
        logger.warning("use all data for test")
        return [], [], list(range(n))

    if method == "designed":
        # DEBUG PURPOSE
        val = set([0, 12, 20, 34, 41, 46])
        test = set([3, 5, 9, 33, 37, 45])
        train = set(range(n)) - val - test
    elif method == "designed2":
        # DEBUG PURPOSE
        val = set([11, 36, 41, 43, 10, 30])
        test = set([27, 29, 7, 45, 2, 31])
        train = set(range(n)) - val - test
    elif method == "random":
        n_val = int(n * 0.2)
        train = set(range(n))
        idx = list(np.random.choice(list(train), size=2 * n_val, replace=False))
        val, test = set(idx[:n_val]), set(idx[n_val:])
        train -= val + test
    elif method == "random_balanced":
        if len(labels) != n:
            logger.error(
                f"number of labels (got {len(labels)}) not matched n (got {n}) \
                for random balanced split method"
            )
            raise ValueError()

        counter = Counter(labels)
        n_val = int(counter.most_common()[-1][-1] * 0.2)  # find the least common class as reference
        val, test = [], []
        for label in list(counter):
            idx = [i for i, _label in enumerate(labels) if _label == label]
            idx = list(np.random.choice(idx, size=2 * n_val, replace=False))
            val += idx[:n_val]
            test += idx[n_val:]
        train = set(range(n)) - set(val) - set(test)
    else:
        logger.error(
            f"method must be one of 'random', 'random_balanced', or 'all_test' (got '{method}')"
        )
        raise ValueError()

    train, val, test = sorted(list(train)), sorted(list(val)), sorted(list(test))
    logger.info(f"number of files in train / val / test: {len(train)} / {len(val)} / {len(test)}")
    return train, val, test


def preprocess_files(files, labels=[], **kwargs):
    if len(files) == 0:
        return np.array([]), []

    if len(labels) > 0:
        assert len(files) == len(
            labels
        ), f"number of files and labels not mathced (got {len(files)} vs {len(labels)})"
    else:
        labels = [None] * len(files)

    xs = []
    ms = []
    for f, lbl in zip(files, labels):
        x, m = preprocess_single_file(f, lbl, **kwargs)
        xs.append(x)
        ms.append(m)
    xs = np.concatenate(xs, axis=0)
    ms = list(itertools.chain(*ms))

    return xs, ms


def isvalid_csv(file, min_len=300):
    file = Path(file)
    if not file.is_file():
        logger.warning(f"'{str(file)}' doesn't exist")
        return False

    df = pd.read_csv(file)
    for arm in ["LEFT", "RIGHT"]:
        n_data = sum(df["Limb"] == (arm + "_ARM"))
        if n_data < min_len:
            logger.warning(f"{file.name} has {n_data} {arm} but needs at least {min_len}")
            return False
    return True


def check_csv_files(files, labels=[], min_len=300, num_classes=3):
    logger.info(f"checkcing {len(files)} csv files...")
    no_labels = len(labels) == 0
    if no_labels:
        labels = [1] * len(files)

    valid_files = []
    valid_labels = []
    for f, lbl in zip(files, labels):
        if np.isnan(lbl):
            logger.warning(f"No label for '{f.name}' in annotation file. Skip this file. ")
            continue

        lbl = int(lbl)
        if lbl == 0:
            continue
        if lbl > num_classes or lbl < 0:
            logger.warning(
                f"No useful label for '{f.name}' in annotation file (got {lbl} but had {num_classes} classes). \
                Skip this file. "
            )
            continue

        if not isvalid_csv(f, min_len):
            logger.warning("skip this file")
            continue

        valid_files.append(f)
        valid_labels.append(lbl)

    if len(valid_files) == 0:
        logger.error("no valid csv files found ")
        raise AssertionError

    logger.info(f"found {len(valid_files)} valid csv files out of {len(files)} ")
    c = Counter(valid_labels)
    s = " / ".join(["{}"] * len(c))
    s += ": " + s
    s = f"number of files in class {s}"
    logger.info(s.format(*c.keys(), *c.values()))

    if no_labels:
        valid_labels = []

    return valid_files, valid_labels


def generate_dataset(
    data_dir,
    anno_file="",
    save_path="",
    *,
    split="random_balanced",
    n_samples_train=10,
    n_samples_test="center",
    switch_prob_train=0.5,
    switch_prob_test=-1,
    **kwargs,
):
    data_dir = Path(data_dir)
    if anno_file:
        anno = pd.read_csv(anno_file)
        files = anno["Filename"].tolist()
        files = [data_dir.joinpath(f"{f}.csv") for f in files]
        labels = anno["Label"].tolist()
        del anno
    else:
        files = list(data_dir.glob("*.csv"))
        labels = []

    files, labels = check_csv_files(files, labels)

    dataset = OrderedDict()
    train_ids, val_ids, test_ids = get_split_ids(len(files), method=split, labels=labels)
    for ids, dset in zip([train_ids, val_ids, test_ids], ["train", "val", "test"]):
        _files = [files[i] for i in ids]
        _labels = [labels[i] for i in ids] if len(labels) > 0 else []

        n_samples = n_samples_train if dset == "train" else n_samples_test
        switch_prob = switch_prob_train if dset == "train" else switch_prob_test
        xs, ms = preprocess_files(
            _files, _labels, n_samples=n_samples, switch_prob=switch_prob, **kwargs
        )
        ys = np.array([m["label"] for m in ms])
        for m in ms:
            m["dataset"] = dset

        ids = list(range(len(xs)))
        np.random.shuffle(ids)
        xs, ys = xs[ids], ys[ids]
        ms = [ms[i] for i in ids]

        dataset[f"X_{dset}"] = xs
        dataset[f"y_{dset}"] = ys
        dataset[f"meta_{dset}"] = ms

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True)
        np.savez(save_path, **dataset)
    return dataset


def demo(file, **kwargs):
    if not isvalid_csv(file):
        logger.error("file for demo is not valid")
        raise ValueError

    import matplotlib.pyplot as plt

    df = pd.read_csv(file)
    left, t_left = (
        df.Value[df.Limb == "LEFT_ARM"].to_numpy(),
        df.Time[df.Limb == "LEFT_ARM"].to_numpy(),
    )
    right, t_right = (
        df.Value[df.Limb == "RIGHT_ARM"].to_numpy(),
        df.Time[df.Limb == "RIGHT_ARM"].to_numpy(),
    )
    t_left = (t_left - t_left[0]) / 1e9
    t_right = (t_right - t_right[0]) / 1e9
    sample, meta = preprocess_single_file(file, **kwargs)
    sample, meta = sample[0], meta[0]
    drop, post_start = meta["drop"], meta["post_start"]
    pre_length, post_length = meta["pre_length"], meta["post_length"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    for t, v, lbl in zip([t_left, t_right], [left, right], ["left", "right"]):
        p1 = ax1.plot(t, v, label=lbl, lw=1)
        ax1.plot(
            t[drop:][post_start : post_start + post_length],
            v[drop:][post_start : post_start + post_length],
            lw=3,
            color=p1[-1].get_color(),
        )
        p2 = ax1.plot(t[:drop], v[:drop], color="k", lw=3)  # head drop
        ax1.plot(t[drop + pre_length :], v[drop + pre_length :], color="k", lw=3)  # tail drop
    p2[-1].set_label("drop")
    ax1.set_title("original")
    ax1.set_xlabel("time")
    ax1.set_ylabel("value")
    ax1.legend()

    left, right = sample[:post_length], sample[post_length:]
    t_left = t_left[drop:][post_start : post_start + post_length]
    t_right = t_right[drop:][post_start : post_start + post_length]
    ax2.plot(t_left, left, label="left")
    ax2.plot(t_right, right, label="right")

    title = "sampled "
    title = title + "(switched)" if meta["switch"] else title + "(not switched)"
    ax2.set_title(title)
    ax2.set_xlabel("time")
    ax2.set_ylabel("processed value")
    ax2.legend()

    fig.tight_layout()
    plt.show()
    return fig, (ax1, ax2)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data and prepare datasets. ")
    parser.add_argument(
        "--data-dir",
        metavar="DIR",
        default="rawdata/files",
        type=Path,
        help="directory of data files (default: 'rawdata/files')",
    )
    parser.add_argument(
        "--anno-file",
        metavar="PATH",
        default="",
        type=str,
        help="path of annotation file (default: '')",
    )
    parser.add_argument(
        "--save-path",
        metavar="PATH",
        default="data/dataset.npz",
        type=Path,
        help="path (including file name) to save datesets in npz (default: 'data/dataset.npz')",
    )
    parser.add_argument(
        "--split",
        metavar="METHOD",
        default="random_balanced",
        type=str,
        choices=["random", "random_balanced", "all_test"],
        help="the way to split train, validation, and test datesets. \
            choices: 'random', 'random_balanced' (default, random but makes class balanced), \
            'all_test' (use all data for test, no train and validation)",
    )
    parser.add_argument(
        "--head-drop",
        metavar="N",
        default=150,
        type=int,
        help="number of points to drop from head for each data file to \
            prevent noise at the beginning (default: 150)",
    )
    parser.add_argument(
        "--n-samples-train",
        metavar="N",
        default="10",
        type=str,
        help="for training dataset, use 'center' to generate one center cropped sample or \
            specify number of samples randomly generated from each date file after head dropping \
            (default: 10)",
    )
    parser.add_argument(
        "--n-samples-test",
        metavar="N",
        default="center",
        type=str,
        help="for validation or test dataset, use 'center' to generate one center cropped sample or \
            specify number of samples randomly generated from each date file after head dropping \
            (default: 'center')",
    )
    parser.add_argument(
        "--len-sample",
        metavar="N",
        default=300,
        type=int,
        help="number of data points for each sample (default: 300)",
    )
    parser.add_argument(
        "--preprocess",
        metavar="METHOD",
        default="normalized",
        type=str,
        choices=["normalized", "first_order", "second_order"],
        help="preprocessing method to use. choices: 'normalized' (default), \
            'first_order', 'second_order'",
    )
    parser.add_argument(
        "--switch-prob-train",
        metavar="FLOAT",
        default=0.5,
        type=float,
        help="probablity to switch left and right for train. set -1 to turn off. (default: 0.5)",
    )
    parser.add_argument(
        "--switch-prob-test",
        metavar="FLOAT",
        default=-1,
        type=float,
        help="probablity to switch left and right for test. set -1 to turn off, (default: -1)",
    )
    parser.add_argument(
        "--demo",
        metavar="PATH",
        default="",
        type=str,
        help="a csv file to demo (default: '')",
    )
    parser.add_argument(
        "--seed",
        metavar="N",
        default=0,
        type=int,
        help="seed for random to re-produce the same result (default: 0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"command line arguments: {str(args)}")
    if args.n_samples_train.isdigit():
        args.n_samples_train = int(args.n_samples_train)

    if args.n_samples_test.isdigit():
        args.n_samples_test = int(args.n_samples_test)

    if args.demo:
        csv = Path(args.demo)
        demo(
            csv,
            head_drop=args.head_drop,
            n_samples=args.n_samples_test,
            len_sample=args.len_sample,
            preprocess=args.preprocess,
            switch_prob=args.switch_prob_test,
        )
        exit(0)

    np.random.seed(args.seed)
    logger.info("start generating dataset...")
    dataset = generate_dataset(
        args.data_dir,
        args.anno_file,
        args.save_path,
        split=args.split,
        head_drop=args.head_drop,
        n_samples_train=args.n_samples_train,
        n_samples_test=args.n_samples_test,
        len_sample=args.len_sample,
        preprocess=args.preprocess,
        switch_prob_train=args.switch_prob_train,
        switch_prob_test=args.switch_prob_test,
    )
    logger.info(f"dataset saved to '{str(args.save_path)}'")
    logger.info("dateset statistics: ")
    for dset in ["train", "val", "test"]:
        ys = dataset[f"y_{dset}"]
        if (len(ys) > 0) and (ys[0] is None):
            logger.info(f"{dset}: {len(ys)} data without labales")
        else:
            n_samples = [(ys == lbl).sum() for lbl in range(1, 4)]
            logger.info("{}: {:d} / {:d} / {:d} for label 1 / 2 / 3".format(dset, *n_samples))
