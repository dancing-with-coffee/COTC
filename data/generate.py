import sys

sys.dont_write_bytecode = True

###

import re
import math
import random
from tqdm import tqdm
import numpy as np
import torch
from nlpaug.augmenter.word import ContextualWordEmbsAug
import os

###


def clean(string):
    # regular operation
    string = re.sub(r"[^A-Za-z0-9\']", " ", string)
    string = re.sub(r"\'s", " #s", string)
    string = re.sub(r"\'re", " #re", string)
    string = re.sub(r"\'ll", " #ll", string)
    string = re.sub(r"\'d", " #d", string)
    string = re.sub(r"\'ve", " #ve", string)
    string = re.sub(r"n\'t", " n#t", string)
    string = re.sub(r"\'", " ", string)
    string = re.sub(r"#", "'", string)

    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


###


def load_20newsgroups():
    """Load 20newsgroups dataset from text and label files"""
    text_path = "dataset/20newsgroups/20newsgroups.txt"
    label_path = "dataset/20newsgroups/20newsgroups_labels.txt"

    # Check if files exist
    if not os.path.exists(text_path) or not os.path.exists(label_path):
        print(f"Error: {text_path} or {label_path} not found!")
        return [], []

    # Read texts
    with open(text_path, mode="r", encoding="utf-8") as stream:
        text_lines = stream.readlines()

    # Read labels
    with open(label_path, mode="r", encoding="utf-8") as stream:
        label_lines = stream.readlines()

    # Process data
    labels = []
    texts = []

    for label, text in zip(label_lines, text_lines):
        labels.append(int(label.strip()) - 1)  # Convert to 0-based indexing
        texts.append(clean(text.strip()))

    return labels, texts


def load_bbc():
    """Load BBC dataset from text and label files"""
    text_path = "dataset/bbc/bbc.txt"
    label_path = "dataset/bbc/bbc_labels.txt"

    # Check if files exist
    if not os.path.exists(text_path) or not os.path.exists(label_path):
        print(f"Error: {text_path} or {label_path} not found!")
        return [], []

    # Read texts
    with open(text_path, mode="r", encoding="utf-8") as stream:
        text_lines = stream.readlines()

    # Read labels
    with open(label_path, mode="r", encoding="utf-8") as stream:
        label_lines = stream.readlines()

    # Process data
    labels = []
    texts = []

    for label, text in zip(label_lines, text_lines):
        labels.append(int(label.strip()) - 1)  # Convert to 0-based indexing
        texts.append(clean(text.strip()))

    return labels, texts


def load_reuters8():
    """Load Reuters8 dataset from text and label files"""
    text_path = "dataset/reuters8/reuters8.txt"
    label_path = "dataset/reuters8/reuters8_labels.txt"

    # Check if files exist
    if not os.path.exists(text_path) or not os.path.exists(label_path):
        print(f"Error: {text_path} or {label_path} not found!")
        return [], []

    # Read texts
    with open(text_path, mode="r", encoding="utf-8") as stream:
        text_lines = stream.readlines()

    # Read labels
    with open(label_path, mode="r", encoding="utf-8") as stream:
        label_lines = stream.readlines()

    # Process data
    labels = []
    texts = []

    for label, text in zip(label_lines, text_lines):
        labels.append(int(label.strip()) - 1)  # Convert to 0-based indexing
        texts.append(clean(text.strip()))

    return labels, texts


def load_webkb():
    """Load WebKB dataset from text and label files"""
    text_path = "dataset/webkb/webkb.txt"
    label_path = "dataset/webkb/webkb_labels.txt"

    # Check if files exist
    if not os.path.exists(text_path) or not os.path.exists(label_path):
        print(f"Error: {text_path} or {label_path} not found!")
        return [], []

    # Read texts
    with open(text_path, mode="r", encoding="utf-8") as stream:
        text_lines = stream.readlines()

    # Read labels
    with open(label_path, mode="r", encoding="utf-8") as stream:
        label_lines = stream.readlines()

    # Process data
    labels = []
    texts = []

    for label, text in zip(label_lines, text_lines):
        labels.append(int(label.strip()) - 1)  # Convert to 0-based indexing
        texts.append(clean(text.strip()))

    return labels, texts


###

if __name__ == "__main__":
    # initialize
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    datasets = [
        "agnews",
        "biomedical",
        "googlenews-s",
        "googlenews-t",
        "googlenews-ts",
        "searchsnippets",
        "stackoverflow",
        "tweet",
        "20newsgroups",
        "bbc",
        "reuters8",
        "webkb",
    ]

    for dataset in datasets:
        print(f"Processing {dataset}...")

        if dataset == "agnews":
            # define path
            label_text_path = "dataset/" + dataset + "/agnewsdataraw-8000"
            csv_path = dataset + ".csv"

            # read from txt file
            with open(label_text_path, mode="r", encoding="utf-8") as stream:
                label_text_lines = stream.readlines()

            # get labels and texts
            labels = []
            texts = []

            for line in label_text_lines:
                line = line.split("\t")

                labels.append(int(line[0]) - 1)
                texts.append(clean("\t".join(line[1:])))

        elif dataset == "biomedical" or dataset == "stackoverflow":
            # define path
            if dataset == "biomedical":
                label_path = "dataset/" + dataset + "/Biomedical_gnd.txt"
                text_path = "dataset/" + dataset + "/Biomedical.txt"
            elif dataset == "stackoverflow":
                label_path = "dataset/" + dataset + "/StackOverflow_gnd.txt"
                text_path = "dataset/" + dataset + "/StackOverflow.txt"

            csv_path = dataset + ".csv"

            # read from txt file
            with open(label_path, mode="r", encoding="utf-8") as stream:
                label_lines = stream.readlines()

            with open(text_path, mode="r", encoding="utf-8") as stream:
                text_lines = stream.readlines()

            # get labels and texts
            labels = []
            texts = []

            for label, text in zip(label_lines, text_lines):
                labels.append(int(label) - 1)
                texts.append(clean(text))

        elif (
            dataset == "googlenews-s"
            or dataset == "googlenews-t"
            or dataset == "googlenews-ts"
            or dataset == "tweet"
        ):
            # define path
            if dataset == "googlenews-s":
                text_label_path = "dataset/" + dataset + "/S"
            elif dataset == "googlenews-t":
                text_label_path = "dataset/" + dataset + "/T"
            elif dataset == "googlenews-ts":
                text_label_path = "dataset/" + dataset + "/TS"
            elif dataset == "tweet":
                text_label_path = "dataset/" + dataset + "/Tweet"

            csv_path = dataset + ".csv"

            # read from txt file
            with open(text_label_path, mode="r", encoding="utf-8") as stream:
                text_label_lines = stream.readlines()

            # get labels and texts
            labels = []
            texts = []

            for line in text_label_lines:
                line = line.split('{"text": "')[1].split('", "cluster": ')

                labels.append(int(line[1].split("}")[0]) - 1)
                texts.append(clean(line[0]))

            miss = []

            for i in range(max(labels) + 1):
                if i not in labels:
                    miss.append(i)

            for i in range(len(labels)):
                labels[i] = labels[i] - len(
                    [None for label in miss if label < labels[i]]
                )

        elif dataset == "searchsnippets":
            # define path
            train_text_label_path = "dataset/" + dataset + "/train.txt"
            test_text_label_path = "dataset/" + dataset + "/test.txt"
            csv_path = dataset + ".csv"

            # read from txt file
            with open(train_text_label_path, mode="r", encoding="utf-8") as stream:
                train_text_label_lines = stream.readlines()

            with open(test_text_label_path, mode="r", encoding="utf-8") as stream:
                test_text_label_lines = stream.readlines()

            # get labels and texts
            labels = []
            texts = []

            for line in train_text_label_lines + test_text_label_lines:
                line = line.split(" ")

                labels.append(line[-1].strip().lower())
                texts.append(clean(" ".join(line[:-1])))

            mapper = {}
            counter = 0

            for i in range(len(labels)):
                if labels[i] not in mapper:
                    mapper[labels[i]] = counter
                    counter = counter + 1

                labels[i] = mapper[labels[i]]

        elif dataset == "20newsgroups":
            csv_path = dataset + ".csv"
            labels, texts = load_20newsgroups()

        elif dataset == "bbc":
            csv_path = dataset + ".csv"
            labels, texts = load_bbc()

        elif dataset == "reuters8":
            csv_path = dataset + ".csv"
            labels, texts = load_reuters8()

        elif dataset == "webkb":
            csv_path = dataset + ".csv"
            labels, texts = load_webkb()

        # Skip augmentation if texts are empty
        if not texts:
            print(f"No texts found for {dataset}, skipping...")
            continue

        # augment data
        bert_all_texts = []
        roberta_all_texts = []

        for percentage in [10, 20, 30]:
            bert = ContextualWordEmbsAug(
                "google-bert/bert-base-uncased",
                action="substitute",
                aug_min=1,
                aug_p=percentage / 100,
                device="cuda:0",
                batch_size=128,
            )
            roberta = ContextualWordEmbsAug(
                "FacebookAI/roberta-base",
                action="substitute",
                aug_min=1,
                aug_p=percentage / 100,
                device="cuda:1",
                batch_size=128,
            )

            bert_texts = []
            roberta_texts = []

            for i in tqdm(range(math.ceil(len(texts) / 128))):
                text = texts[128 * i : 128 * (i + 1)]

                bert_text = bert.augment(text)
                roberta_text = roberta.augment(text)

                for text_0, text_1 in zip(bert_text, roberta_text):
                    text_0 = clean(str(text_0))
                    text_0 = re.sub(r"unk", " ", text_0)
                    text_0 = re.sub(r"\s{2,}", " ", text_0)

                    text_1 = clean(str(text_1))
                    text_1 = re.sub(r"unk", " ", text_1)
                    text_1 = re.sub(r"\s{2,}", " ", text_1)

                    bert_texts.append(text_0)
                    roberta_texts.append(text_1)

            bert_all_texts.append(bert_texts)
            roberta_all_texts.append(roberta_texts)

        # write to csv file
        with open(csv_path, mode="w", encoding="utf-8") as stream:
            for (
                label,
                text,
                text_10_0,
                text_10_1,
                text_20_0,
                text_20_1,
                text_30_0,
                text_30_1,
            ) in zip(
                labels,
                texts,
                bert_all_texts[0],
                roberta_all_texts[0],
                bert_all_texts[1],
                roberta_all_texts[1],
                bert_all_texts[2],
                roberta_all_texts[2],
            ):
                stream.write(
                    str(label)
                    + "\t"
                    + text
                    + "\t"
                    + text_10_0
                    + "\t"
                    + text_10_1
                    + "\t"
                    + text_20_0
                    + "\t"
                    + text_20_1
                    + "\t"
                    + text_30_0
                    + "\t"
                    + text_30_1
                    + "\n"
                )

        # collect information
        num_classes = max(labels) + 1
        num_samples = len(texts)

        classes_sizes = [0 for i in range(num_classes)]
        total_sample_length = 0

        for i in range(num_samples):
            classes_sizes[labels[i]] = classes_sizes[labels[i]] + 1
            total_sample_length = total_sample_length + len(texts[i].split(" "))

        print("collect information")
        print("generate " + dataset + " successfully")
        print("number of classes: " + str(num_classes))
        print(
            "largest versus smallest class size: "
            + str(max(classes_sizes) / min(classes_sizes))
        )
        print("number of samples: " + str(num_samples))
        print("average sample length: " + str(total_sample_length / num_samples))
        print("###")

###
