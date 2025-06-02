from util.packet import *

###

from transformers import AutoTokenizer
from util.helper import knn
from util.proc import read_texts

###

class Data(TemplateData):
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def load_datasets(self):
        self.labels, self.texts, self.texts_0, self.texts_1, self.tfidfs = read_texts(self.config.data_name, max_features=self.config.max_features)
        self.texts_n = self.find_neighbors()
        self.tfidfs_n = self.texts_n
        self.pseudo_labels = None
        self.confident_masks = None

        self.pre_dataset = PreDataset(self.tfidfs, self.tfidfs_n)
        self.train_dataset = TrainDataset(self.texts, self.texts_0, self.texts_1, self.tfidfs, self.texts_n, self.tfidfs_n, max_length=self.config.max_length)
        self.test_dataset = TestDataset(self.texts, self.tfidfs, max_length=self.config.max_length)

        return self.pre_dataset, self.train_dataset, self.test_dataset

    def get_loaders(self):
        pre_loader = DataLoader(self.pre_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers, drop_last=True)
        train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers, drop_last=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, drop_last=False)

        return pre_loader, train_loader, test_loader

    def find_neighbors(self):
        tfidfs = torch.from_numpy(self.tfidfs).to(self.device).float()
        indexes = knn(tfidfs, k=self.config.num_neighbors)

        return indexes.cpu().numpy().tolist()

###

class PreDataset(Dataset):
    def __init__(self, tfidfs, tfidfs_n):
        super().__init__()

        self.tfidfs = tfidfs
        self.tfidfs_n = tfidfs_n

    def __getitem__(self, index):
        tfidf = self.tfidfs[index]
        tfidf_n = self.tfidfs[random.choice(self.tfidfs_n[index])]

        return tfidf, tfidf_n

    def __len__(self):
        return len(self.tfidfs)

###

class TrainDataset(Dataset):
    def __init__(self, texts, texts_0, texts_1, tfidfs, texts_n, tfidfs_n, max_length):
        super().__init__()

        self.texts = texts
        self.texts_0 = texts_0
        self.texts_1 = texts_1
        self.tfidfs = tfidfs
        self.texts_n = texts_n
        self.tfidfs_n = tfidfs_n
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")

    def __getitem__(self, index):
        text = self.tokenizer(self.texts[index], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        text_0 = self.tokenizer(self.texts_0[index], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        text_1 = self.tokenizer(self.texts_1[index], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        tfidf = self.tfidfs[index]
        text_n = self.tokenizer(self.texts[random.choice(self.texts_n[index])], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        tfidf_n = self.tfidfs[random.choice(self.tfidfs_n[index])]

        return text, text_0, text_1, tfidf, text_n, tfidf_n, index

    def __len__(self):
        return len(self.texts)

###

class TestDataset(Dataset):
    def __init__(self, texts, tfidfs, max_length):
        super().__init__()

        self.texts = texts
        self.tfidfs = tfidfs
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")

    def __getitem__(self, index):
        text = self.tokenizer(self.texts[index], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        tfidf = self.tfidfs[index]

        return text, tfidf

    def __len__(self):
        return len(self.texts)

###
