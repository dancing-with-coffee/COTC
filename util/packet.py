import random
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

###

class TemplateData(object):
    def __init__(self, config, logger):
        super().__init__()

        self.config = config
        self.logger = logger
        self.device = torch.device("cuda") if self.config.cuda else torch.device("cpu")

    def load_datasets(self):
        raise NotImplementedError

    def get_loaders(self):
        raise NotImplementedError

###

class TemplateModel(nn.Module):
    def __init__(self, config, logger):
        super().__init__()

        self.config = config
        self.logger = logger
        self.device = torch.device("cuda") if self.config.cuda else torch.device("cpu")

    def define(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

###

class TemplateRunner(object):
    def __init__(self, config, logger, data, model):
        super().__init__()

        self.config = config
        self.logger = logger
        self.data = data
        self.model = model
        self.device = torch.device("cuda") if self.config.cuda else torch.device("cpu")

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

###
