import os
import inspect
from random import randint
from typing import List

import torch


class Dataset(torch.utils.data.Dataset):
    def getitem(self, idx):
        pass

    def __len__(self):
        return

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as e:
            raise e

    def __repr__(self):
        return f'{self.__class__.__name__}'
