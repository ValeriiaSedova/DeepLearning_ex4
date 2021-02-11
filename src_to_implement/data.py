from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        if mode == 'train':
            self._transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std),
            # tv.transforms.RandomHorizontalFlip(),
            # tv.transforms.RandomVerticalFlip(),
            # tv.transforms.
        ])
        if mode == 'val':
             self._transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        fname = sample.filename
        label = sample.crack, sample.inactive
        img = gray2rgb(imread(fname))

        if self._transform:
            img = self._transform(img)

        return torch.tensor(img), torch.tensor(label, dtype=torch.float)

        
