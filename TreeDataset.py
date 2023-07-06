import torch as t
from torch.utils.data.dataset import Dataset


class TreeDataset(Dataset):
    def __init__(self, data, labels, transforms=None):
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]

        if self.transforms:
            image = self.transforms(image)

        return (image, label)
