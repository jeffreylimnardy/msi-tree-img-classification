import torch as t
from torch.utils.data.dataset import Dataset
from torch.nn.functional import one_hot
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transformer


class TreeDataset(Dataset):
    def __init__(self, data, labels, resize=False, transforms=None):
        if resize:
            resizer = transformer.Resize((16, 16))
            self.data = t.stack(([resizer(img) for img in data]))
        else:
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
