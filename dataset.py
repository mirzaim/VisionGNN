import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageNetteDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.split = (split != 'train')
        self.root = root
        self.transform = transform
        self.images = pd.read_csv(
            root + '/noisy_imagenette.csv'
        )[['path', 'noisy_labels_0', 'is_valid']]
        self.images = self.images[self.images['is_valid'] == self.split]
        self.images['noisy_labels_0'] = pd.Categorical(
            self.images['noisy_labels_0']).codes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images.iloc[idx]
        img = Image.open(self.root + '/' + item['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, item['noisy_labels_0']
