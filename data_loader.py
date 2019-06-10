from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random

class UnalignedDataset(data.Dataset):

    def __init__(self, dataset, image_dir, transform, mode):
        self.transform = transform
        self.mode = mode
        self.dir_A = os.path.join(image_dir + dataset, mode + 'A') if dataset != 'CelebA' else os.path.join(image_dir, mode + 'A')
        self.dir_B = os.path.join(image_dir + dataset, mode + 'B') if dataset != 'CelebA' else os.path.join(image_dir, mode + 'B')
        self.A_paths = self.make_dataset(self.dir_A)
        self.B_paths = self.make_dataset(self.dir_B)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.train_dataset = []
        self.test_dataset = []

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def make_dataset(self, dir):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for fname in sorted(os.listdir(dir)):
            path = os.path.join(dir, fname)
            images.append(path)

        return images

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]

        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)

        return A, B

    def __len__(self):
        return max(self.A_size, self.B_size)


class TestDataset(data.Dataset):

    def __init__(self, dataset, image_dir, transform, mode):
        self.transform = transform
        self.mode = mode
        self.dir_A = os.path.join(image_dir + dataset, mode + 'A') if dataset != 'CelebA' else os.path.join(image_dir, mode + 'A')
        self.dir_B = os.path.join(image_dir + dataset, mode + 'B') if dataset != 'CelebA' else os.path.join(image_dir, mode + 'B')
        self.A_paths = self.make_dataset(self.dir_A)
        self.B_paths = self.make_dataset(self.dir_B)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.train_dataset = []
        self.test_dataset = []

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def make_dataset(self, dir):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for fname in sorted(os.listdir(dir)):
            path = os.path.join(dir, fname)
            images.append(path)

        return images

    def __getitem__(self, index):
        # A_path = self.A_paths[index % self.A_size]

        # index_B = random.randint(0, self.B_size - 1)
        # B_path = self.B_paths[index_B]

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)

        return A, B

    def __len__(self):
        return max(self.A_size, self.B_size)


def get_loader(image_dir, crop_size=216, resize=216, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())

    transform.append(T.Resize(resize, Image.BICUBIC))
    transform.append(T.RandomCrop(crop_size))

    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    transform = T.Compose(transform)
    
    dataset = UnalignedDataset(dataset, image_dir, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader

