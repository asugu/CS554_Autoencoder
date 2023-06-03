import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, random_split
import random


class MNISTDataset(Dataset):
    def __init__(self, root, transform=transforms.Compose([transforms.ToTensor(),])):
        self.transform = transform
        self.mode = 'train'
        self.train = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
        self.test = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)

        self.train = torch.utils.data.Subset(self.train, list(range(9000)))
        self.test = torch.utils.data.Subset(self.test, list(range(1000)))

        self.train, self.val = random_split(self.train, [8000, 1000])

    def __getitem__(self, index):
        if self.mode == "train":
            image, label = self.train[index]
        elif self.mode == "val":
            image, label = self.val[index]
        elif self.mode == "test":
            image, label = self.test[index]
        else:
            raise ValueError("Invalid mode. Expected 'train', 'val', or 'test'.")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)


class CIFAR10Dataset(Dataset):
    def __init__(self, root, transform=transforms.Compose([transforms.ToTensor(),])):
        self.mode = 'train'
        self.train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        self.test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

        self.train = torch.utils.data.Subset(self.train, list(range(9000)))
        self.test = torch.utils.data.Subset(self.test, list(range(1000)))

        self.train, self.val = random_split(self.train, [8000, 1000])

    def __getitem__(self, index):
        if self.mode == "train":
            image, label = self.train[index]
        elif self.mode == "val":
            image, label = self.val[index]
        elif self.mode == "test":
            image, label = self.test[index]
        else:
            raise ValueError("Invalid mode. Expected 'train', 'val', or 'test'.")

      

        return image, label

    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)
    
class CINIC10Dataset(Dataset):
    def __init__(self, root):
        self.mode = "train"
        transform = transforms.Compose([transforms.ToTensor(),])
        root = Path(root)
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]    
        
        # transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cinic_mean,std=cinic_std)])
        train_data = torchvision.datasets.ImageFolder(root / 'train', transform=transform)
        val_data = torchvision.datasets.ImageFolder(root / 'valid', transform=transform)
        test_data = torchvision.datasets.ImageFolder(root / 'test', transform=transform)

        # Randomly select subsets from the datasets
        train_indices = random.sample(range(len(train_data)), 8000)
        val_indices = random.sample(range(len(val_data)), 1000)
        test_indices = random.sample(range(len(test_data)), 1000)

        self.train = torch.utils.data.Subset(train_data, train_indices)
        self.val = torch.utils.data.Subset(val_data, val_indices)
        self.test = torch.utils.data.Subset(test_data, test_indices)


    def set_mode(self, mode):
        assert mode in ["train", "val", "test"]
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == "train":
            image, label = self.train[index]
        elif self.mode == "val":
            image, label = self.val[index]
        elif self.mode == "test":
            image, label = self.test[index]
        else:
            raise ValueError("Invalid mode. Expected 'train', 'val', or 'test'.")

        return image, label

    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)


if __name__ == "__main__":
    mnist = MNISTDataset(root="./mnist", transform=transforms.ToTensor())
    cifar10 = CIFAR10Dataset(root="./cifar10", transform=transforms.ToTensor())
    cinic10 = CINIC10Dataset(root="./cinic10")
    print(len(mnist), len(mnist.train), len(mnist.val), len(mnist.test))
    print(len(cifar10), len(cifar10.train), len(cifar10.val), len(cifar10.test))
    print(len(cinic10), len(cinic10.train), len(cinic10.val), len(cinic10.test))
    # Get an image and its corresponding label from the dataset
    image, label = cinic10[20]
    print(image.shape)

    # Visualize the image using matplotlib
    # plt.imsave("image.jpg",image)
