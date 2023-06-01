import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, random_split


class MNISTDataset(Dataset):
    def __init__(self, root, transform=transforms.Compose([transforms.ToTensor(),])):
        self.transform = transform

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
        self.transform = transform

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

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)


if __name__ == "__main__":
    mnist = MNISTDataset(root="./mnist", transform=transforms.ToTensor())
    cifar10 = CIFAR10Dataset(root="./cifar10", transform=transforms.ToTensor())
    print(len(mnist), len(mnist.train), len(mnist.val), len(mnist.test))
    print(len(cifar10), len(cifar10.train), len(cifar10.val), len(cifar10.test))
    # Get an image and its corresponding label from the dataset
    # image, label = mnist[0]

    # print(image.shape)

    # Visualize the image using matplotlib
    # plt.imsave("image.jpg",image)
