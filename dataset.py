import torchvision.datasets as datasets
import torchvision.transforms as transforms

def _load_cifar10(dataroot):
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.ToTensor(),
         transforms.Normalize(normMean, normStd)
    ])
    transform_test = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(normMean, normStd)
    ])
    train_set = datasets.CIFAR10(dataroot, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(dataroot, train=False, download=True, transform=transform_test)
    return train_set, test_set

def load_dataset(dataset, dataroot):
    if dataset == 'CIFAR10':
        return _load_cifar10(dataroot)
    else:
        raise Exception('Invalid dataset!')
