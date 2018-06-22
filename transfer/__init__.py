import os
import sys
import logging
import time
import copy
#
import numpy as np
#
# from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
#
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------------------------------
IMAGE_BASE_DIR = '/Users/g6714/Data/fastai/dogscats'

log = logging.getLogger('transfer')
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def preflight_check():
    log.info(f'PyTorch version: {torch.__version__}')
    log.info(f'Cuda is available: {torch.cuda.is_available()}')


def aug_norm_transforms():
    """
    Define data augmentation and normalization transforms
    :return:
    """
    # Data augmentation and normalization for training
    # Just normalization for validation
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_training_data(dataloaders, class_names):
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs, nrow=4)
    # show
    imshow(out, title=[class_names[x] for x in classes])
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------------------------------------------------


def main(visualize=False):
    # Make sure that the libraries are set up correctly
    preflight_check()

    # Configure the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f'Device: {device}')

    # Augmentation and Normalization
    aug_norm = aug_norm_transforms()

    # Load the data
    image_datasets = {x: datasets.ImageFolder(os.path.join(IMAGE_BASE_DIR, x), aug_norm[x]) for x in ['train', 'valid']}

    # Basic dataset info
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes
    log.info(f"train dataset size: {dataset_sizes['train']}")
    log.info(f"validation dataset size: {dataset_sizes['valid']}")
    log.info(f"class names: {class_names}")

    # Define the data loaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4) for x in ['train', 'valid']}

    # Visualize some data
    if visualize:
        visualize_training_data(dataloaders, class_names)


if __name__ == '__main__':
    main(visualize=True)
