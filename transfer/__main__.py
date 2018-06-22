import os
import sys
import logging
import time
import copy
import argparse
#
import numpy as np
#
# from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
#
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------------------------------
IMAGE_BASE_DIR = '/Users/g6714/Data/fastai/dogscats'
EPOCH_COUNT = 25
WORKER_COUNT = 1

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


def train_model(device, dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
# ----------------------------------------------------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------------------------------------------------


def main(imgdir, epochs, workers, visualize=False):
    # Make sure that the libraries are set up correctly
    preflight_check()

    # Configure the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f'Device: {device}')

    # Augmentation and Normalization
    aug_norm = aug_norm_transforms()

    # Load the data
    log.info(f'Basic image dir: {imgdir}')
    image_datasets = {x: datasets.ImageFolder(os.path.join(imgdir, x), aug_norm[x]) for x in ['train', 'valid']}

    # Basic dataset info
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes
    log.info(f"train dataset size: {dataset_sizes['train']}")
    log.info(f"validation dataset size: {dataset_sizes['valid']}")
    log.info(f"class names: {class_names}")

    # Define the data loaders
    log.info(f'Number of data loader workers: {workers}')
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=workers) for x in ['train', 'valid']}

    # Visualize some data
    if visualize:
        visualize_training_data(dataloaders, class_names)

    # Load a pretrained model and configure all hyper parameters
    log.info('Load a pretrained resnet34 model...')
    # model_ft = models.resnet18(pretrained=True)
    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    log.info(f'Final fully connected layer output: {num_ftrs}')
    log.info(f'Append a customized final linear layer')
    model_output_features = 2
    model_ft.fc = nn.Linear(num_ftrs, model_output_features)
    log.info(f'Model output features: {model_output_features}')
    model_ft = model_ft.to(device)
    log.info(f'Define a cross-entropy loss function')
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    log.info(f'Define a stochastic gradient descent optimizer with momentum')
    learning_rate = 0.001
    momentum = 0.9
    log.info(f'Learning rate: {learning_rate}')
    log.info(f'Momentum: {momentum}')
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=momentum)
    # Learning rate scheduler
    lr_decay_rate = 0.1
    lr_decay_interval = 7
    log.info('Setting up a learning rate scheduler')
    log.info(f'Learning rate decay rate: {lr_decay_rate}')
    log.info(f'Decay learning rate every {lr_decay_interval} epochs')
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=lr_decay_interval, gamma=lr_decay_rate)

    # Train the model
    train_model(device, dataloaders, dataset_sizes, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a network with transfer learning')
    parser.add_argument('--image_dir', dest='imgdir', default=IMAGE_BASE_DIR, help='Location of image training data')
    parser.add_argument('--epochs', dest='epochs', type=int, default=EPOCH_COUNT, help='Number of epochs')
    parser.add_argument('--workers', dest='workers', type=int, default=WORKER_COUNT, help='Number of workers')
    args = parser.parse_args()

    main(args.imgdir, args.epochs, args.workers, visualize=False)
