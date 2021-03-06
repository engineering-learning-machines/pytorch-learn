import os
import sys
import logging
import time
import copy
import argparse
import numpy as np
#
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn

# Project imports
from .data import PascalDataset
from .vis import show_grid, plot_anchor_grid
from .transforms import Rescale, ToTensor
from .net import SSDHead

# ----------------------------------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------------------------------
DATASET_BASE_DIR = '/Users/g6714/Data/pascal'
DATASET_ANNOTATION_JSON = os.path.join(DATASET_BASE_DIR, 'PASCAL_VOC/pascal_train2007.json')
IMAGE_ROOT_DIR = os.path.join(DATASET_BASE_DIR, 'train/VOC2007/JPEGImages')
EPOCH_COUNT = 25
WORKER_COUNT = 1

log = logging.getLogger('detection')
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


def hw2corners(ctr, hw):
    return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)


def create_anchor(anc_grid=4, k=1):
    """
    Square anchor grid
    :param anchor_grid_size:
    :param k:
    :return:
    """
    anc_offset = 1/(anc_grid*2)
    anc_x = np.repeat(np.linspace(anc_offset, 1-anc_offset, anc_grid), anc_grid)
    anc_y = np.tile(np.linspace(anc_offset, 1-anc_offset, anc_grid), anc_grid)

    anc_ctrs = np.tile(np.stack([anc_x,anc_y], axis=1), (k,1))
    anc_sizes = np.array([[1/anc_grid,1/anc_grid] for i in range(anc_grid*anc_grid)])
    #
    anchors = torch.from_numpy(np.concatenate([anc_ctrs, anc_sizes], axis=1)).float()
    anchors.requires_grad = False
    grid_sizes = torch.from_numpy(np.array([1/anc_grid])).unsqueeze(1)
    grid_sizes.requires_grad = False

    anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])

# ----------------------------------------------------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------------------------------------------------


def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())


def num_features(m):
    c = children(m)
    if len(c) == 0:
        return None
    for l in reversed(c):
        if hasattr(l, 'num_features'):
            return l.num_features
        res = num_features(l)
        if res is not None:
            return res


def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data.cpu()]


def main(imgdir, epochs, workers, visualize=False):
    # Make sure that the libraries are set up correctly
    preflight_check()

    # Configure the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f'Device: {device}')

    # Data loading

    pascal_dataset = PascalDataset(DATASET_ANNOTATION_JSON, IMAGE_ROOT_DIR)

    # show_grid needs 12 items in the sample
    sample = [pascal_dataset[i] for i in range(12)]
    # show_grid(sample, file_name='original.png')

    # show some transformed images
    composed = transforms.Compose([Rescale(256),
                                   ])

    rescaled_sample = [composed(s) for s in sample]
    # show_grid(rescaled_sample, file_name='rescaled.png')

    # The transformation ToTensor reshapes the image to be used by PyTorch
    # TODO: Normalize(), Rotate(), Lighting(), Flip()
    transformations = transforms.Compose([Rescale(224), ToTensor()])
    transformed_dataset = PascalDataset(DATASET_ANNOTATION_JSON, IMAGE_ROOT_DIR, transform=transformations)
    dataloader = DataLoader(transformed_dataset, batch_size=64, shuffle=True, num_workers=4)

    # for batch_id, batch in enumerate(dataloader):
    #    print(batch_id, batch['image'].size(), batch['scene'])

    create_anchor(4, 1)

    # We add an additional class for the background
    num_classes = len(pascal_dataset.categories) + 1

    # WTF is k?
    k = 1
    n_act = k * (4 + num_classes)

    log.info(f'Number of categories: {num_classes}')

    # log.info(f'Number of data loader workers: {workers}')
    # Load a pretrained model and configure all hyper parameters
    log.info('Load a pretrained resnet34 model...')
    resnet_model = models.resnet34(pretrained=True)
    # model = models.resnet50(pretrained=True)

    # Isolate the last 8 layers of resnet
    resnet_layers = list(resnet_model.children())[:8]
    ssd_layers = [SSDHead(k, -3.0, num_classes)]

    # We need to convert to a model before we concatenate to resnet
    ssd_head_model = nn.Sequential(*ssd_layers).to(device)
    model = nn.Sequential(*(resnet_layers+ssd_layers)).to(device)




    # TODO: Define the loss function
    # log.info(f'Define a cross-entropy loss function')
    # criterion = nn.CrossEntropyLoss()

    # Optimizer
    # log.info(f'Define a stochastic gradient descent optimizer with momentum')
    # learning_rate = 0.001
    # momentum = 0.9
    # log.info(f'Learning rate: {learning_rate}')
    # log.info(f'Momentum: {momentum}')
    # # Note: Only the last layer parameters are getting optimized
    # optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=learning_rate, momentum=momentum)


    # ??? learn.opt_fn = optim.Adam





    # Augmentation and Normalization
    # aug_norm = aug_norm_transforms()
    #
    # # Load the data
    # log.info(f'Basic image dir: {imgdir}')
    # image_datasets = {x: datasets.ImageFolder(os.path.join(imgdir, x), aug_norm[x]) for x in ['train', 'valid']}
    #
    # # Basic dataset info
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    # class_names = image_datasets['train'].classes
    # log.info(f"train dataset size: {dataset_sizes['train']}")
    # log.info(f"validation dataset size: {dataset_sizes['valid']}")
    # log.info(f"class names: {class_names}")
    #
    # # Define the data loaders
    # log.info(f'Number of data loader workers: {workers}')
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=workers) for x in ['train', 'valid']}
    #
    # # Visualize some data
    # if visualize:
    #     visualize_training_data(dataloaders, class_names)
    #
    # # Load a pretrained model and configure all hyper parameters
    # log.info('Load a pretrained resnet34 model...')
    # model_conv = models.resnet34(pretrained=True)
    #
    # log.info('Freeze all existing layers')
    # for param in model_conv.parameters():
    #     param.requires_grad = False
    #
    # num_ftrs = model_conv.fc.in_features
    # log.info(f'Final fully connected layer output: {num_ftrs}')
    # log.info(f'Append a customized final linear layer. Its parameters are not frozen by default.')
    # model_output_features = 2
    # log.info(f'Model output features: {model_output_features}')
    # # Connect
    # model_conv.fc = nn.Linear(num_ftrs, model_output_features)
    # log.info('Preload the model on GPU')
    # model_conv = model_conv.to(device)
    # log.info(f'Define a cross-entropy loss function')
    # criterion = nn.CrossEntropyLoss()
    #
    # # Optimizer
    # log.info(f'Define a stochastic gradient descent optimizer with momentum')
    # learning_rate = 0.001
    # momentum = 0.9
    # log.info(f'Learning rate: {learning_rate}')
    # log.info(f'Momentum: {momentum}')
    # # Note: Only the last layer parameters are getting optimized
    # optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=learning_rate, momentum=momentum)
    # # Learning rate scheduler
    # lr_decay_rate = 0.1
    # lr_decay_interval = 7
    # log.info('Setting up a learning rate scheduler')
    # log.info(f'Learning rate decay rate: {lr_decay_rate}')
    # log.info(f'Decay learning rate every {lr_decay_interval} epochs')
    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=lr_decay_interval, gamma=lr_decay_rate)
    #
    # # Train the model
    # train_model(device, dataloaders, dataset_sizes, model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a network with transfer learning')
    parser.add_argument('--data_dir', dest='imgdir', default=DATASET_BASE_DIR, help='Location of image training data')
    parser.add_argument('--epochs', dest='epochs', type=int, default=EPOCH_COUNT, help='Number of epochs')
    parser.add_argument('--workers', dest='workers', type=int, default=WORKER_COUNT, help='Number of workers')
    args = parser.parse_args()

    main(args.imgdir, args.epochs, args.workers, visualize=False)
