import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image


def load_data(data_dir = 'flowers'):
#     data_dir = 
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    bs = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = bs, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = bs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = bs)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    transform = transforms.Compose(
        [transforms.Resize(255),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(
             [0.485, 0.456, 0.406], 
             [0.229, 0.224, 0.225])])
    
    return transform(pil_image)



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax