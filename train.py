import model_helper
import data_helper
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image


def prepare_args():
    
    parser = argparse.ArgumentParser(description = 'Train model')
    parser.add_argument('--data_dir', action='store', default='flowers', help='Directory for images')
    # parser.add_argument('data_dir', nargs = '?', action = "store", default = "./flowers/")
    parser.add_argument('--save_dir', dest = 'save_dir', nargs = '?', action = 'store', default = './checkpoint.pth')
    parser.add_argument('--arch', dest = 'arch', nargs = '?', action = "store", default = 'vgg16')
    parser.add_argument('--learning_rate', dest = 'lr', nargs='?', action="store", type = int, default=0.001)
    parser.add_argument('--hidden_units', dest = 'hidden_units', nargs='?', action="store", type = int, default=500)
    parser.add_argument('--epochs', dest = 'epochs', nargs='?', action="store", type = int, default=5)
    parser.add_argument('--gpu', dest = 'gpu', nargs='?', action="store", default='GPU')
    
    return parser.parse_args()



def load_data(data_dir = 'flowers'):
    data_dir = prepare_args().data_dir
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




# train_loaders, vaild_loaders, test_loaders, class_to_idx = data_helper.load_data(data_dir)
parsed_args = prepare_args()
# data_dir = prepare_args().data_dir
model, criterion, optimizer = model_helper.setup_model(structure = parsed_args().arch, dropout = 0.5, lr=0.001, power = prepare_args().gpu, hidden_layer = prepare_args().hidden_units)
model_fuc.train_model(model, criterion, optimizer, train_loaders, vaild_loaders, power = gpu, epochs = epochs)
model_fuc.save_model(class_to_idx, save_dir, model, arch, optimizer)