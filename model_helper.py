import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from data_helper import process_image
from data_helper import imshow
from collections import OrderedDict


def setup_model(structure = 'vgg16', dropout = 0.5, lr=0.001, power = 'GPU', hidden_layer = 512):
    
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    if structure == 'alexnet':
        model = models.alexnet(pretrained=True)
    if structure == 'densenet121':
        model = models.densenet121(pretrained=True)


    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[0].in_features

    input_size = model.classifier[0].in_features
    output_size = len(cat_to_name)

    learning_rate = lr

    classifier = nn.Sequential(
    OrderedDict([
        ('fc1', nn.Linear(input_size, 500)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(500, int(output_size * 1.5))),
        ('relu2', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.35)),
        ('fc3', nn.Linear(int(output_size * 1.5), output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
 

    model.classifier = classifier

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    return model, criterion, optimizer


def train_model(model, criterion, optimizer, train_loader, valid_loader, power = 'GPU', epochs=1):
    
    for ep in range(epochs):
        running_loss = 0
        model.train()

        for images, labels in train_loader:

            images = images.to(device) 
            labels = labels.to(device)

            optimizer.zero_grad()

            l_probabilities = model.forward(images)
            loss = criterion(l_probabilities, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        with torch.no_grad():
            valid_loss = 0
            accuracy = 0
            model.eval()

            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)

                l_probabilities = model(images)
                batch_loss = criterion(l_probabilities, labels)
                valid_loss += batch_loss.item()

                probabilities = torch.exp(l_probabilities)
                top_p, top_class = probabilities.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # Validation Loss and Accuracy
            training_loss = running_loss / len(train_loader)
            validation_loss = valid_loss / len(valid_loader)
            testing_accuracy = accuracy / len(valid_loader)

            print("Epoch at {}/{} ".format(ep + 1, epochs),
                  "\t\t with Training Loss of {:.3f} ".format(training_loss),
                  "\t and Test Loss of {:.3f} ".format(validation_loss),
                  "\t and Test Accuracy of {:.3f}".format(testing_accuracy * 100))


def test_model(model, train_loader, criterion, power = 'GPU'):

    accuracy = 0
    n_correct = 0
    model = model.to(device)
    model.eval()

    with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                l_probabilities = model.forward(images)
                batch_loss = criterion(l_probabilities, labels)
                valid_loss += batch_loss.item()
                
                probabilities = torch.exp(l_probabilities)
                top_p, top_class = probabilities.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    test_accuracy = accuracy / len(test_loader)
    print("Test Accuracy of {:.3f} ".format(test_accuracy * 100))

def save_model(class_to_idx, path, model, structure, optimizer):

    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {
        'batch_size': bs,
        'input_size': input_size,
        'output_size': output_size,
        'class_to_idx': model.class_to_idx,
        'epochs': epochs,
        'arch': 'vgg16',
        'learning_rate': learning_rate,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'model_classifier' : model.classifier             
    }

    torch.save(checkpoint, 'checkpoint.pth')


def load_model(path = 'checkpoint.pth'):

    checkpoint = torch.load(path)

    classifier = checkpoint['classifier']

    structure = checkpoint['arch']

    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    if structure == 'alexnet':
        model = models.alexnet(pretrained=True)
    if structure == 'densenet121':
        model = models.densenet121(pretrained=True)

    checkpoint = torch.load(path)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['model_classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def predict(image_path, model, topk=5, power = 'GPU', category_names = 'cat_to_name.json'):


    model.cpu()
    model.eval()
    
    
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    
    
    l_probabilities = model(image)

    probabilities = torch.exp(l_probabilities)
    top_p, top_class = probabilities.topk(topk, dim=1)
    
    classes = top_class[0].tolist()
    probability = top_p[0].tolist()
    
    
    return probability, classes

