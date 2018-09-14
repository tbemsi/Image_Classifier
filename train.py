# -*- coding: utf-8 -*-
import time
import json
import copy
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from PIL import Image
from collections import OrderedDict

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms


data_dir = '/home/workspace/aipnd-project/flowers'
train_dir = args.data_directory + '/train'
valid_dir = args.data_directory + '/valid'
test_dir = args.data_directory + '/test'


parser = argparse.ArgumentParser(description='Testing app for Image Classifier')
parser.add_argument('data_directory', type=str, help='Directory containing data' , default='data_dir')
parser.add_argument('--gpu', type=bool, default=True, help='Whether to use GPU during training or not')
parser.add_argument('--epochs', type=str, default=10, help='Choose number of epochs to train for')
parser.add_argument('--lr', type=str, default=0.0001, help='Choose learning rate to use for training')
parser.add_argument('--hidden_sizes', type=list, default=[4096,1024], help='Input list of hidden layer sizes')
args = parser.parse_args()

device = 'cuda' if args.gpu == True else 'cpu'

# Define your transforms for the training, validation, and testing sets
data_transforms = {
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
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# TODO: Load the datasets with ImageFolder
dirs = {'train':train_dir,
        'valid': valid_dir,
        'test': test_dir}

image_datasets = {x: datasets.ImageFolder(dirs[x],   transform=data_transforms[x]) for x in ['train', 'valid', 'test']}


# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                              shuffle=True) for x in ['train', 'valid', 'test']}

dataset_sizes = {x: len(image_datasets[x]) 
                              for x in ['train', 'valid', 'test']}

class_names = image_datasets['train'].classes

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


#Loading a pre-trained model
model = getattr(models, args.arch)(pretrained=True)

#modifying the classifier

input_size = 25088 if args.arch == 'vgg19' else 1024
hidden_sizes = [4096, 1024]
output_size = 102

hidden_layer_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])

layers_map = [('fc1', nn.Linear(input_size, hidden_sizes[0])),
                    ('relu1', nn.ReLU()),
                    ('dropout1', nn.Dropout(0.2))]
count = 1
for h1, h2 in hidden_layer_sizes:
    count +=1
    layers_map.append(('fc'+str(count), nn.Linear(h1,h2)))
    layers_map.append(('relu'+str(count), nn.ReLU()))
    layers_map.append(('dropout'+str(count), nn.Dropout(0.2)))
layers_map.append(('fc'+str(count+1), nn.Linear(hidden_sizes[-1], output_size)))
layers_map.append(('output', nn.LogSoftmax(dim=1)))
classifier = nn.Sequential(OrderedDict(layers_map))

model.classifier = classifier
if args.gpu:
    model.cuda()
else:
    model.cpu()

for param in model.parameters():
    param.requires_grad = False
    
model.classifier = classifier

criteria = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
eps=10

def train_model(model, criterion, optimizer, scheduler,    
                                      num_epochs=args.epochs, device='cuda'):
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

model_ft = train_model(model, criteria, optimizer, sched, eps, 'cuda')

# ## Save the checkpoint
model.class_to_idx = image_datasets['train'].class_to_idx
model.cpu()
torch.save({'arch': args.arch,
            'state_dict': model_ft.state_dict(), 
            'class_to_idx': model_ft.class_to_idx}, 
            'classifier.pth')

model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'arch': args.arch,'classifier': model_ft.classifier, 'optimizer': optimizer,
              'optimizer_dict': optimizer.state_dict(),
            'state_dict': model_ft.state_dict(), 
            'class_to_idx': model_ft.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')
