# Imports here
import matplotlib.pyplot as plt
import os
import torch
import argparse
import json
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image


d_checkpoints_dir   = ''
d_categories_json   = 'cat_to_name.json'
d_arch              = 'vgg19_bn'
d_learning_rate     = 0.0025
d_hidden_units      = 4096
d_dropout           = 0.1
d_epochs            = 1 #just for testing if script works



arg_parser = argparse.ArgumentParser(description='Training arguments parser')

arg_parser.add_argument('data_dir', metavar='DIR', type=str, help='Dataset dir')
arg_parser.add_argument('--save_dir', metavar='DIR', type=str, default=d_checkpoints_dir, help='Save checkpoints dir')
arg_parser.add_argument('--cat_json', metavar='FILE', type=str, default=d_categories_json, help='Categories to name json')
arg_parser.add_argument('--arch', metavar='ARCH', type=str, default=d_arch, choices=['vgg19_bn','densenet121'], help='Model architecture')

arg_parser.add_argument('--learning_rate', default=d_learning_rate, type=float, help='Learning Rate')
arg_parser.add_argument('--hidden_units', default=d_hidden_units, type=int, help='Hidden Units')
arg_parser.add_argument('--dropout', default=d_dropout, type=float, help='Dropout Prob')
arg_parser.add_argument('--epochs', default=d_epochs, type=int, help='Training Epochs')
arg_parser.add_argument('--gpu', action='store_true', help='Implement GPU')

def transforms_imgs(args):

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_dir = os.path.join(args.data_dir+'/train')
    valid_dir = os.path.join(args.data_dir+'/valid')
    test_dir = os.path.join(args.data_dir+'/test')
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return trainloader, validloader, testloader, train_data

def model_build(args):
    model = models.__dict__[args.arch](pretrained=True)
    if args.arch == 'vgg_19bn':
        input_size = model.classifier[0].in_features
    else:
        input_size = model.classifier.in_features
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size , args.hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=args.dropout)),
                              ('fc2', nn.Linear(args.hidden_units, args.hidden_units)),
                              ('relu2', nn.ReLU()),
                              ('dropout2', nn.Dropout(p=args.dropout)),
                              ('fc3', nn.Linear(args.hidden_units, args.output_size )),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    return model, criterion, optimizer

def gpu(args):
    if args.gpu and torch.cuda.is_available():
        args.gpu = True
    else:
        args.gpu = False
    gpu = args.gpu
    gpu = args.gpu
    
    return gpu
        

def validation(args, model, validloader, criterion):
    valid_loss = 0
    valiad_accuracy = 0
    for images, labels in validloader:
        
        if gpu:
            images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        valiad_accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, valiad_accuracy

def model_train(args, model, trainloader, criterion):

    epochs = args.epochs
    print_every = 40
    steps = 0

    if gpu:
        model.to('cuda')
        criterion.to('cuda')
    print("Start to train the model")
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            
            if gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                
                model.eval()

                
                with torch.no_grad():
                    valid_loss, accuracy = validation(args, model, validloader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                
                model.train()
    save_model(args,model)
    print("Model has been trained and saved!")
    
def save_model(args,model):
    input_size     = model.classifier[0].in_features
    model.class_to_idx = train_data.class_to_idx
    #need to create the save_dir in advance if other than default
    checkpoint_filename = args.save_dir+'checkpoint.pth'
    checkpoint = {
    'arch': args.arch,
    'input_size': input_size,
    'output_size': args.output_size,
    'hidden_size': args.hidden_units,
    'dropout': args.dropout,
    'optim_state': optimizer.state_dict(),
    'model_state': model.state_dict()}
    torch.save(checkpoint, checkpoint_filename)
    

if __name__ == '__main__':


    args = arg_parser.parse_args()
    
    #set i/o size
    with open(args.cat_json, 'r') as f:
        cat_to_name = json.load(f)
    args.output_size    = len(cat_to_name)
    
    #load data
    trainloader, validloader, testloader, train_data = transforms_imgs(args)

    #build model and set criterion and optimizer
    model, criterion, optimizer = model_build(args)
    
    #check gpu setting
    gpu = gpu(args)

    #train the model
    model_train(args, model, trainloader, criterion)