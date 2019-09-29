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
import json

d_checkpoint        = 'checkpoint.pth'
d_categories_json   = 'cat_to_name.json'
d_topk              = 1


arg_parser = argparse.ArgumentParser(description='Predict arguments parser')

arg_parser.add_argument('image_path', metavar='FILE_PATH', type=str, help='image file')
arg_parser.add_argument('checkpoint', metavar='FILE_PATH', type=str, default=d_checkpoint, help='checkpoint file')
arg_parser.add_argument('--category_names', metavar='FILE_PATH', type=str, default=d_categories_json, help='Categories to name json')
arg_parser.add_argument('--top_k', metavar='INT', type=int, default=d_topk, help='Number of Top Pred')
arg_parser.add_argument('--gpu', action='store_true', help='Implement GPU')

def cat_name(cat_filepath):
    with open(cat_filepath, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
    
def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_units = checkpoint['hidden_size']
    dropout = checkpoint['dropout']
    
    model = models.__dict__[arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=dropout)),
                              ('fc2', nn.Linear(hidden_units, hidden_units)),
                              ('relu2', nn.ReLU()),
                              ('dropout2', nn.Dropout(p=dropout)),
                              ('fc3', nn.Linear(hidden_units, output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    
    model.load_state_dict(checkpoint['model_state'])
    
    return model

def process_image(image):
    # Process a PIL image for use in a PyTorch model
    try:
        image = Image.open(image)
    except IOError:
        print("error", image)
        return
    
    loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    image = loader(image).float()
    np_image = np.array(image)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image


def gpu(args):
    if args.gpu and torch.cuda.is_available():
        args.gpu = True
    else:
        args.gpu = False
    gpu = args.gpu
    
    return gpu


def predict(args):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = process_image(args.image_path)
    image = torch.autograd.Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)
    
    model = load_checkpoint(args.checkpoint)
    model.train(False)
    model.eval()

    if gpu:
        image.to("cuda")
    
    results = model(image).topk(args.top_k)
    
    probs = torch.nn.functional.softmax(results[0].data, dim=1).cpu().numpy()[0]
    classes = results[1].data.cpu().numpy()[0]
#     ind = list(probs).index(max(list(probs)))
#     max_probs = probs[ind]
#     pred_class = classes[ind]
    cat_to_name = cat_name(args.category_names)
#     if str(pred_class) in (cat_to_name.keys()):
#         pred_class = cat_to_name[str(pred_class)]
    names = []
    for name in classes:
        if str(name) in (cat_to_name.keys()):
            names.append(cat_to_name[str(name)])
        else:
            names.append("NA")
    
    print("Prediction: {}".format(names))
    print("Probability: {}".format(list(probs)))
    
    return names, probs



if __name__ == '__main__':

    args = arg_parser.parse_args()
    
    #
    gpu = gpu(args)

    #
    predict(args)