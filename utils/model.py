import torch
from torch.nn.modules.activation import Softmax
from torch.serialization import save
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tarfile
import sys
import cv2 as cv
from PIL import Image,TarIO
import matplotlib.pyplot as plt
import copy
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib

class LeafDataset(Dataset):
    def __init__(self, filepath, class_names):
        self.names = []
        self.images = []
        self.masks = []
        self.labels = []
        with tarfile.open(filepath,'r') as tf:
            for f in tf.getmembers():
                tmp = '.'.join(f.name.split('.')[0:-1])
                ext = '.' + f.name.split('.')[-1]
                if tmp.split('_')[-1] == 'mask':
                    continue
                im = Image.open(TarIO.TarIO(filepath, tmp + ext)).convert('RGB')
                preprocess = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                input_tensor = preprocess(im)
                input_tensor.requires_grad = True
                self.names.append(tmp)
                self.labels.append(class_names.index(tmp.split('_')[-1]))
                self.images.append(input_tensor)
                # Get mask
                mask_name = tmp + '_mask' + ext
                if mask_name in tf.getnames():
                    im=Image.open(TarIO.TarIO(filepath, mask_name)).convert('RGB')
                    preprocess = transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                    ])
                    input_tensor = preprocess(im)
                    input_tensor.requires_grad = True
                    self.masks.append(input_tensor)
                else:
                    self.masks.append(np.zeros_like(input_tensor.detach().numpy()))
                
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self,idx):
        img_name = self.names[idx]
        label = torch.tensor(self.labels[idx])
        image = self.images[idx]
        mask = self.masks[idx]
        image = image.permute(0,1,2).to(torch.float32)
        return image, label, mask, img_name


class RRR_loss(nn.Module):
    def __init__(self, RRR_weight = 1):
        super().__init__()
        self.RRR_weight = RRR_weight
        return
    
    def forward(self,x,y, masks=None, grads = None):
        loss=nn.CrossEntropyLoss()
        if grads is None: #grads is None):
            return loss(x,y)
        else:
            loss_1=loss(x,y)
            #print(loss_1)
            if masks is not None:
                loss_2 = torch.sum(grads*masks)
                if torch.any(torch.isnan(grads)):
                    print("Grads are NaN")
            else:
                loss_2 = 0
            #print(loss_1, loss_2)
            if(loss_2.item()<1e-16):
                return loss_1
            else:
                return loss_1+self.RRR_weight * torch.pow(10,torch.log10(loss_1/loss_2).ceil())*loss_2

def print_cuda():
    print("CUDA Available: %s" % torch.cuda.is_available())
    print("CUDA Device: %s" % torch.cuda.get_device_name(torch.cuda.current_device()))
    print("GPU Memory: %.2f GB" % (torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory/(1024*1024*1024)))


# Train and test loop

def train_loop(dataloader, model, optimizer, loss_fn, useRRR = False):
    size = len(dataloader.dataset)
    for batch, (X, y, masks, names) in enumerate(dataloader):
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
            masks = masks.cuda()
        # Compute prediction and loss
        pred = model(X)
        if useRRR:  ### RRR Loss:
            grads=torch.autograd.grad(outputs=torch.log1p(pred),inputs=X,grad_outputs=torch.ones_like(pred),retain_graph=True)[0]
            loss = loss_fn(pred, y, masks, grads)
        else:       ### Cross Entropy Loss:
            loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_loop(dataloader, model, loss_fn, name):
    size = len(dataloader.dataset)
    correct=0.0
    loss=0.0
    for images, labels, masks, names in dataloader:
        images, labels, masks = images.cuda(), labels.cuda(), masks.cuda()
        result = model(images)
        _, predicted = torch.max(result, 1)
        correct += torch.eq(labels, predicted).sum().item()
        loss += loss_fn(result, labels)
    print(name + f" Accuracy: {(100*correct/size):>0.1f}%, Avg loss: {loss/size:>8f} \n")
    return correct/size

def build_loaders(class_names, batchsize = 8):
    trainset=LeafDataset('train_data.tar', class_names)
    testset=LeafDataset('test_data.tar', class_names)
    validationset=LeafDataset('validation_data.tar', class_names)
    train_loader = DataLoader(trainset,batch_size=batchsize,num_workers=0,shuffle=True)
    test_loader = DataLoader(testset,batch_size=batchsize,num_workers=0,shuffle=True)
    validation_loader = DataLoader(validationset,batch_size=batchsize,num_workers=0,shuffle=True)
    loaders = (train_loader, validation_loader, test_loader)
    return loaders

def build_network(model_filename, num_classes, pretrained = True, pretrained_weights = False, ignore_load = False):
    if not ignore_load:
        loaded = False
        if os.path.isfile(model_filename):
            print("Loaded model from file")
            net = torch.load(model_filename)
            loaded = True
            return net, loaded
    # Neural network structure
    net = torchvision.models.alexnet(pretrained=pretrained_weights)
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=9216, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=4096, out_features=num_classes, bias=True),
    )
    return net, loaded


def run_train(net, loaders, useRRR, epochs = 30, RRR_weight = 2, lr=0.00001, filename = "", save_result = False):
    train_loader, validation_loader, test_loader = loaders
    train_criterion = nn.CrossEntropyLoss()
    if useRRR:
        train_criterion = RRR_loss(RRR_weight)
    test_criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        net = net.cuda()
        train_criterion = train_criterion.cuda()
        test_criterion = test_criterion.cuda()
    optimizer = optim.RMSprop(net.parameters(), lr=lr)
    best_accuracy = 0.0
    best_model = copy.deepcopy(net)
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_loader, net, optimizer, train_criterion, useRRR)
        _ = net.eval()
        train_accuracy = test_loop(train_loader, net,test_criterion,"Train")
        validation_accuracy = test_loop(validation_loader, net,test_criterion,"Validation")
        _ = net.train()
        ### CHANGED
        if(validation_accuracy > best_accuracy):
            best_model = copy.deepcopy(net)
            best_accuracy = validation_accuracy
    #save net parameters
    net = copy.deepcopy(best_model)
    if save_result:
        torch.save(net, filename)
    return net

def run_test(net, loaders):
    criterion = nn.CrossEntropyLoss()
    train_loader, validation_loader, test_loader = loaders
    net.eval()
    test_loop(test_loader, net, criterion, "Test")