import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import matplotlib.pyplot as plt
import csv
import pandas as pd
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from IPython.display import Image
from ResNet import ResNet
to_img = ToPILImage()


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic=True

train_dir = "data/trainset"
test_dir = "data/testset"
valid_dir = "data/validset"

train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.4,
        ),
        transforms.RandomAffine(
            degrees=(-90,90),
            translate=(1.0, 1.0),
            scale=(0.1, 2.0),
            shear=(-90,90),
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Validation transformations (No need to randomly make validation harder)
valid_trans =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_trans =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_datasets = datasets.ImageFolder(train_dir, train_trans)
valid_datasets = datasets.ImageFolder(valid_dir, valid_trans)

class_names = train_datasets.classes

train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=32, num_workers=1, shuffle=True, pin_memory=True)

validation_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32, num_workers=1, shuffle=True, pin_memory=True)

test_datasets = datasets.ImageFolder(test_dir, train_trans)
dataset_size = len(test_datasets)
indices = list(range(dataset_size))
test_sampler = SequentialSampler(indices)

test_dataloaders = torch.utils.data.DataLoader(test_datasets, shuffle=False, num_workers=1, batch_size=32, pin_memory=False, sampler=test_sampler)

    
def main():
    n_epochs = 50
    cnn = ResNet()

    if torch.cuda.is_available():
        cnn = cnn.cuda()
    training_loss_per_epoch,validation_loss_per_epoch,training_error_per_epoch,valid_error_per_epoch,validation_accuracy = cnn.train_model(train_dataloaders, validation_dataloaders,num_epochs = n_epochs)
    print("Final validation accuracy: ", validation_accuracy)
    
    epoch_range = list(range(n_epochs))
    plt.figure(1)
    plt.plot(epoch_range, training_loss_per_epoch, 'g^', epoch_range, validation_loss_per_epoch, 'bs')
    plt.ylabel("loss")
    plt.xlabel("epoch iteration")
    plt.show()

    plt.figure(2)
    plt.plot(epoch_range, training_error_per_epoch, 'g^', epoch_range, valid_error_per_epoch, 'bs')
    plt.ylabel("error")
    plt.xlabel("epoch iteration")
    plt.show()

    final_predictions = np.empty([0,0])
    final_predictions2 = np.empty([0,0])
    cnn.eval()
    for i, data in enumerate(test_dataloaders):
        inputs,target = data
        inputs = inputs.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(inputs)
        target_var = torch.autograd.Variable(target)
        output = cnn(input_var)
        _, predicted = torch.max(output.data, 1)
        final_predictions2 = np.append(final_predictions2, predicted.cpu().numpy())
    
    dictionary = {}
    i = 0
    for f in os.listdir('./testset/test/'):
        id = int(os.path.basename(os.path.splitext(f)[0]))
        dictionary[id] = final_predictions2[i]
        i+=1
    for key in sorted(dictionary.keys()):
        final_predictions = np.append(final_predictions, dictionary[key])

    df = pd.read_csv('./sample_submission.csv', delimiter= ',')
    map_classes = np.vectorize(lambda t: class_names[t])
    df['label'] = map_classes(final_predictions.astype(int).flatten())

    df.to_csv('./prediction.csv', index=False, sep=',')   

if __name__=='__main__':
    main()
