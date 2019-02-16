import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import logging
import importlib
import copy
from collections import OrderedDict
import matplotlib as mpl
#import matplotlib.pylot as plt
print(dir(mpl))

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic=True

DEFAULT_NUM_EPOCHS = 10

DEFAULT_MODEL_PATH = '/content/drive/My Drive/IFT 6135/question2_model'

DEFAULT_VALID_CONV_PARAMS = [
    'input_size', 'output_size', 'padding_size', 'stride_size', 'dilation_size'
]

# Hyper-parameters
HYPER_PARAMS_VGG = OrderedDict(
    
    conv1 = dict(
        input_size = 1,
        output_size = 64,
        kernel_size = (3,3),
        padding_size = 0,
        stride_size = 1,
        dilation_size = 1,
    ),
     
    relu1 = dict(),
    
    conv2 = dict(
        input_size = 64,
        output_size = 128,
        kernel_size = (3,3),
        padding_size = 0,
        stride_size = 1,
        dilation_size = 1,
    ),
     
    relu2 = dict(),
    
    pool1 = dict(
        kernel_size = (2,2),
        stride_size= 2,
    ),
    
    conv3 = dict(
        input_size = 128,
        output_size = 256,
        kernel_size = (3,3),
        padding_size = 2,
        stride_size = 2,
        dilation_size = 1,
    ),
    
    relu3 = dict(),
    
    conv4 = dict(
        input_size = 256,
        output_size = 256,
        kernel_size = (3,3),
        padding_size = 2,
        stride_size = 2,
        dilation_size = 1,
    ),
    
    relu4 = dict(),
    
    pool2 = dict(
        kernel_size = (2,2),
        stride_size = 2,
    ),
    
    pool3 = dict(
        kernel_size = (2,2),
        stride_size = 2,
    ),
    
    fc1 = dict(
        input_size=256,
        output_size=64,
    ),
    
    relu5 = dict(),
    
    fc2 = dict(
        input_size=64,
        output_size=10
    ),
    
 
)


# Sets up logging
importlib.reload(logging)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'
)


class CNN(torch.nn.Module):
  
    def __init__(self, optimizer=torch.optim.SGD,
                 learning_rate =1e-2, 
                 criterion=torch.nn.CrossEntropyLoss,
                 use_cuda=True, model_path=DEFAULT_MODEL_PATH, *args, **kwargs):
        """
        Initializes a generic CNN. Layers and parameters can be passed via 
        keyword arguments.
        
        """
      
        # Call to super-constructor
        super(CNN, self).__init__()
        print(learning_rate)
        self.path = model_path
        # Sets up separate ordered dictionaries for conv and fc parts
        self.conv_params = OrderedDict()
        self.fc_params = OrderedDict()
               
        # Parses dictionary of parameters
        previous = None
        for key, value in kwargs.items():
            
            if 'conv' in key:
                try:
                    self.conv_params[key] = nn.Conv2d(
                        in_channels = value.get('input_size', None),
                        out_channels = value.get('output_size', None),
                        stride = value.get('stride_size', 1),
                        padding = value.get('padding_size', 0),
                        dilation = value.get('dilation_size', 1),
                        kernel_size = value.get('kernel_size', (3,3)),
                    )
                    previous = 'conv'
                except Exception as e:
                    logging.warning(
                        'Unable to create Conv layer {}. Exception: {} '.format(
                        key, e)
                    )

            elif 'relu' in key:
                # Need to check which type of layer was before this one
                if previous == 'conv':
                    self.conv_params[key] = nn.ReLU()
                elif previous == 'fc':
                    self.fc_params[key] = nn.ReLU()
                else:
                    logging.warning(
                    'Cannot start with an activation layer. Ignoring {}'.format(
                        key)
                    )
            elif 'pool' in key:
                try:
                    self.conv_params[key] = nn.MaxPool2d(
                        stride = value.get('stride_size', None),
                        kernel_size = value.get('kernel_size', None),
                    )
                except Exception as e:
                    logging.warning('Unable to create MaxPool layer {}'.format(
                        key)
                    )
                    previous = 'conv'
            elif 'fc' in key:  
                try:
                    self.fc_params[key] = nn.Linear(
                        value.get('input_size', None),
                        value.get('output_size', None),
                    )
                    previous = 'fc'
                except Exception as e:
                    logging.warning(
                        'Unable to create linear layer {}. Exception: {} '.format(
                        key, e)
                    )
            elif key in self.conv_params or key in self.fc_params:
                logging.warning(
                    'Parameter {} is already set. Ignored redefinition.'.format(
                        key)
                )
            else:
                logging.warning(
                    'Ignored unrecognized parameter {}.'.format(
                        key)
                )
        
        # Sets up layers based on parameters
        self.conv = nn.Sequential(self.conv_params)
        self.fc = nn.Sequential(self.fc_params)
        
        # Sets up loss and optimizers
        self.optimizer = optimizer(
            self.parameters(), lr=learning_rate, momentum=0.0
        )
        self.criterion = criterion()
        
        # Sets up other parameters
        if torch.cuda.is_available() and use_cuda:
            self.use_cuda = True
        
    def forward(self, x):      
        """Forward pass of the neural network. """
        return self.fc(self.conv(x).squeeze())
    
    def save(self, path=None):
        """Saves the model to the desired path."""
        if path is None:
            path = self.path
            
        torch.save(self.state_dict(), path)
        
    def load(self, path=None):
        """Loads the model from desired path."""
        if path is None:
            path = self.path
            
        self.load_state_dict(torch.load(path))
        
    def test(self, test_loader):
      total = 0
      correct = 0
      for i,data in enumerate(test_loader):
        inputs,targets = data
        if self.use_cuda:
           inputs, targets = inputs.cuda(), targets.cuda()
        
        outputs = self(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
      return 100*(correct/total)
        
    def train_model(self, train_loader, valid_loader,
                    num_epochs=DEFAULT_NUM_EPOCHS, save_mode='every',
                    smart_detection=True,  silent=False):
        """Trains the model.
        

        :param train_loader: DataLoader for the training set.
        :param test_loader: DataLoader for the testing set.
        :param save_mode: Mode of saving. Choose between:
            None: No saving.
            (string) 'every': Save at every epoch.
            (string) 'end': Save at end of training.
            (int) n: Save at every n'th epoch.
        :param smart_detection: Autodetect best epoch.
        :param num_epochs: Number of epochs to train for.
        :param silent: Whether or not to output messages at every epoch.
        
        
        """
        
        train_losses = []
        valid_losses = []
        
        train_error = []
        valid_error = []
        
        for epoch in range(num_epochs):
            print("running epoch: ", epoch)  
            losses = []
            total = 0
            correct = 0
            self.train()
            for batch, (inputs, targets) in enumerate(train_loader):
                
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.data.item())
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()


                if batch%50==0 and not silent:
                    logging.info('Epoch : %d Loss : %.3f ' % (epoch, np.mean(losses)))
                
            train_error.append(100. - 100.*(correct/total))
            total = 0
            correct = 0
            self.eval()
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()
                
            train_losses.append(np.mean(losses))
            print('Epoch : %d Test Acc : %.3f' % (epoch, 100.*correct/total))
            print('--------------------------------------------------------------')
            valid_error.append(100. - 100.*(correct/total))
            
            if save_mode == 'every' or epoch % save_mode == 0:
                self.save('{}_{}'.format(self.path, epoch))
                
        return train_error,valid_error     
    
        