import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

DEFAULT_MODEL_PATH = 'models/'

class CNN(torch.nn.Module):
  
    def __init__(self, model_path=DEFAULT_MODEL_PATH, *args, **kwargs):
      
        # Call to super-constructor
        super(CNN, self).__init__()
        #conv1_out_size = self.compute_layer_size(self.input_size, self.kernel_size_1,  self.padding_size_1, self.stride_size_1, self.dilation_size_1)
        self.path = model_path     
        # Sets up our layers
        self.conv1 = torch.nn.Conv2d(3, 64, 3)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(64, 64, 3)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = torch.nn.Conv2d(64, 128, 3)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.conv4 = torch.nn.Conv2d(128, 128, 3)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = torch.nn.Conv2d(128, 256, 3)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        self.conv6 = torch.nn.Conv2d(256, 256, 3)
        torch.nn.init.xavier_uniform_(self.conv6.weight)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv7 = torch.nn.Conv2d(256, 512, 3)
        torch.nn.init.xavier_uniform_(self.conv7.weight)
        self.conv8 = torch.nn.Conv2d(512, 512, 2)
        torch.nn.init.xavier_uniform_(self.conv8.weight)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = torch.nn.Linear(512, 64)

        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x):
        
        # First convolutional layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool4(x)

        # First fully connected layer
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = F.relu(x)
        
        # Second fully connected module
        x = self.fc2(x)
        return x

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

    def train_model(self, train_loader, valid_loader,
                  num_epochs=10, save_mode='every',
                    smart_detection=True,  silent=False):

        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.0)
        criterion = torch.nn.CrossEntropyLoss()
        validation_loss_per_epoch = []
        training_loss_per_epoch = []
        training_error_per_epoch = []
        valid_error_per_epoch = []
        validation_accuracy = 0
        
        for epoch in range(num_epochs):
            print("Starting epoch ", epoch)
            training_loss = []
            correct_trainning = 0
            total = 0
            correct = 0
            lmbd = 0.9 
            for i, data in enumerate(train_loader, 0):
                inputs,target = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    target = target.cuda()
                optimizer.zero_grad()
                input_var = torch.autograd.Variable(inputs)
                target_var = torch.autograd.Variable(target)
                output = self(input_var)
                loss = criterion(output, target_var)
                reg = 1e-6
                l2_loss = None
                for name, param in self.named_parameters():
                    if 'bias' not in name: 
                        if l2_loss is None:
                            l2_loss = 0.5 * reg *torch.sum(torch.pow(param, 2))
                        else:
                            l2_loss += (0.5 * reg *torch.sum(torch.pow(param, 2)))

                loss += l2_loss
                training_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(output.data, 1)
                total += target_var.size(0)
                correct += (predicted == target_var).sum().item()
                if i%50==0 and not silent:
                    print('Epoch : %d accuracy : %.3f ' % (epoch, 100*(correct/total)))

            training_loss_per_epoch.append(np.mean(training_loss))
            training_error_per_epoch.append(1 - (correct/total))
            validation_loss = [];
            val_size = len(list(valid_loader))
            val_correct = 0 
            val_total = 0
            for i, data in enumerate(valid_loader, 0):
                inputs,target = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    target = target.cuda()
                input_var = torch.autograd.Variable(inputs)
                target_var = torch.autograd.Variable(target)
                output = self.forward(input_var)
                loss = criterion(output, target_var)
                validation_loss.append(loss.item())
                _, predicted = torch.max(output.data, 1)
                val_total += target_var.size(0)
                val_correct += (predicted == target_var).sum().item()
                if i%50==0 and not silent:
                    print('Epoch : %d validation accuracy: %.3f ' % (epoch, 100*(val_correct/val_total)))
            validation_loss_per_epoch.append(np.mean(validation_loss))
            validation_accuracy = val_correct/val_total
            valid_error_per_epoch.append(1 - (val_correct/val_total))
            print('Epoch : %d validation Acc : %.3f' % (epoch, 100.*val_correct/val_total))
            print('--------------------------------------------------------------')
            print("done epoch ", epoch)

            if save_mode == 'every' or epoch % save_mode == 0:
                self.save('{}_{}.{}'.format(self.path, epoch,'pth'))

        return training_loss_per_epoch,validation_loss_per_epoch,training_error_per_epoch,valid_error_per_epoch,validation_accuracy

    def compute_layer_size(self, input_size, kernel_size, padding = 0, stride = 1, dilation = 1):
    
        ks = (kernel_size) + (dilation - 1) * (kernel_size - 1)
        
        return np.floor((input_size - ks - (2 * padding)) / (stride)) - 1