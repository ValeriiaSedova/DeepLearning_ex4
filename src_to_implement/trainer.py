import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd

class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # TODO: perform following steps:
        # [x] - reset the gradients
        # [x] - propagate through the network
        # [x] - calculate the loss
        # [x] - compute gradient by backward propagation
        # [x] - update weights
        # [x] - return the loss
        
        self._optim.zero_grad()
        x = self._model.forward(x)
        loss = self._crit(x, y)
        loss.backward()
        self._optim.step()
        return loss
        
        
    
    def val_test_step(self, x, y):
        # TODO:
        # [x] - predict
        # [x] - propagate through the network and calculate the loss and predictions
        # [x] - return the loss and the predictions
        
        pred = self._model.forward(x)
        loss = self._crit(x, y)
        return loss, pred
        
    def train_epoch(self):
        # TODO:
        # [x] - set training mode
        # [x] - iterate through the training set
        # [x] - transfer the batch to "cuda()" -> the gpu if a gpu is given
        # [x] - perform a training step
        # [x] - calculate the average loss for the epoch and return it
        
        self._model.train()
        losses = []
        for sample, label in self._train_dl:
            sample, label = sample.cuda(), label.cuda()
            loss = self.train_step(sample, label)
            losses.append(loss)
        return t.mean(t.tensor(losses))

    def val_test(self):
        # TODO:
        # [x] - set eval mode
        # [x] - disable gradient computation
        # [x] - iterate through the validation set
        # [x] - transfer the batch to the gpu if given
        # [x] - perform a validation step
        # [x] - save the predictions and the labels for each batch
        # [x] - calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # [x] - return the loss and print the calculated metrics
        
        self._model.eval()
        t.no_grad()
        losses = []
        for sample, label in self._val_test_dl:
            sample, label = sample.cuda(), label.cuda()
            loss, pred = self.val_test_step(sample, label)
            losses.append(loss)
        return t.mean(t.tensor(losses))

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # TODO:
        # [x] - create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        validation_losses = []
        epoch = 0
        epoch_cd = 0
        prev_vl = 0
        while True:
      
            # TODO:
            # [x] - stop by epoch number
            # [x] - train for a epoch and then calculate the loss and metrics on the validation set
            # [x] - append the losses to the respective lists
            # [x] - use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # [x] - check whether early stopping should be performed using the early stopping criterion and stop if so
            # [x] - return the losses for both training and validation
            
            train_loss = self.train_epoch()
            valid_loss = self.val_test()

            if valid_loss > prev_vl: epoch_cd += 1
            else:                    epoch_cd =  0

            train_losses.append(train_loss)
            validation_losses.append(valid_loss)   

            if epoch_cd >= self._early_stopping_patience:
                break

            epoch += 1
        self.save_checkpoint(epoch)   
        return train_losses, validation_losses
        
        
    # def write_loss(self, train_loss, valid_loss):
    #     df = pd.read_csv('train_data.csv',sep = ';')
    #     df.add