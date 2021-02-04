import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# TODO: this can be accomplished using the already imported pandas and sklearn.model_selection modules
df = pd.read_csv('data.csv',sep=';')
data_train, data_valid = train_test_split(df, train_size = 0.75, test_size = 0.25)
dataset_train = ChallengeDataset(data_train, mode='train')
dataset_valid = ChallengeDataset(data_valid, mode='val')
# TODO: set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
data_loader = t.utils.data.DataLoader(  dataset_train,
                                        batch_size = 1,
                                        shuffle = True)
# TODO: create an instance of our ResNet model
model = model.ResNet()
# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
loss = t.nn.BCELoss()
# set up the optimizer (see t.optim)
learning_rate = 1e-4
optimizer = t.optim.Adam(model.parameters(), lr = learning_rate)
# TODO: create an object of type Trainer and set its early stopping criterion
trainer = Trainer(model, loss, optimizer, dataset_train, dataset_valid, True, 5)
# TODO: go, go, go... call fit on trainer
res = trainer.fit()

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')