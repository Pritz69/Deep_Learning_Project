import torch
import jovian
import torchvision
import matplotlib
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split

dataframe_raw = pd.read_csv('winequalityred.csv',delimiter = ';')

dataframe_raw.head()

dataframe_raw.shape

input_cols = list(dataframe_raw.columns)[:-1]
output_cols = ['quality']
input_cols,output_cols

def dataframe_to_arrays(dataframe):
    dataframe1 = dataframe_raw.copy(deep=True)
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array

inputs_array, targets_array = dataframe_to_arrays(dataframe_raw)
inputs_array, targets_array

inputs = torch.from_numpy(inputs_array).type(torch.float)
targets = torch.from_numpy(targets_array).type(torch.float)
inputs,targets

print('Shape of input tensor and target tensor::  ',inputs.shape, targets.shape)
print('datatype of input tensor and target tensor::  ',inputs.dtype, targets.dtype)

dataset = TensorDataset(inputs, targets)
dataset

train_ds, val_ds = random_split(dataset, [1300, 299])
batch_size=50
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

for xb, yb in train_loader:
    print("inputs:", xb)
    print("targets:", yb)
    break

print(xb.dtype,yb.dtype)

input_size = len(input_cols)
output_size = len(output_cols)
input_size,output_size

class WineQuality(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size,output_size) #???                  # fill this (hint: use input_size & output_size defined above)
        
    def forward(self, xb): 
        out = self.linear(xb) #???                          # fill this
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)          
        # Calcuate loss
        loss = F.l1_loss(out,targets) #???                          # fill this
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.l1_loss(out,targets) #???                           # fill this    
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 100th epoch
        if (epoch+1) % 100 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))
            
model=WineQuality()

list(model.parameters())

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history

result = evaluate(model, val_loader)
print(result)

epochs = 1000 
lr = 1e-2    
history1 = fit(epochs, lr, model, train_loader, val_loader)

plt.title('val_loss vs. No. of epochs');
loss_mat = [res['val_loss'] for res in [result] + history1]
plt.plot(loss_mat, '-x')
plt.xlabel('epoch')
plt.ylabel('val_loss')

val_loss = loss_mat[-1]

epochs = 1000
lr = 1e-3
history2 = fit(epochs, lr, model, train_loader, val_loader)

plt.title('val_loss vs. No. of epochs');
loss_mat = [res['val_loss'] for res in [result] + history2]
plt.plot(loss_mat, '-x')
plt.xlabel('epoch')
plt.ylabel('val_loss')

val_loss = loss_mat[-1]

epochs = 1000
lr = 1e-4
history3 = fit(epochs, lr, model, train_loader, val_loader)

plt.title('val_loss vs. No. of epochs');
loss_mat = [res['val_loss'] for res in [result] + history3]
plt.plot(loss_mat, '-x')
plt.xlabel('epoch')
plt.ylabel('val_loss')

val_loss = loss_mat[-1]

epochs = 1000
lr = 1e-5
history4 = fit(epochs, lr, model, train_loader, val_loader)

plt.title('val_loss vs. No. of epochs');
loss_mat = [res['val_loss'] for res in [result] + history4]
plt.plot(loss_mat, '-x')
plt.xlabel('epoch')
plt.ylabel('val_loss')

val_loss = loss_mat[-1]

epochs = 1500
lr = 1e-6
history5 = fit(epochs, lr, model, train_loader, val_loader)

plt.title('val_loss vs. No. of epochs');
loss_mat = [res['val_loss'] for res in [result] + history5]
plt.plot(loss_mat, '-x')
plt.xlabel('epoch')
plt.ylabel('val_loss')

val_loss = loss_mat[-1]

def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)#???                
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)
    
input, target = val_ds[62]
predict_single(input, target, model)