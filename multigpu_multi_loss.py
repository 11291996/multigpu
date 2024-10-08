#this code distributes loss calculation to each gpu making no bottleneck
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import time
import os
from accelerate import Accelerator #multi gpu processing library

input_size = 10
output_size = 1

batch_size = 10000
data_size = 100000
learning_rate = 0.001
epoch = 10

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
        self.output = torch.randn(length, output_size)

    def __getitem__(self, index):
        return self.data[index], self.output[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=False)

class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)

        return output

accelerator = Accelerator(mixed_precision="bf16") #define multiprocessing kernel

model = Model(input_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) 

rand_loader, model, optimizer, criterion = accelerator.prepare(rand_loader, model, optimizer, criterion) #sends data and model to gpu

#to use accumulative loss
accelerator = Accelerator(accumulation_steps=2) #accumulates loss for 2 steps

i = 1
start = time.time()
for epoch in range(epoch): #for tqdms one can send in an argument to print the bar for single processin
    for data, target in rand_loader:
        accelerator.accumulate(model) #for automatic loss accumulation in step() function below
        data = data
        target = target
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        accelerator.backward(loss) #distributed loss calculation
        optimizer.step() #updates weights
        loss = accelerator.gather(loss) #creates a tensor of losses from each gpu
        if i % 10 == 0:
            accelerator.print("Train Step : {}\tLoss : {:3f}".format(i, loss.sum().item())) #print for single process
        i += 1
end = time.time()
accelerator.print("Time taken : {}".format(end - start)) #print for single process


