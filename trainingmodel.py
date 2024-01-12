import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from neuralnetwork import Net

img_size = 50 

training_data = np.load("melanoma_training_data.npy",allow_pickle=True)

# this is how each element in the numpy array looks : [img_array, [0,1]]

# for row in training_data:
#     print(row[0])
#     print(row[1])
#     print()
#     print()
#     print()


train_X = torch.Tensor([item[0] for item in training_data])
#THIS PUTS ALL IMAGE ARRAYS IN A TENSOR OBJECT 
train_X=train_X/255 #noramlising the data between 0 and 1 

# for row in train_X:
#     print(row)
#     print()

#     input()

train_Y=torch.Tensor([item[1] for item in training_data])

net=Net()
#creating new object of class net 

optimiser = optim.Adam(net.parameters(),lr=0.001) # lr= learning rate 

loss_function = nn.MSELoss() # meansquared error loss function 

batch_size = 100 
# number of images being passed at a time through the neural network 

epochs = 2

for epoch in range(epochs):

    for i in range(0,len(train_X),batch_size): 

        print("epoch : ", epoch+1)
        print("fraction complete : ", i/len(train_X))

        batch_X = train_X[i: i+batch_size].view(-1,1,img_size,img_size)
        batch_Y = train_Y[i: i+batch_size]

        optimiser.zero_grad() #resetting gradients of model parameters to 0 from the last batch 

        outputs = net(batch_X)
        # [0,1] : one hot vector 
        # [0.35,0.66] : output 

        loss = loss_function(outputs,batch_Y)
        #finds loss between predicted output and actual one hot vector label

        loss.backward() #backward propagation to go back and minimise the losses 

        optimiser.step()
        #optimiser updates model parameters based on the gradient 


torch.save(net.state_dict(),"save_model.pth")












