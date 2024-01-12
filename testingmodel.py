import numpy as np
import torch
from neuralnetwork import Net

img_size=50
net = Net()
net.load_state_dict(torch.load('save_model.pth'))
net.eval()

testing_data=np.load("melanoma_testing_data.npy",allow_pickle=True)



test_X = torch.Tensor([item[0] for item in testing_data])
#THIS PUTS ALL IMAGE ARRAYS IN A TENSOR OBJECT 
test_X=test_X/255 #noramlising the data between 0 and 1 


test_Y=torch.Tensor([item[1] for item in testing_data])

correct=0
total=0

with torch.no_grad():

    #tells pytorch to stop saving gradients. it just makes the code slightly faster 

    for i in range(len(test_X)):

        output=net(test_X[i].view(-1,1,img_size,img_size))[0]

        if output[0] >= output[1]:
            guess = "BENIGN"
        else:
            guess = "MALIGNANT"

        real_label = test_Y[i]

        if real_label[0] >= real_label[1]:
            answer = "BENIGN"
        else:
            answer= "MALIGNANT"

        if guess == answer:
            correct+=1
            total+=1
        else:
            total+=1

print("Accuracy :", correct/total)
        

        

        



