import cv2
import numpy as np
import torch
from neuralnetwork import Net 

def apply_model(path):
    print()
    print()

    print("#######################################")

    print("WARNING : NOT FOR REAL MEDICAL PURPOSES")

    print("#######################################")

    print()
    print()

    img_size = 50

    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img, (img_size,img_size))

    img_array=np.array(img)
    img_array=img_array/255
    img_array=torch.Tensor(img_array)

    net=Net()
    net.load_state_dict(torch.load("save_model.pth"))
    net.eval() #lets pytorch know that we are not training just evaluating 

    net_out=net(img_array.view(-1,1,img_size,img_size))[0]
    print(net_out)


    if net_out[0] >= net_out[1]:
        print("Prediction = BENIGN")
        print()
        print("Confidence = ", float(net_out[0]))
        print()
        
    else:
         print("Prediction =MALIGNANT")
         print()
         print("Confidence = ", float(net_out[0]))
         print()

apply_model("/Users/jaeeponde/Desktop/projectcurecancer/testpics/mal2.jpg")
