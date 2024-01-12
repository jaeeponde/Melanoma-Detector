import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

img_size = 50 

#location of image files 

ben_training = "melanoma_cancer_dataset/train/benign/"
mal_training = "melanoma_cancer_dataset/train/malignant/"



ben_testing = "melanoma_cancer_dataset/test/benign/"
mal_testing = "melanoma_cancer_dataset/test/malignant/"

ben_training_data =[]
mal_training_data =[]

ben_testing_data =[]
mal_testing_data =[]


for filename in os.listdir(ben_training):

    try: 
       #print(filename)
        path = ben_training + filename
        #print(path)

        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img, (img_size,img_size))

        img_array=np.array(img)

        #[1,0] : benign [0,1] : melanoma

        ben_training_data.append([img_array, np.array([1,0])])

    except:
        pass

for filename in os.listdir(mal_training):

    try: 
       #print(filename)
        path = mal_training + filename
        #print(path)

        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img, (img_size,img_size))

        img_array=np.array(img)

        #[1,0] : benign [0,1] : melanoma

        mal_training_data.append([img_array, np.array([0,1])])

    except:
        pass

for filename in os.listdir(ben_testing):

    try: 
       #print(filename)
        path = ben_testing + filename
        #print(path)

        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img, (img_size,img_size))

        img_array=np.array(img)

        #[1,0] : benign ; [0,1] : melanoma

        ben_testing_data.append([img_array, np.array([1,0])])

    except:
        pass


for filename in os.listdir(mal_testing):

    try: 
       #print(filename)
        path = mal_testing + filename
        #print(path)

        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img, (img_size,img_size))

        img_array=np.array(img)

        #[1,0] : benign [0,1] : melanoma this works as a label for the code to recognise weather an image is benign or malignant 

        mal_testing_data.append([img_array, np.array([0,1])])

    except:
        pass

ben_training_data=ben_training_data[0:len(mal_training_data)]

print ()
print ()
bentraincount = len(ben_training_data)
maltraincount = len(mal_training_data)
print("Benign training count: " , bentraincount)
print("Malignant training count: " , maltraincount)
print ()
bentestcount = len(ben_testing_data)
maltestcount = len(mal_testing_data)
print("Benign training count: " , bentestcount)
print("Malignant training count: " , maltestcount)

training_data = ben_training_data+mal_training_data
np.random.shuffle(training_data)
melanoma_training_data = np.array(training_data,dtype=object)
np.save("melanoma_training_data.npy", melanoma_training_data)
#np.save("melanoma_training_data.npy", training_data)


testing_data = ben_testing_data+mal_testing_data
np.random.shuffle(testing_data)
melanoma_testing_data = np.array(testing_data,dtype=object)
np.save("melanoma_testing_data.npy", melanoma_testing_data)