#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
sns.set(style = "whitegrid")
import os 
import glob as gb 
import cv2 
import tensorflow as tf 
import keras 


# In[49]:


### reading from local 
trainpath = 'C:/Users/moayy/Desktop/seg_train'
testpath  = 'C:/Users/moayy/Desktop/seg_test'
predpath  = 'C:/Users/moayy/Desktop/seg_pred'


# In[50]:


for folder in os.listdir(trainpath + 'seg_train'):
    files = gb.glob(pathname = str(trainpath + 'seg_train'+ folder + '/*.jpg'))
    print(f'for training data , found {len{files} in folder {folder}}')
    


# In[ ]:


# for predictions 
files = gb.glob(pathname + 'seg_pred/*.jpg')
print(f'for prediction data found {len(files)}')     


# In[7]:


#chekcing images 
code = {'building':0 , 'foreset':1 , 'glacier':2 , 'mountain':3 , 'sea':4 , 'street':5}
def getcode(n):
    for x,y in code.items():
        if n==y:
            return x


# In[ ]:


#image sizes
for folder in os.listdir(trainpath + 'seg_train'):
    files =gb.glob(pathname = str(trainpath + 'seg_train//' + folder + '/*.jpg'))
    for file in files:
        image = plt.imread(file)
        size.append(image.shape)
pdf.series(size).value_counts()    # list nums with their num of reptition 


# In[ ]:


#Reading and Resizing images 
s = 100                    # refers to size of image 
x_train = []
y_train = []
for folder in os.listdir(tainpath + 'seg_train'):
    files = gb.glob(pathname= str(trainpath + 'seg_train//' + folder + '/*.jpg'))
    for file in files : 
        image = cv2.imread(file)
        image_array = cv2.resize(image(s,s))
        x_train.append(list(image_array))
        y_train.append(code[folder])


# In[10]:


# now how many items in x_train
print(f'we have {len(x_train)} items in x_train') 


# In[ ]:


# show images + labels 
plt.figure(figsize = (20 , 20 ))
for n , i in enumerate(list(np.random.randint(0, len(x_train) , 36))):
    plt.subplot(6 , 6 , n+1 ) 
    plt.imshow(x_train[i])
    plt.axis('off')
    plt.title(getcode(y_train[i]))  # write title above every image in the output screen 


# In[ ]:


# repeat smae steps in test data 
x_test = []
y_test = []
for folder in os.listdir(testpath + 'sef_test'):
    files = gb.glob(pathname = str(testpath + 'seg_test//' + folder + '/*.jpg'))
    for file in files : 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        x_test.append(list(image_array))
        y_test.append(code[folder])
print(f'we have {len(x_test)} items in x_test') # we have 3000 items in x_test


# In[ ]:


#show image tests 
plt.figure(figsize = (20 ,20 ))
for n , i in enumerate (list(np.random.randint (0,len(x_test) , 36))):
    plt.subplot(6, 6,n+1)
    plt.imshow(x_test[i])
    plt.axis('off')
    plt.title(getcode(y_test[i]))


# In[ ]:


# prediction data_without having title 
# classify (predict) images according to types
x_pred = [] # no Y_pred
files = gb.glob(pathname = str(predpath + 'seg_pred/*.jpg'))
for file in files:
    image = cv2.imread(file)
    image_array=cv2.resize(image , (s,s))
    x_pred.append(list(image_array))
print(f'we have {len(x_pred)} items in x_pred') 


# In[ ]:


plt.figure(figsize=(20 ,20 ))
for n , i in enumerate (list(np.random.randint(0 , len(x_pred) , 36))):
    plt.subplot(6,6,n+1)
    plt.imshow(x_pred[i])
    plt.axix('off')


# In[21]:


#covert data to arrays 
x_train = np.array(x_train)
x_test= np.array(x_test) 
x_pred_array = np.array(x_pred) 
x_train = np.array(y_train)
y_test = np.array(y_test)

print(f'x_train shape is {x_train.shape}')
print(f'x_test shape is {x_test.shape}')
print(f'x_pred shpae is {x_pred_array.shape}')
print(f'x_train shape is {x_train.shape}')
print(f'x_test shape is {x_test.shape}')


# In[ ]:


# Building Model of CNN   
kerasModel = keras.models.Sequential([
    keras.layers.Conv2D(200 , kernel_size = (3,3) , activation = 'relu' , input_shape = (s,s,3 )),
    keras.layers.Conv2D(150 , kernel_size= (3,3) , activation = 'relu ' , input_shape = (s,s,3)),
    keras.layers.maxpool2D(4,4) , 
    keras.layers.Conv2D(120 , kernel_size = (3,3) , activation = 'relu'),
    keras.layers.Conv2D(80 , kernel_size = (3,3) , activation = 'relu') ,
    keras.layers.Conv2D(50 , kernel_size =(3,3) ,activation = 'relu') , 
    keras.layers.maxpoolWd(4,4), 
    keras.layers.Flatten(), 
    keras.layers.Dense(120 , activation = 'relu'), 
    keras.layers.Dense(100 , activation = 'relu'), 
    keras.layers.Dense(50 , activation = 'relu') ,
    keras.layers.Dense(50 , activation = 'relu') ,
    keras.layers.Dropout(rate= 0.5) ,
    keras.layers.Dense(6 , activation = 'softmax')
])

# customize model 
kerasModel.compile(optimizer = 'adam' , loss='sparse_categorical_crossentropy' , metrics = ['accuracy'])


# In[ ]:


# print model summary 
print('Model Details are: ')
print(kerasModel.summary())


# In[ ]:


# train model 
epochs = 50 
thisModel = kerasModel.fit(x_train , y_train , epochs = epochs , batch_size = 64 , verbose = 1 ) 


# In[ ]:


#Test model
modelLoss , modelAccuracy = kerasModel.evaluate(x_test , y_test) 
print('Test loss is {}'.format(modelLoss))
print('test Accuracy is {}'.format(modelAccuracy))


# In[ ]:


#predict x_test 
y_pred = kerasModel.predict(x_test)
print('Prediction shape is {}'.format(y_pred.shape))


# In[ ]:


# x_predict images 
y_result = kerasModel.predict(x_pred_array)
print('prediction shape is {}'.format(y_result.shape))


# In[ ]:


# Show images with titles 
plt.figure(figsize = (20 , 20))
for n , i in enumerate(list(np.random.randint(0,len(x_pred) , 36))):
    plt.subplot(6 , 6 , n+1) 
    plt.imshow(x_pred[i])
    plt.axis('off')
    plt.title(getcode(np.argmax(y_result[i])))

