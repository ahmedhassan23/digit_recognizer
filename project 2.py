#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D

from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# In[8]:


from keras.utils.np_utils import to_categorical


# In[10]:


train=pd.read_csv("D:\\machine\\projects\\project 2\\train.csv")
test=pd.read_csv("D:\\machine\\projects\\project 2\\test.csv")


# In[11]:


train.head()


# In[20]:


cols=train.shape
x_train=train.drop(labels=["label"],axis=1)
y_train=train["label"]


# In[21]:


x_train.head()


# In[22]:


y_train.head()


# In[23]:


y_train.value_counts()


# In[24]:


x_train.isnull().any().describe()


# In[26]:


test.isnull().any().describe()


# In[27]:


x_train=x_train/255.0
test=test/255.0


# In[28]:


x_train=x_train.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)


# In[29]:


y_train=to_categorical(y_train,num_classes=10)


# In[30]:


x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1,random_state=2)


# In[35]:


g = plt.imshow(x_train[4][:,:,0])


# In[36]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# In[37]:


optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# In[38]:


model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[39]:


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[40]:


epochs = 25 
batch_size = 85


# In[42]:


datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.1,  
        width_shift_range=0.1, 
        height_shift_range=0.1,
        horizontal_flip=False
        vertical_flip=False)  


datagen.fit(x_train)


# In[ ]:


history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_val,y_val),
                              verbose = 2, steps_per_epoch=x_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])


# In[ ]:


fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

