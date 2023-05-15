#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile


# In[2]:


f1 = zipfile.ZipFile("pet classification.zip")
f1.extractall()


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator,array_to_img, img_to_array
import os
import random
import seaborn as sns


# In[3]:


datagen = ImageDataGenerator(
rotation_range = 40,
width_shift_range=0.2,
height_shift_range=0.2,
rescale = 1/255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True,
fill_mode = 'nearest')


# In[4]:


datagen = ImageDataGenerator(
rotation_range = 40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True,
fill_mode = 'nearest')

img = load_img("D:/DEEP LEARNING/train/cats/9.jpg") # this is PIL image
type(img)
x = img_to_array(img)


# In[5]:


img = load_img("D:/DEEP LEARNING/train/cats/9.jpg")
img


# In[6]:


x.shape


# In[7]:


x = x.reshape((1,)+ x.shape) # here we are adding (num_of_image,height,width,channels) this format of image keras can read


# In[8]:


x.shape


# In[9]:


x.ndim


# In[10]:


i = 0

for batch in datagen.flow(x, batch_size =1,save_to_dir = 'Preview',save_prefix = 'cat',save_format='jpeg'):
    i += 1
    if i>20:
        break


# In[11]:


x = img_to_array(img)


# In[12]:


img = load_img("C:/Users/amit/Preview/dog_0_91.jpeg")
y = img_to_array(img)


# In[13]:


y.shape


# In[14]:


y.ndim


# In[14]:


from tensorflow.keras import backend as K


# In[15]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout,Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D


# In[16]:


model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (150,150,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[17]:


model.summary()


# In[18]:


# Compile the Model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[19]:


batch_size = 8
train_datagen = ImageDataGenerator(
rescale = 1/255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale= 1/255) 


train_generator = train_datagen.flow_from_directory('D:/DEEP LEARNING/train',target_size = (150,150),batch_size = batch_size,
                                                    class_mode = 'binary',shuffle = True)

validation_generator = test_datagen.flow_from_directory('D:/DEEP LEARNING/test',target_size = (150,150),batch_size = batch_size,
                                                    class_mode = 'binary',shuffle = True)
                         


# In[20]:


from livelossplot import PlotLossesKerasTF


# In[21]:


model.fit(train_generator, steps_per_epoch = 40// batch_size ,epochs = 100,
                 
          validation_data =validation_generator,
         
          validation_steps = 20// batch_size, callbacks = [ PlotLossesKerasTF()])


# In[41]:


#model.save('dog_cat_model.h5')


# #### Steps to Predict new image or Train model

# In[56]:


# Load the new image
original_new = load_img("C:/Users/amit/Downloads/tiger image.jpg")
original_new


# In[57]:


type(original_new)


# In[58]:


# Convert the Load Image in Desired Size
convert_image = load_img("C:/Users/amit/Downloads/tiger image.jpg", target_size = (150,150,3))
convert_image


# In[59]:


type(convert_image)


# In[60]:


# Convert image into array because computer understand array form of image
convert_image_array = img_to_array(convert_image)
type(convert_image_array)


# In[61]:


# Scaling the converted image to array
convert_image_array = convert_image_array/255


# In[62]:


convert_image_array.shape


# In[63]:


# Reshape convert image into (number of image,height,width,channels)
y = convert_image_array.reshape((1,) + convert_image_array.shape)


# In[64]:


y.shape


# In[65]:


# Predict the image with model
model.predict(y)


# In[ ]:




