#!/usr/bin/env python
# coding: utf-8

# ### Import important Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
import keras
import os
from tqdm import tqdm
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator,array_to_img, img_to_array


# #### importing train and test dataset

# In[2]:


def load_healthy(healthy_path):
    healthy_files=np.array(os.listdir(healthy_path))
    healthy_labels=np.array(['Healthy']*len(healthy_files))
    
    healthy_images=[]
    for image in tqdm(healthy_files):
        image=cv2.imread(healthy_path+image)
        image=cv2.resize(image,dsize=(48,48))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        healthy_images.append(image)
        
    healthy_images=np.array(healthy_images)
    return healthy_images, healthy_labels

def load_type_1(type_1_path):
    type_1_files=np.array(os.listdir(type_1_path))
    type_1_labels=np.array(['Type 1 disease']*len(type_1_files))
    
    type_1_images=[]
    for image in tqdm(type_1_files):
        image=cv2.imread(type_1_path+image)
        image=cv2.resize(image,dsize=(48,48))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        type_1_images.append(image)
        
    type_1_images=np.array(type_1_images)
    return type_1_images,type_1_labels

def load_type_2(type_2_path):
    type_2_files=np.array(os.listdir(type_2_path))
    type_2_labels=np.array(['Type 2 disease']*len(type_2_files))
    
    type_2_images=[]
    for image in tqdm(type_2_files):
        image=cv2.imread(type_2_path+image)
        image=cv2.resize(image,dsize=(48,48))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        type_2_images.append(image)
        
    type_2_images=np.array(type_2_images)
    return type_2_images,type_2_labels   


# In[3]:


healthy_images,healthy_labels=load_healthy("D:/data/train/Healthy/")


# In[4]:


type_1_images,type_1_labels=load_type_1(r"D:\data\train\Type 1 disease/")


# In[5]:


type_2_images,type_2_labels=load_type_2(r"D:\data\train\Type 2 disease/")


# In[6]:


train_X=np.vstack([healthy_images,type_1_images,type_2_images])


# In[7]:


train_Y=np.hstack([healthy_labels,type_1_labels,type_2_labels])


# In[8]:


type(train_X)


# In[9]:


plt.imshow(train_X[0])


# In[10]:


train_X.shape


# In[11]:


train_Y.shape


# In[12]:


fig,axes=plt.subplots(nrows=2,ncols=7,figsize=(16,4))
indices=np.random.choice(len(train_X),14)
count=0

for i in range(2):
    for j in range(7):
        axes[i,j].set_title(train_Y[indices[count]])
        axes[i,j].imshow(train_X[indices[count]],cmap='gray')
        axes[i,j].get_xaxis().set_visible(False)
        axes[i,j].get_yaxis().set_visible(False)
        count+=1
plt.show()


# In[13]:


# importing test dataset
healthy_images_test,healthy_images_label=load_healthy(r"D:\data\test\healthy/")


# In[14]:


type_1_images_test,type_1_images_label=load_type_1(r"D:\data\test\Type 1 disease/")


# In[15]:


type_2_images_test,type_2_images_label = load_type_2(r"D:\data\test\Type 2 disease/")


# In[16]:


test_X=np.vstack([healthy_images_test,type_1_images_test,type_2_images_test])


# In[17]:


test_X.shape


# In[18]:


test_Y=np.hstack([healthy_images_label,type_1_images_label,type_2_images_label])


# In[19]:


train_data=train_X.reshape(train_X.shape[0],train_X.shape[1],train_X.shape[2],1)


# In[20]:


train_data.shape


# In[21]:


test_data=test_X.reshape(test_X.shape[0],test_X.shape[1],test_X.shape[2],1)


# In[22]:


test_data.shape


# #### Scaling the Data like minmaxscaler

# In[23]:


train_data = train_data/255
test_data = test_data/255


# #### CNN Model

# In[24]:


from sklearn.preprocessing import LabelEncoder
one_hot_encoder=LabelEncoder()
train_Y_hot=one_hot_encoder.fit_transform(train_Y)
test_Y_hot=one_hot_encoder.transform(test_Y)


# In[25]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,BatchNormalization,Dropout,Activation
model=Sequential()
model.add(Conv2D(64,(3,3),activation='relu',input_shape=(train_X.shape[1],train_X.shape[2],1)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='softmax'))


# In[26]:


model.summary()


# In[27]:


from livelossplot import PlotLossesKerasTF
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[28]:


# Train the model
res = model.fit(train_data, train_Y_hot,epochs = 100, validation_data = (test_data,test_Y_hot),shuffle=True,callbacks = [ PlotLossesKerasTF()])


# In[29]:


predictions=model.predict(test_X)
predictions


# In[30]:


predictions=one_hot_encoder.inverse_transform(predictions.astype(int))


# In[31]:


from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(test_Y.astype(str),predictions.astype(str))

classnames=['Healthy','Type 1 disease','Type 2 disease']
plt.figure(figsize=(10,8))
plt.title('confusion matrix')
sn.heatmap(cm,annot=True,xticklabels=classnames,yticklabels=classnames,fmt='d',cmap=plt.cm.Blues)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[32]:


print(classification_report(test_Y.astype(str),predictions.astype(str)))


# #### Transfer Learning using Mobilenet

# In[33]:


train_path=r"D:/data/train/"
test_path=r"D:\data\test/"


# In[34]:


print("Train set:")
print("-"*60)
num_healthy=len(os.listdir(os.path.join(train_path,'healthy')))
num_type_1=len(os.listdir(os.path.join(train_path,'Type 1 disease')))
num_type_2=len(os.listdir(os.path.join(train_path,'Type 2 disease')))
print(f"Healthy={num_healthy}")
print(f"Type_1_disease={num_type_1}")
print(f"Type_2_disease={num_type_2}")


# In[35]:


print('Test set:')
print('-'*60)
num_healthy=len(os.listdir(os.path.join(test_path,'healthy')))
num_type_1=len(os.listdir(os.path.join(test_path,'Type 1 disease')))
num_type_2=len(os.listdir(os.path.join(test_path,'Type 2 disease')))
print(f'Healthy={num_healthy}')
print(f'Type_1_disease={num_type_1}')
print(f'Type_2_disease={num_type_2}')


# In[36]:


healthy_dir=r"D:\data\test\healthy/"
healthy=os.listdir(r"D:\data\test\healthy/")

plt.figure(figsize=(15,5))

for i in range(9):
    plt.subplot(330+i+1)
    img=cv2.imread(os.path.join(healthy_dir,healthy[i]))
    plt.title('Healthy')
    plt.imshow(img,cmap='gray')
    plt.axis('off')
    
plt.tight_layout()


# In[37]:


type_1_dir=r"D:\data\test\Type 1 disease/"
Type_1_disease=os.listdir(r"D:\data\test\Type 1 disease/")

plt.figure(figsize=(15,5))
for i in range(9):
    plt.subplot(330+i+1)
    img=cv2.imread(os.path.join(type_1_dir,Type_1_disease[i]))
    plt.title('Type 1 disease')
    plt.imshow(img,cmap='gray')
    plt.axis('off')
plt.tight_layout()


# In[38]:


type_2_dir=r"D:\data\test\Type 2 disease/"
Type_2_disease=os.listdir(r"D:\data\test\Type 2 disease")

plt.figure(figsize=(15,5))
for i in range(9):
    plt.subplot(330+i+1)
    img=cv2.imread(os.path.join(type_2_dir,Type_2_disease[i]))
    plt.title('Type 2 disease')
    plt.imshow(img,cmap='gray')
    plt.axis('off')
plt.tight_layout()


# In[39]:


import glob

healthy_train=glob.glob(train_path+"/healthy/*.jpeg")
type_1_train=glob.glob(train_path+"/Type 1 disease/*.jpeg")
type_2_train=glob.glob(train_path+"/Type 2 disease/*.jpeg")


# In[40]:


data=pd.DataFrame(np.concatenate([[0]*len(healthy_train),[1]*len(type_1_train),[2]*len(type_2_train)]),columns=['Class'])


# In[41]:


data.head()


# In[42]:


plt.figure(figsize=(15,10))
sn.countplot(data['Class'],data=data,palette='rocket')
plt.title('Haealthy vs Type 1 diseae vs Type 2 disease')
plt.show()


# In[43]:


train_set=ImageDataGenerator(rotation_range=15,
                             rescale=1./255,
                              shear_range=0.2,
                            zoom_range=0.1,
                            width_shift_range=0.1,
                            height_shift_range=0.1)
train=train_set.flow_from_directory(train_path,shuffle=True,target_size=(48,48),class_mode='categorical',batch_size=8)


# In[44]:


train_images,train_labels=next(train)


# In[45]:


test_set=ImageDataGenerator(rotation_range=15,
                             rescale=1./255,
                              shear_range=0.2,
                            zoom_range=0.1,
                            width_shift_range=0.1,
                            height_shift_range=0.1)
test=test_set.flow_from_directory(test_path,shuffle=True,target_size=(48,48),class_mode='categorical',batch_size=8)


# In[46]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet


# In[47]:


basemodel=MobileNet(weights='imagenet',include_top=False,input_tensor=Input(shape=(train_X.shape[1],train_X.shape[2],3)))
basemodel.summary()


# In[48]:


for layers in basemodel.layers[:-15]:
    layers.trainable=False


# In[49]:


headmodel=basemodel.input
headmodel=GlobalAveragePooling2D()(headmodel)
headmodel=Flatten(name='flatten')(headmodel)

headmodel=Dense(256,activation='relu')(headmodel)
headmodel=BatchNormalization()(headmodel)
headmodel=Dropout(0.2)(headmodel)
headmodel=Dense(128,activation='relu')(headmodel)
headmodel=BatchNormalization()(headmodel)
headmodel=Dropout(0.2)(headmodel)
headmodel=Dense(64,activation='relu')(headmodel)
headmodel=BatchNormalization()(headmodel)
headmodel=Dropout(0.2)(headmodel)

headmodel=Dense(3,activation='softmax')(headmodel)

model=Model(inputs=basemodel.input,outputs=headmodel)


# In[50]:


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


# In[51]:


from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
early_stop=EarlyStopping(monitor='val_loss',patience=2)
lr=ReduceLROnPlateau(monotor='val_acc',factor=0.1,patience=2,min_delta=0.0001)
callback=[early_stop,lr]


# In[52]:


h1=model.fit_generator(train,epochs=10,steps_per_epoch=(251//8),validation_data=test,validation_steps=(66//8),callbacks=callback)


# In[53]:


score=model.evaluate_generator(train)

for idx,metric in enumerate(model.metrics_names):
    print('{}:{}'.format(metric,score[idx]))
    


# In[54]:


score=model.evaluate_generator(test)

for idx,metric in enumerate(model.metrics_names):
    print('{}:{}'.format(metric,score[idx]))


# In[55]:


plt.figure(figsize=(12,8))
plt.title('Evaluation of model Mobilenet')

plt.subplot(2,2,1)
plt.plot(h1.history['loss'],label='loss')
plt.plot(h1.history['val_loss'],label='val_loss')
plt.legend()
plt.title('Loss evaluation')

plt.subplot(2,2,2)
plt.plot(h1.history['accuracy'],label='accuracy')
plt.plot(h1.history['val_accuracy'],label='val_accuracy')
plt.legend()
plt.title('Accuracy evaluation')

plt.show()


# In[56]:


y_pred=model.predict(test)
y_pred


# In[57]:


y_pred=np.argmax(y_pred,axis=1)
y_pred


# In[58]:


classes=['healthy','Type 1 disease','Type 2 disease']


# In[59]:


def predict_image(img):
    plt.figure(figsize=(40,8))
    print()
    print('-----------------Lung infection Detection-----------------')
    print()
    print('-----------------Result-------------------------')
    print()
    x=img_to_array(img)
    x=x/255
    plt.imshow(img)
    x=np.expand_dims(img,axis=0)
    
    print(classes[np.argmax(model.predict(x))])


# #### Transfer Learning using Densenet121

# In[60]:


train_gen=ImageDataGenerator(rotation_range=15,
                             rescale=1./255,
                              shear_range=0.2,
                            zoom_range=0.1,
                            width_shift_range=0.1,
                            height_shift_range=0.1)
train_set=train_gen.flow_from_directory(train_path,target_size=(224,224),shuffle=True,class_mode='categorical',batch_size=32)


# In[61]:


train_images,train_labels=next(train_set)


# In[62]:


train_images.shape


# In[63]:


labels={0:'healthy',1:'Type 1 disease',2:'Type 2 disease '}


# In[64]:


l=5
w=5

fig,axes=plt.subplots(l,w,figsize=(12,12))
axes=axes.ravel()

for i in np.arange(0,l*w):
    axes[i].imshow(train_images[i])
    axes[i].set_title(labels[np.argmax(train_labels[i])])
    axes[i].axis('off')
    
plt.subplots_adjust(wspace=0.5)


# In[65]:


from tensorflow.keras.applications import DenseNet121
basemodel=DenseNet121(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))


# In[66]:


basemodel.summary()


# In[67]:


for layers in basemodel.layers[:-10]:
    layers.trainable=False


# In[68]:


X=basemodel.output
X=GlobalAveragePooling2D()(X)
X=Flatten(name='flatten')(X)
X=Dense(256,activation='relu')(X)
X=BatchNormalization()(X)
X=Dropout(0.2)(X)

X=Dense(128,activation='relu')(X)
X=Dropout(0.2)(X)
X=Dense(64,activation='relu')(X)
X=BatchNormalization()(X)
X=Dropout(0.2)(X)
X=Dense(3,activation='softmax')(X)
model=Model(inputs=basemodel.inputs,outputs=X)


# In[69]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')


# In[70]:


train_gen=ImageDataGenerator(rotation_range=15,
                             rescale=1./255,
                              shear_range=0.2,
                            zoom_range=0.1,
                            width_shift_range=0.1,
                            height_shift_range=0.1)
train_set=train_gen.flow_from_directory(train_path,target_size=(224,224),shuffle=True,class_mode='categorical',batch_size=8)


# In[71]:


test_gen=ImageDataGenerator(rotation_range=15,
                             rescale=1./255,
                              shear_range=0.2,
                            zoom_range=0.1,
                            width_shift_range=0.1,
                            height_shift_range=0.1)
test_set=test_gen.flow_from_directory(test_path,target_size=(224,224),shuffle=True,class_mode='categorical',batch_size=8)


# In[72]:


from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
early_stop=EarlyStopping(monitor='val_loss',patience=2)
lr=ReduceLROnPlateau(monotor='val_acc',factor=0.1,patience=2,min_delta=0.0001)
callback=[early_stop,lr]


# In[73]:


h2=model.fit_generator(train_set,steps_per_epoch=train_set.n//8,epochs=10,validation_data=test_set,validation_steps=test_set.n//8,callbacks=callback)


# In[74]:


score=model.evaluate_generator(train_set)

for idx,metric in enumerate(model.metrics_names):
    print('{}:{}'.format(metric,score[idx]))


# In[75]:


score=model.evaluate(test_set)

for idx,metric in enumerate(model.metrics_names):
    print('{}:{}'.format(metric,score[idx]))


# In[76]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)

plt.plot(h2.history['loss'],label='loss')
plt.plot(h2.history['val_loss'],label='val_loss')
plt.title('Loss Evaluation')
plt.legend()

plt.subplot(2,2,2)
plt.plot(h2.history['accuracy'],label='accuracy')
plt.plot(h2.history['val_accuracy'],label='accuracy')
plt.title('Accuracy Evaluation')
plt.legend()

plt.tight_layout()


# * Densenet121 Model is more accurate  than Mobilenet

# In[77]:


basemodel.trainable=True


# In[78]:


print('Number of layers in the basemodel:', len(basemodel.layers))


# In[79]:


fine_tune_at=100

for layers in basemodel.layers[:fine_tune_at]:
    layers.trainable=False


# In[80]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


# In[81]:


fine_tune_epochs=10
raw_epochs=10
total_epoch=fine_tune_epochs+raw_epochs

h2=model.fit(train_set,epochs=total_epoch,
            validation_data=test_set,
            steps_per_epoch=100,
            batch_size=32,callbacks=callback)


# In[ ]:




