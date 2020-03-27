#!/usr/bin/env python
# coding: utf-8

# #### Convolutional Neural Networks :  Cats vs Dogs  (with GPU support)

# In[11]:


import torch
import torchvision
import numpy as np


# In[2]:


import os
import cv2
from tqdm import tqdm


# In[3]:


REBUILD_DATA = False
class dvc():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS :0 , DOGS :1}
    training_data = []
    catcount = 0
    dogcount = 0
    
    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for file in tqdm(os.listdir(label)):
                try:   #SOME IMAGES MAYBE CORRUPT
                    path = os.path.join(label,file)
                    #convert image to grayscale to keep it simple
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    #resize to 50*50
                    img = cv2.resize(img, (self.IMG_SIZE,self.IMG_SIZE))
                    self.training_data.append([np.array(img),np.eye(2)[self.LABELS[label]]])
                    #print(label,self.CATS)
                    if str(label) == str(self.CATS):
                        self.catcount += 1
                    elif str(label) == str(self.DOGS):    
                        self.dogcount += 1
                except Exception as e:
                    pass
        np.random.shuffle(self.training_data)
        np.save("training_data.npy",self.training_data)
        
        print("Cat class = ",self.catcount)
        print("Dog class = ",self.dogcount)
        
if REBUILD_DATA:
    dc = dvc()
    dc.make_training_data()
    
    
         
                
                
        


# In[4]:


#len(dc.training_data) 


# In[16]:


#training_data[0]


# In[4]:


training_data = np.load("training_data.npy",allow_pickle=True)


# #### Meow

# In[6]:


#import matplotlib.pyplot as plt
#plt.imshow(training_data[2][0],cmap="gray")
#plt.show()


# #### CNN Architecture 

# In[7]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class cnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,5)      # 1 image, 32 channels, 5*5 krnel default stride=1
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)
        
        x = torch.randn(50,50).view(-1,1,50,50)
        self.to_linear = None   #auxillary variable to calculate shape of output of conv+max_pool
        self.convs(x)        
        
        self.fc1 = nn.Linear(self.to_linear,512)
        self.fc2 = nn.Linear(512,2)
        
    def convs(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))
        if self.to_linear is None:
            self.to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    def forward(self,x):
        x = self.convs(x)
        x = x.view(-1,self.to_linear)   #flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x,dim=1)
        


# In[12]:


torch.cuda.is_available()


# In[9]:


torch.backends.cudnn.enabled


# In[13]:


## Do check for cuda device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU Process")
else:
    device = torch.device("cpu")
    print("CPU Process")


# In[14]:


net = cnet().to(device)   #transferiing class objet to gpu


# In[15]:


import torch.optim as optim
optimizer = optim.Adam(net.parameters(),lr=0.001)
loss_function = nn.MSELoss()


# In[16]:


#PREPARING TRAINING,TESTING

X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])


# In[17]:


val_ratio = 0.1
val_size = int(len(X)*val_ratio)
print(val_size)


# In[18]:


train_X = X[:-val_size]
train_y = y[:-val_size]
test_X = X[-val_size:]
test_y = y[-val_size:]


# In[19]:


print(len(train_X))


# In[20]:


def train(net):    
    BATCH_SIZE = 100
    EPOCHS = 10
    for epoch in range(EPOCHS):
        for i in tqdm(range(0,len(train_X),BATCH_SIZE)):
            batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
            batch_y = train_y[i:i+BATCH_SIZE]
            batch_X,batch_y = batch_X.to(device), batch_y.to(device)
            net.zero_grad()
            out = net(batch_X)
            loss = loss_function(out,batch_y)
            loss.backward()
            optimizer.step()
        print("Epoch = "+str(epoch)+", loss = "+str(loss))


# In[21]:


def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0]  # returns a list, 
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy: ", round(correct/total, 3))


# In[22]:


train(net)


# In[23]:


test(net)


# In[ ]:




