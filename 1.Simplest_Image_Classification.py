#!/usr/bin/env python
# coding: utf-8

# ## 이미지 인식을 위해 "사전 훈련된 신경만 가져오기"

# In[1]:


from torchvision import models


# In[2]:


dir(models)


# ### 그 중 ResNet101은 어떻게 생겼나 살펴 보자

# In[5]:


resnet = models.resnet101(pretrained=True)


# In[6]:


resnet


# ## 기본적인 Preprocessing 함수: torchvision 모듈에서 transform 제공

# In[7]:


from torchvision import transforms


# In[10]:


preprocess = transforms.Compose([
    transforms.Resize(256), # 입력 이미지 크기를 256 X 256 으로 조정
    transforms.CenterCrop(224), # 중심으로부터 224 X 224 로 잘라냄
    transforms.ToTensor(), # Pytorch 다차원 배열인 텐서 형태로 전환 (이 예에서는 색, 높이, 너비의 3차원 배열인 텐서)
    transforms.Normalize( # 지정된 평균과 표준편차를 가지도록 RGB를 정규화. (훈련에 사용하는 이미지 형식과 일치하게 만들기 위해 이런 작업이 필요)
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])


# ## Pillow: 파이썬 이미지 조작 모듈

# #### 이미지 불러오기

# In[13]:


from PIL import Image
img = Image.open("./Pytorch/data/bobby.jpg")


# In[15]:


img # 새 창으로 뷰어를 띄우려면, img.show()


# ## 전처리 파이프라인으로 이미지 통과시키기

# In[16]:


img_t = preprocess(img)


# ## 전처리해서 만들어진 텐서를 신경망에 넣을 수 있도록 준비🔪

# In[17]:


import torch
batch_t = torch.unsqueeze(img_t, 0)


# ## ResNet으로 Inference 해보기

# In[19]:


resnet.eval() # 신경망을 eval 모드로 설정


# In[20]:


out = resnet(batch_t)


# In[21]:


out 


# In[26]:


out.size()


# 4,450만 개에 이르는 파라미터가 관련된 엄청난 연산이 방금 실행되어 1,000개의 스코어를 만들어냈고,
# 
# 점수 하나 하나는 이미지넷 클래스에 각각 대응된다. 시간도 얼마 걸리지 않았다.
# 
# 이제 점수가 가장 높은 클래스의 레이블만 찾으면 된다. 
# 
# 레이블은 모델이 이미지에서 무엇을 봤는지를 우리에게 알려준다.

# 예측한 순서대로 레이블 리스트를 만들기 위해서는, 
# 
# 출력과 동일한 순서로 나열된 레이블 텍스트 파일을 읽어놓은 후 
# 
# 점수가 높은 출력의 인덱스부터 동일한 인덱스에 있는 레이블을 리스트에 추가하면 된다. 
# 
# 이미지 인식 관련 모델 대부분은 여기서 보여주는 출력 형태와 유사하다.
# 

# #### 이미지넷 데이터 클래스에 대한 1,000개의 레이블이 담긴 파일을 읽어보자.

# In[23]:


with open('./Pytorch/data/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]


# In[24]:


labels


# In[28]:


len(labels)


# ### ResNet의 inference를 결과인 out 텐서에서, 가장 높은 score를 가진 index를 찾는다. 

# In[35]:


_, index = torch.max(out, dim=1) # dim = 1 ➡️  , dim = 0 🔽


# In[41]:


index


# In[51]:


percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100


# In[53]:


labels[index[0]],  percentage[index[0]].item()


# index는 파이썬 숫자가 아닌, 1개 요소를 가지는 1차원 tensor([207])이기 때문에, 
# 
# 인덱스로 사용하기를 원하는 실제 정수값을 얻기 위해, index[0]을 사용할 필요가 있음.

# 모델은 여러 점수를 출력하므로,
# 
# 두 번째, 세 번째 등의 결과가 무엇인지도 알 수 있다.

# In[58]:


_, indices = torch.sort(out, descending=True)
indices


# In[59]:


[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[ ]:




