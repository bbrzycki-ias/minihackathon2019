#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pytorch_transformers
import torch
import torchvision
import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score as precision
from sklearn.metrics import accuracy_score as accuracy
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import csv
import io
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import spacy 
from spacy_langdetect import LanguageDetector
import time


# In[9]:


config = pytorch_transformers.BertConfig.from_pretrained('bert-base-multilingual-cased')


# In[14]:


config


# In[10]:


tokenizer = pytorch_transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# In[15]:


tokenizer


# In[13]:


model = pytorch_transformers.BertForSequenceClassification(config)


# In[16]:


model


# In[25]:


input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)


# In[26]:


input_ids


# In[27]:


labels = torch.tensor([1]).unsqueeze(0)


# In[28]:


labels


# In[29]:


outputs = model(input_ids, labels=labels)


# In[30]:


outputs


# In[34]:


loss, logits = outputs


# In[35]:


loss


# In[36]:


logits


# In[44]:


data = pd.read_csv('/Users/dnissani/Desktop/Test_SageMaker/bert_drug_train.csv')


# In[45]:


data.head()


# In[46]:


len(data)


# In[47]:


inputs = []
labels = []
for el in data.values:
    if el[0] == 'low':
        labels.append(torch.tensor([0]).unsqueeze(0))
    elif el[0] == 'moderate':
        labels.append(torch.tensor([1]).unsqueeze(0))
    else:
        labels.append(torch.tensor([2]).unsqueeze(0))
        
    inputs.append(torch.tensor(tokenizer.encode(el[1])).unsqueeze(0))


# In[ ]:




