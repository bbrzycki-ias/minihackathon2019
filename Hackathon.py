#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pytorch_transformers
import torch
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
import time


# In[8]:


data = pd.read_csv('/home/ubuntu/Bert_Data/bert_drug_train.csv')


# In[9]:


tokenizer = pytorch_transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# In[10]:


pytorch_transformers.BertConfig.from_pretrained('bert-base-multilingual-cased')


# In[11]:


model = pytorch_transformers.BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states = True)
model.eval()


# In[12]:


text_model = pytorch_transformers.BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels = 3)


# In[13]:


input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)


# In[14]:


input_ids


# In[15]:


labels = torch.tensor([1]).unsqueeze(0)


# In[16]:


outputs = text_model(input_ids, labels = labels)


# In[17]:


outputs


# In[19]:


inputs = []
labels = []
for el in data.values:
    if el[0] == 'low':
        labels.append(torch.tensor([0]).unsqueeze(0))
    elif el[0] == 'moderate':
        labels.append(torch.tensor([1]).unsqueeze(0))
    else:
        labels.append(torch.tensor([2]).unsqueeze(0))
    
    text = tokenizer.tokenize(el[1])
    text = ' '.join(text)[:512]
    inputs.append(torch.tensor(tokenizer.encode(text)).unsqueeze(0))


# In[21]:



    for el, label in zip(inputs, labels):


# In[22]:


output


# In[ ]:




