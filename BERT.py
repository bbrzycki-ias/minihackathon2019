#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


data = pd.read_csv('/Users/dnissani/Desktop/Test_SageMaker/bert_drug_train.csv')


# In[ ]:


inputs = []
labels = []
for el in data.values:
    if el[0] == 'low':
        labels.append(torch.tensor([0]).unsqueeze(0))
    elif el[0] == 'moderate':
        labels.append(torch.tensor([1]).unsqueeze(0))
    else:
        labels.append(torch.tensor([2]).unsqueeze(0))
    
    text = el[1].split(' ' and '\t' and '\n' and '/')
    text = ' '.join(text)[:512]
    inputs.append(torch.tensor(tokenizer.encode(text)).unsqueeze(0))


# In[ ]:


tokenizer = pytorch_transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# In[ ]:


model = pytorch_transformers.BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states = True)
model.eval()


# In[ ]:


text_model = pytorch_transformers.BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels = 3)


# In[ ]:


input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)


# In[ ]:


input_ids


# In[ ]:


labels = torch.tensor([1]).unsqueeze(0)


# In[ ]:


labels


# In[ ]:


outputs = text_model(input_ids)


# In[ ]:


outputs


# In[ ]:


loss, logits = outputs


# In[ ]:


loss


# In[ ]:


logits


# In[ ]:


data.head()


# In[ ]:


data.values[0,1]


# In[ ]:


inputs[:10]


# In[ ]:


output = []
for el, label in zip(inputs, labels):
    output.append(text_model(el, label))


# In[ ]:


output[4]


# In[ ]:




