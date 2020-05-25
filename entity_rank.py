#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt


# In[16]:


embeddings_dict = {}
words = {}
i = 0
glove_data = '../glove.6B.200d.txt'
f = open(glove_data, encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    words[i] = word
    vector = np.asarray(values[1:], "float32")
    embeddings_dict[word] = vector
    i += 1

print(len(words))
print(len(embeddings_dict))


# In[17]:


def find_closest_embeddings_euclidean(vector_map, embedding):
    return sorted(vector_map.keys(), key=lambda word: spatial.distance.euclidean(vector_map[word], embedding))

def find_closest_embeddings_cosine(vector_map, embedding):
    return sorted(vector_map.keys(), key=lambda word: spatial.distance.cosine(vector_map[word], embedding))

def find_euclidean_distance(vec1, vec2):
    return spatial.distance.euclidean(vec1, vec2)

def find_cosine_distance(vec1, vec2):
    return spatial.distance.cosine(vec1, vec2)

def get_avg_sum_embedding(line, dictionary):
    sum_of_vector = np.full((200, ), 0)
    word_vec = line.split()
    for word in word_vec:
        embedding = dictionary.get(word, np.full((200, ), 0))
        sum_of_vector = np.add(embedding, sum_of_vector)
    sum_of_vector = np.divide(sum_of_vector, len(word_vec)) 
    return sum_of_vector


# In[18]:


attr_map = 'Trane-Demos/three_datasets/flight-delay/FlightDelay.mapping'
attr_description_map = {}
attr_vector_map = {}
f = open(attr_map, encoding="utf8")
for line in f:
    values = line.split('|')
    word = values[0]
    attr_description_map[word] = values[1].split('\n')[0]

for key, val in attr_description_map.items():
    attr_vector_map[key] = get_avg_sum_embedding(val, embeddings_dict)


# In[19]:


human_query_dict = {}
query_file_loc = 'human_query/query.txt'
f = open(query_file_loc, encoding="utf8")
writer = open('output.txt', 'w')
sum_of_human_query_vector = np.full((200, ), 0)
for line in f:
    writer.write(line + "\n")
    query = line.split("|")[0]
    sum_of_human_query_vector = get_avg_sum_embedding(query, embeddings_dict)
    human_query_dict[query] = sum_of_human_query_vector
    rank_euclidian = find_closest_embeddings_euclidean(attr_vector_map, sum_of_human_query_vector)
    rank_cosine = find_closest_embeddings_cosine(attr_vector_map, sum_of_human_query_vector)
    writer.write("Euclidian rank -> " + str(rank_euclidian) + "\n\n")
    writer.write("Cosine rank -> " + str(rank_cosine) + "\n\n")


# In[ ]:




