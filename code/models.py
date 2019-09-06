import re
import os

from tqdm import tqdm, tnrange, tqdm_notebook
import json
import pandas as pd
import os
from collections import namedtuple

import utils
import parsers
from nltk.corpus import wordnet as wn
import copy
import numpy as np

import tensorflow.keras as K

class MFS(object):
    """
    Word Sense Disambiguiation performed via Most Frequent Sense tagging
    """
    def __init__(self):
        pass
    
# class Basic(object):
#     """
#     Word Sense Disambiguiation performed via a basic sequence tagging
#     """
#     def __init__(self, test):
#         self.test = test
    
#     def build(self, vocab_size, embedding_size, hidden_size, PADDING_SIZE, LEARNING_RATE, INPUT_DROPOUT, LSTM_DROPOUT,RECURRENT_DROPOUT, N_EPOCHS):
        
#         print("Creating KERAS model")
#         inputs = K.layers.Input(shape=(PADDING_SIZE,))
#         embeddings = K.layers.Embedding(vocab_size,
#                                         embedding_size,
#                                         mask_zero=True,
#                                         name = 'embedding')(inputs)


#         BI_LSTM = (K.layers.Bidirectional(
#                    K.layers.LSTM(hidden_size, dropout = LSTM_DROPOUT,
#                                  recurrent_dropout = RECURRENT_DROPOUT,
#                                  return_sequences=True,
#                                  kernel_regularizer=K.regularizers.l2(0.01),
#                                  activity_regularizer=K.regularizers.l1(0.01)
#                                 ), name = 'Bi-directional_LSTM'))(embeddings)

#         predictions = K.layers.TimeDistributed(K.layers.Dense(vocab_size, activation='softmax'))(BI_LSTM)

#         model = K.models.Model(inputs=[inputs], outputs=predictions)

#         model.compile(loss = 'sparse_categorical_crossentropy',
#                       optimizer = K.optimizers.Adam(lr=LEARNING_RATE, decay = 0.001/N_EPOCHS, amsgrad=False),
#                       metrics = ['acc'])

#         return model

def Basic(vocab_size, embedding_size, hidden_size, PADDING_SIZE, LEARNING_RATE, INPUT_DROPOUT, LSTM_DROPOUT,RECURRENT_DROPOUT, N_EPOCHS):
    print("Creating KERAS model")
    inputs = K.layers.Input(shape=(PADDING_SIZE,))
    embeddings = K.layers.Embedding(vocab_size,
                                    embedding_size,
                                    mask_zero=True,
                                    name = 'embedding')(inputs)


    BI_LSTM = (K.layers.Bidirectional(
               K.layers.LSTM(hidden_size, dropout = LSTM_DROPOUT,
                             recurrent_dropout = RECURRENT_DROPOUT,
                             return_sequences=True,
                             kernel_regularizer=K.regularizers.l2(0.01),
                             activity_regularizer=K.regularizers.l1(0.01)
                            ), name = 'Bi-directional_LSTM'))(embeddings)

    predictions = K.layers.TimeDistributed(K.layers.Dense(vocab_size, activation='softmax'))(BI_LSTM)

    model = K.models.Model(inputs=[inputs], outputs=predictions)

    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = K.optimizers.Adam(lr=LEARNING_RATE, decay = 0.001/N_EPOCHS, amsgrad=False),
                  metrics = ['acc'])

    return model
    
    
class MultiTask(object):
    """
    Word Sense Disambiguiation performed via elaborated basic sequence tagging
    """
    def __init__(self):
        pass
