import re
import os

from tqdm import tqdm, tnrange
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
    WSD performed via Most Frequent Sense (MFS) which always returns the predominant sense of a lemma
    """

    def retrieve_item(lemma):
        """
        Retreives the MFS of a lemma from the nltk package if exists else word
        """
        synsets = wn.synsets(str(lemma))

        if len(synsets) == 0:
            return lemma
        else:
            mfs = synsets[0]
            return utils.WordNet.from_synset(mfs)

    def predict(sentence):
        """
        param: sentence of lemmas
        return sentence MFS
        """
        return [MFS.retrieve_item(lemma) for lemma in sentence]



def Basic(vocab_size, embedding_size, hidden_size, PADDING_SIZE, LEARNING_RATE, INPUT_DROPOUT, LSTM_DROPOUT,RECURRENT_DROPOUT, N_EPOCHS):
    """
    Word Sense Disambiguiation performed via a basic WSD
    """
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
    Word Sense Disambiguiation performed via elaborated methods
    """
    def __init__(self):
        pass

def Multitask(vocab_size, embedding_size, hidden_size, PADDING_SIZE, LEARNING_RATE, INPUT_DROPOUT, LSTM_DROPOUT,RECURRENT_DROPOUT, N_EPOCHS):
    """
    Word Sense Disambiguiation performed via elaborated methods
    """
    print("Creating MultitaskKERAS model")
    inputs = K.layers.Input(shape=(PADDING_SIZE,))
    embeddings = K.layers.Embedding(vocab_size['senses'],
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

    predictions_1 = K.layers.TimeDistributed(K.layers.Dense(
        vocab_size['senses'], activation='softmax', name='senses'))(BI_LSTM)
    
    predictions_2 = K.layers.TimeDistributed(K.layers.Dense(
        vocab_size['wordnet_domains'], activation='softmax', name = 'wordnet_domains'))(BI_LSTM)
    
    predictions_3 = K.layers.TimeDistributed(K.layers.Dense(
        vocab_size['lexicographer'],activation='softmax', name = 'lexicographer'))(BI_LSTM)


    model = K.models.Model(inputs=[inputs], outputs=[predictions_1,
                                                     predictions_2,
                                                     predictions_3])

    model.compile(loss = "sparse_categorical_crossentropy",#,"sparse_categorical_crossentropy","sparse_categorical_crossentropy"],
                  optimizer = K.optimizers.Adam(lr=LEARNING_RATE, decay = 0.001/N_EPOCHS, amsgrad=False),
                  metrics = ['acc'])

    return model


def save_model(model, model_name):
    """
    param: model
    param: model_name
    return None:
    """
    rel_path = '../resources/models'
    if not os.path.exists(rel_path):
        os.mkdir(rel_path)

    weights = os.path.join(rel_path,'model_weights_'+model_name+'.h5')
    model_name_save = os.path.join(rel_path,'model_'+model_name+'.h5')

    print("saving weights to: {}".format(weights))
    model.save_weights(weights) #saving weights for further analysis

    print("saving model to: {}".format(model_name_save))
    model.save(model_name_save)
