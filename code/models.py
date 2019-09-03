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
    
class Basic(object):
    """
    Word Sense Disambiguiation performed via a basic sequence tagging
    """
    def __init__(self, test):
        self.test = test
    
    def build(self, vocab_size, embedding_size, hidden_size, PADDING_SIZE, LEARNING_RATE, INPUT_DROPOUT, LSTM_DROPOUT,RECURRENT_DROPOUT, N_EPOCHS):
        
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
    
    def train(self):
        pass
    
    
    @staticmethod
    def apply_padding(output, key, maxlen=50, value=1):
        """
        Applies padding to output sequences

        param output: dict
        param key: key of dict
        param maxlen: length to pad 
        param value: pad with this value 
        return padded list of lists
        """
        x = output[key]
        if key == 'candidates':
            for candidate in range(len(x)):
                x[candidate] =  x[candidate] + [[value]] * (maxlen-len(x[candidate]))
            return x
        else:
            return K.preprocessing.sequence.pad_sequences(x, truncating='pre', padding='post', maxlen=maxlen, value = value )

        
        
    def prepare_sentence_batch(self, batch_size, training_file_path, antivocab, output_vocab, PADDING_SIZE = 50, gold_file_path = None):
        """
        Batch procesing generator, yields a dict of sentences, candidates and labels if in training mode (determined if gold_file_path is specified)

        param batch_size: 
        param training_file_path: 
        param antivocab: 
        param output_vocab: 
        param gold_file_path: 
        return: generator object
        """
        batch = {"sentences" : [], "candidates" : []}

        training_data_flow = parsers.TrainingParser(training_file_path)
        if gold_file_path:
            gold_data_flow = parsers.GoldParser(gold_file_path)
            batch.update({"labels" : []})


        for batch_count, sentence in enumerate(training_data_flow.parse(), start = 1):
            #training mode
            if gold_file_path:
                labels = gold_data_flow.parse()         
                output = self.prepare_sentence(sentence, antivocab, output_vocab, labels)

                batch['sentences'].append(output['sentence'])
                batch['candidates'].append(output['candidates'])
                batch['labels'].append(output['labels'])

            #evaulation mode
            else:
                output = self.prepare_sentence(sentence, antivocab, output_vocab)

                batch['sentences'].append(output['sentence'])
                batch['candidates'].append(output['candidates'])

            if int(batch_count)%int(batch_size)==0:

                for key in batch.keys():
                    batch[key] = self.apply_padding(batch, key, maxlen = PADDING_SIZE, value = 1)
                batch_count = 0
                
                yield batch['sentences'], np.expand_dims(batch['labels'], axis=-1)
                
                batch = {"sentences" : [], "candidates" : []}
                if gold_file_path:
                    batch.update({"labels" : []})

    
            
    @staticmethod
    def prepare_sentence(sentence, antivocab, output_vocab, labels=None):
        """
        Prepares an output sentence consisting of the sentence itself along with labels and candidates

        param sentence: 
        param antivocab: 
        param output_vocab: 
        param labels: 

        return output: dict with keys: sentence, labels, candidates all list type objects
        """
        records = namedtuple("Training", "id_ lemma pos instance")

        output = {"sentence" : [], "labels" : [], "candidates": []}
        for entry in sentence:

            id_, lemma, pos, _ = entry

            output_word = utils.replacement_routine(lemma, entry, antivocab, output_vocab)        
            output['sentence'].append(output_word)

            if id_ is None:
                output['labels'].append(output_word)
                candidates = [output_word]

            else:
                if labels is not None:
                    current_label = labels.__next__()
                    assert current_label.id_ == id_, "ID mismatch"

                    sense = current_label.senses[0]
                    sense = output_vocab[sense] if sense in output_vocab else output_vocab["<UNK>"]
                    output['labels'].append(sense)
                candidates = utils.candidate_synsets(lemma, pos)
                candidates = [utils.replacement_routine(c, records(id_=None, lemma=c, pos="X", instance=True), antivocab, output_vocab) for c in candidates]

            output['candidates'].append(candidates)
        return output

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
class MultiTask(object):
    """
    Word Sense Disambiguiation performed via elaborated basic sequence tagging
    """
    def __init__(self):
        pass
