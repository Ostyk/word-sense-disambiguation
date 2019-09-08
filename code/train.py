#!/usr/bin/env python
# coding: utf-8

# In[1]:


training_file_path = '../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml'
gold_file_path =  '../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt'

training_file_path_dev = '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2013/semeval2013.data.xml'
gold_file_path_dev = '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2013/semeval2013.gold.key.txt'
fine_senses_vocab_path = '../resources/semcor.vocab.WordNet.json'
input_vocab_path = '../resources/semcor.input.vocab.json'
input_antivocab_path = '../resources/semcor.leftout.vocab.json'
embedding_size = 32
batch_size = 64
LEARNING_RATE = 0.01
N_EPOCHS = 10
PADDING_SIZE = 50
print_model = False


# In[2]:


import models
import utils
import generators
import generatorsCopy

from tensorflow.random import set_random_seed
set_random_seed(42)
import tensorflow.keras as K
#import keras as K
import time
import os
#from tqdm import tqdm
import numpy as np
import time
import os


# In[3]:


#loading dict
senses = utils.json_vocab_reader(fine_senses_vocab_path)
inputs, antivocab = utils.json_vocab_reader(input_vocab_path, input_antivocab_path)
output_vocab = utils.vocab_merge(senses, inputs)
reverse_output_vocab =  dict((v, k) for k, v in output_vocab.items())

K.backend.clear_session()


# In[11]:


BasicModelNetwork = models.Basic(vocab_size = len(output_vocab),
                                embedding_size = embedding_size,
                                hidden_size = 32,
                                PADDING_SIZE = PADDING_SIZE,
                                LEARNING_RATE = LEARNING_RATE,
                                INPUT_DROPOUT = 0.2,
                                LSTM_DROPOUT = 0.45,
                                RECURRENT_DROPOUT = 0.35,
                                N_EPOCHS = N_EPOCHS)

if print_model is True:
    BasicModelNetwork.summary()


# In[20]:


train_generator = generatorsCopy.get(batch_size = 64,
                                training_file_path = training_file_path,
                                gold_file_path = gold_file_path,
                                antivocab = antivocab,
                                output_vocab = output_vocab,
                                PADDING_SIZE = PADDING_SIZE)

validation_generator = generatorsCopy.get(batch_size = 64,
                                         training_file_path = training_file_path_dev,
                                         gold_file_path = gold_file_path_dev,
                                         antivocab = antivocab,
                                         output_vocab = output_vocab,
                                         PADDING_SIZE = PADDING_SIZE)


# In[21]:


if not os.path.exists('../resources/logging'):
    os.mkdir('../resources/logging')
model_name = time.strftime('%Y-%m-%d_%H:%M:%S_%z')
cbk = K.callbacks.TensorBoard('../resources/logging/keras_model_'+model_name)

early_stopping = K.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=1,
                              verbose=2, mode='auto')
csv_logger = K.callbacks.CSVLogger('../resources/logging/keras_model_'+model_name+'.log')
model_checkpoint = K.callbacks.ModelCheckpoint(filepath = '../resources/logging/keras_model_'+model_name+'.h5',
                                               monitor='val_precision',
                                               verbose=2,
                                               save_best_only=True,
                                               save_weights_only=True,
                                               mode='auto', period=1)


# In[22]:


train_len = generatorsCopy.__len__(training_file_path, batch_size)
val_len = generatorsCopy.__len__(training_file_path_dev, batch_size)


# In[23]:


train_len,val_len


# In[ ]:


BasicModelNetwork.fit_generator(train_generator, 
                                steps_per_epoch=train_len,
                                epochs=N_EPOCHS, 
                                verbose=1,
                                callbacks=[cbk, early_stopping],
                                validation_data=validation_generator,
                                validation_steps=val_len,
                                class_weight=None,
                                max_queue_size=10,
                                workers=-1, 
                                use_multiprocessing=False,
                                shuffle=False,
                                initial_epoch=0)


# In[ ]:


models.save_model(model = BasicModelNetwork, model_name = model_name)

