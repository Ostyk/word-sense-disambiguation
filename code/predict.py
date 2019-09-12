import re
import os
import json
import os
import copy

import pandas as pd
import numpy as np
import tensorflow.keras as K

from collections import namedtuple
from tqdm import tqdm, tnrange
from nltk.corpus import wordnet as wn

import utils
import parsers
import models
import generatorPrototype
import generatorMultitask

def predict_babelnet(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    main_predict_multitask(input_path, output_path, resources_path, prediction_type='babelnet')



def predict_wordnet_domains(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    main_predict_multitask(input_path, output_path, resources_path, prediction_type='wordnet_domains')


def predict_lexicographer(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    main_predict_multitask(input_path, output_path, resources_path, prediction_type='lexicographer')

    
def eval_parser(path, batch_size = 64):
    
    data_flow = parsers.TrainingParser(path)
    sentence_batch = []
    for batch_count, sentence in enumerate(data_flow.parse(), start = 1):
        sentence_batch.append(sentence)

        if len(sentence_batch)==(batch_size):#==0:
            yield sentence_batch
            sentence_batch = []

    if len(sentence_batch)>0:
        yield sentence_batch


def basic_predict(batch_ground_truth_sentences,
                  batch_model_predictions, 
                  candidate_synsets, 
                  PADDING_SIZE,
                  reverse_output_vocab):
    """
    Peforms predictions on a batch for the basic model
    :param batch_ground_truth_sentences:
    :param batch_model_predictions:
    :param candidate_synsets candidate_synsets:
    :param PADDING_SIZE:
    :param reverse_output_vocab:
    return predicted: batch of tuple(Sentence_id, WordNet)
    """
    
    outputs = []
    output = namedtuple("output", "Sentence_id WordNet success")
    
    for idx_sentence, sentence in enumerate(batch_model_predictions):

        ground_truth_sentence = batch_ground_truth_sentences[idx_sentence]
        
        for idx, entry in enumerate(ground_truth_sentence):
    
            if entry.instance == True: #only for instances not wf
                if idx<PADDING_SIZE: 
                    #WSD argmax
                    word_prob = sentence[idx]
                    current_candidate_synsets = candidate_synsets[idx_sentence][idx]
                    prob_dist_candidate_synset = word_prob[current_candidate_synsets]
                    current_synset = np.argmax(prob_dist_candidate_synset)

                    if current_synset>4: #change after deleting start stop
                        item = output(Sentence_id = entry.id_, WordNet = reverse_output_vocab[current_synset], success = True)
                        outputs.append(item)
                    else: #fallback
                        word = entry.lemma
                        item = output(Sentence_id = entry.id_, WordNet = models.MFS.retrieve_item(word), success = False)
                        outputs.append(item)
                else: #predict truncated
                    word = entry.lemma
                    item = output(Sentence_id = entry.id_, WordNet = models.MFS.retrieve_item(word), success = False)
                    outputs.append(item)
                
    return outputs

def main_predict(input_path, output_path, resources_path, prediction_type, batch_size=64, PADDING_SIZE=30):
    """
    :param input_path:
    :param output_path:
    :param resources_path:
    :param prediction_type: either babelnet, wordnet_domains, or lexicographer
    :param batch_size: depends on the model
    :param PADDING_SIZE: depends on the model
    :return: None
    """
    K.backend.clear_session()

    # ##################
    # # vocab loading #
    # #################
    mapping = pd.read_csv(os.path.join(resources_path, "mapping.csv"))
    
    senses = utils.json_vocab_reader(os.path.join(resources_path, 'semcor.vocab.WordNet.json'))
    inputs, antivocab = utils.json_vocab_reader(os.path.join(resources_path, 'semcor.input.vocab.json'),
                                                os.path.join(resources_path, 'semcor.leftout.vocab.json'))
    
    output_vocab = utils.merge_vocabulary(senses, inputs)
    reverse_output_vocab =  dict((v, k) for k, v in output_vocab.items())
    
    # ##################
    # # Model loading #
    # #################
    model_path, model_weight_path = sorted([os.path.join(resources_path,
                                        os.path.join('models/best_model', i)) for i in os.listdir(
                                        os.path.join(resources_path, 'models/best_model')) if i.startswith("model")])
    
    model_path = os.path.join(resources_path, 'models/model_2019-09-09_21:26:42_+0000.h5')
    model_weight_path = os.path.join(resources_path, 'models/model_weights_2019-09-09_21:26:42_+0000.h5')
    
    loaded_model = K.models.load_model(model_path)
    loaded_model.load_weights(model_weight_path)
 
    # ####################
    # # predicting phase #
    # ###################
    
    eval_generator = generatorPrototype.get(batch_size = batch_size,
                                training_file_path = input_path,
                                antivocab = antivocab,
                                output_vocab = output_vocab,
                                PADDING_SIZE = PADDING_SIZE)
    
    real_words = eval_parser(path = input_path, batch_size = batch_size)
    
    #print("starting to  write to:\t{}".format(output_path))
    with open(output_path, mode="a") as out:
        for batch_ground_truth_sentences in tqdm(real_words, desc='batch: '):

            batch_x, candidate_synsets = next(eval_generator)

            batch_model_predictions = loaded_model.predict_on_batch(batch_x)

            batch_outputs = basic_predict(batch_ground_truth_sentences,
                                          batch_model_predictions,
                                          candidate_synsets,
                                          PADDING_SIZE,
                                          reverse_output_vocab)

            for line in batch_outputs:
                #retreive appropriate mapping
                pred = mapping[mapping.WordNet==line.WordNet][prediction_type].values
                assert len(pred)==1, "error in mapping {}" .format(line.WordNet)
                
                fmt = "{} {} \n".format(line.Sentence_id, pred[0])
                
                out.write(fmt)
    print("done writing to:\t{}".format(output_path))
    
    
def MFS_predict_writer(input_path, output_path, resources_path, prediction_type, batch_size=64):
    """
    :param input_path:
    :param output_path:
    :param resources_path:
    :param batch_size: depends on the model
    :param prediction_type: either babelnet, wordnet_domains, or lexicographer
    :return: None
    """
    mapping = pd.read_csv(os.path.join(resources_path, "mapping.csv"))
    real_words = eval_parser(path = input_path, batch_size = batch_size)
    
    def MFS_predict(batch_ground_truth_sentences):
        """
        Peforms MFS predictions on a batch for 
        :param batch_ground_truth_sentences:
        return predicted: batch of tuple(Sentence_id, WordNet)
        """

        outputs = []
        output = namedtuple("output", "Sentence_id WordNet")

        for idx_sentence, sentence in enumerate(batch_ground_truth_sentences):
            for idx, entry in enumerate(sentence):
                if entry.instance == True: #only for instances not wf
                    word = entry.lemma
                    item = output(Sentence_id = entry.id_, WordNet = models.MFS.retrieve_item(word))
                    outputs.append(item)

        return outputs
    
    with open(output_path, mode="a") as out:
        for batch_ground_truth_sentences in tqdm(real_words, desc='batch: '):

            batch_outputs = MFS_predict(batch_ground_truth_sentences)
            
            for line in batch_outputs:
                #retreive appropriate mapping
                pred = mapping[mapping.WordNet==line.WordNet][prediction_type].values
                assert len(pred)==1, "error in mapping {}" .format(line.WordNet)
                
                fmt = "{} {} \n".format(line.Sentence_id, pred[0])
                
                out.write(fmt)
    print("done writing to:\t{}".format(output_path))
    
    
def main_predict_multitask(input_path, output_path, resources_path, prediction_type, batch_size=64, PADDING_SIZE=30):
    """
    :param input_path:
    :param output_path:
    :param resources_path:
    :param prediction_type: either babelnet, wordnet_domains, or lexicographer
    :param batch_size: depends on the model
    :param PADDING_SIZE: depends on the model
    :return: None
    """

    K.backend.clear_session()

    # ##################
    # # vocab loading #
    # #################
    mapping = pd.read_csv(os.path.join(resources_path, "mapping.csv"))

    senses = utils.json_vocab_reader(os.path.join(resources_path, 'semcor.vocab.WordNet.json'))
    wordnet_domains_vocabulary = utils.json_vocab_reader(os.path.join(resources_path, 'semcor.vocab.WordNetDomain.json'))
    lexicographer_vocabulary = utils.json_vocab_reader(os.path.join(resources_path, 'semcor.vocab.LexNames.json'))

    inputs, antivocab = utils.json_vocab_reader(os.path.join(resources_path, 'semcor.input.vocab.json'),
                                                os.path.join(resources_path, 'semcor.leftout.vocab.json'))

    output_vocab = utils.merge_vocabulary(senses, inputs)
    output_vocab2 = utils.merge_vocabulary(wordnet_domains_vocabulary, inputs)
    output_vocab3 = utils.merge_vocabulary(lexicographer_vocabulary, inputs)

    reverse_output1_vocab =  dict((v, k) for k, v in output_vocab.items())
    reverse_output2_vocab =  dict((v, k) for k, v in output_vocab2.items())
    reverse_output3_vocab =  dict((v, k) for k, v in output_vocab3.items())
    
    # ##################
    # # Model loading #
    # #################
#     model_path, model_weight_path = sorted([os.path.join(resources_path,
#                                         os.path.join('models/best_model', i)) for i in os.listdir(
#                                         os.path.join(resources_path, 'models/best_model')) if i.startswith("model")])

#     model_path = os.path.join(resources_path, 'models/model_2019-09-12_14:24:25_+0200.h5')
#     model_weight_path = os.path.join(resources_path, 'models/model_weights_2019-09-12_14:24:25_+0200.h5')
    model_path = os.path.join(resources_path, 'models/best_model/model_2019-09-12_12:13:22_+0000.h5')
    model_weight_path = os.path.join(resources_path, 'models/best_model/model_weights_2019-09-12_12:13:22_+0000.h5')

    loaded_model = K.models.load_model(model_path)
    loaded_model.load_weights(model_weight_path)
    
    ### generator
    eval_generator = generatorMultitask.get(batch_size = 64,
                                        resources_path = resources_path,
                                        training_file_path = input_path,
                                        antivocab = antivocab,
                                        output_vocab = output_vocab,
                                        output_vocab2 = output_vocab2,
                                        output_vocab3 = output_vocab3,
                                        PADDING_SIZE = PADDING_SIZE)

    real_words = eval_parser(path = input_path, batch_size = batch_size)
    
    
    with open(output_path, mode="a") as out:
        for batch_ground_truth_sentences in tqdm(real_words, desc='batch: '):

            batch_x, candidate_synsets_wordnet, candidates_wndomain, candidates_lex = next(eval_generator)

            batch_model_predictions = loaded_model.predict_on_batch(batch_x)

            if prediction_type =='babelnet': #actually using babelnet
                batch_model_predictions = batch_model_predictions[0]
                reverse_output_vocab = reverse_output1_vocab 
                candidate_synsets = candidate_synsets_wordnet

            elif prediction_type =='wordnet_domains': 
                batch_model_predictions = batch_model_predictions[1]
                reverse_output_vocab = reverse_output2_vocab
                candidate_synsets = candidates_wndomain

            elif prediction_type =='lexicographer': 
                batch_model_predictions = batch_model_predictions[2] 
                reverse_output_vocab = reverse_output3_vocab
                candidate_synsets = candidates_lex


            batch_outputs = basic_predict(batch_ground_truth_sentences,
                                          batch_model_predictions,
                                          candidate_synsets,
                                          PADDING_SIZE,
                                          reverse_output_vocab)
            for line in batch_outputs:
                if not line.success:
                    pred = mapping[mapping.WordNet==line.WordNet][prediction_type].values
                    assert len(pred)==1, "error in mapping {}" .format(line.WordNet)
                    fmt = "{} {} \n".format(line.Sentence_id, pred[0])
                else:
                    fmt = "{} {} \n".format(line.Sentence_id, line.WordNet) #not really wordnet in this case
                out.write(fmt)
    print("done writing to:\t{}".format(output_path))