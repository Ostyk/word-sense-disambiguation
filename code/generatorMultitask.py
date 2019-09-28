import parsers
import utils
import tensorflow.keras as K
from collections import namedtuple
import numpy as np
from sklearn.utils import shuffle
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.utils.data_utils import Sequence
import pandas as pd
import os

def __len__(training_file_path, batch_size):
    return parsers.TrainingParser(training_file_path).count() // batch_size

def get(batch_size, resources_path, training_file_path, \
        antivocab, output_vocab, output_vocab2, output_vocab3, \
        PADDING_SIZE = 50, gold_file_path = None):
    """
    Batch procesing generator, yields a dict of sentences, candidates and labels if in training mode (determined if gold_file_path is specified)

    param batch_size:
    param training_file_path:
    param antivocab:
    param output_vocab: senses
    aram output_vocab2: wndomain
    aram output_vocab3: lex
    param gold_file_path:
    return: generator object
    """

    while True:
        batch = {"sentences" : [], "candidates" : [], "candidates_wndomain": [],  "candidates_lex": []}

        training_data_flow = parsers.TrainingParser(training_file_path )
        if gold_file_path:
            gold_data_flow = parsers.GoldParser(gold_file_path)
            batch.update({"labels" : []})
            batch.update({"wndomain_labels" : []})
            batch.update({"lex_labels" : []})
            mapping_file = pd.read_csv(os.path.join(resources_path, "mapping.csv"))


        for batch_count, sentence in enumerate(training_data_flow.parse(), start = 1):

            #training mode
            if gold_file_path:
                labels = gold_data_flow.parse()
                output = prepare_sentence(sentence, antivocab, output_vocab, output_vocab2, output_vocab3, mapping_file, labels)
                batch['sentences'].append(output['sentence'])

                #batch['candidates'].append(output['candidates'])
                #batch['candidates_wndomain'].append(output['candidates_wndomain'])
                #batch['candidates_lex'].append(output['candidates_lex'])

                batch['labels'].append(output['labels'])
                batch['wndomain_labels'].append(output['wndomain_labels'])
                batch['lex_labels'].append(output['lex_labels'])

            #evaulation mode
            else:
                output = prepare_sentence(sentence, antivocab, output_vocab, output_vocab2, output_vocab3)

                batch['sentences'].append(output['sentence'])
                batch['candidates'].append(output['candidates'])
                batch['candidates_wndomain'].append(output['candidates_wndomain'])
                batch['candidates_lex'].append(output['candidates_lex'])

            if int(batch_count)%int(batch_size)==0:

                for key in batch.keys():
                    batch[key] = apply_padding(batch, key, maxlen = PADDING_SIZE, value = 1)

                #TO DO:

                if gold_file_path:
                    x  = batch['sentences']
                    y = [np.expand_dims(batch['labels'], axis=-1),
                         np.expand_dims(batch['wndomain_labels'], axis=-1),
                         np.expand_dims(batch['lex_labels'], axis=-1)]
                    #x, y = shuffle(x, y)
                    yield x, y
                else:
                    yield batch['sentences'], batch['candidates'], batch['candidates_wndomain'], batch['candidates_lex']

                batch = {"sentences" : [], "candidates" : [], "candidates_wndomain": [],  "candidates_lex": []}
                if gold_file_path:
                    batch.update({"labels" : []})
                    batch.update({"wndomain_labels" : []})
                    batch.update({"lex_labels" : []})

        if batch_count>0:
            for key in batch.keys():
                    batch[key] = apply_padding(batch, key, maxlen = PADDING_SIZE, value = 1)
            batch_count = 0

            if gold_file_path:
                x  = batch['sentences']
                y = [np.expand_dims(batch['labels'], axis=-1),
                     np.expand_dims(batch['wndomain_labels'], axis=-1),
                     np.expand_dims(batch['lex_labels'], axis=-1)]
                #x, y = shuffle(x, y)
                yield x, y

            else:

                yield batch['sentences'], batch['candidates'], batch['candidates_wndomain'], batch['candidates_lex']

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
    if key in ['candidates', 'candidates_wndomain', 'candidates_lex']:
        for candidate in range(len(x)):
            x[candidate] =  x[candidate] + [[value]] * (maxlen-len(x[candidate]))
        return x
    else:
        return K.preprocessing.sequence.pad_sequences(x, truncating='pre', padding='post', maxlen=maxlen, value = value )


def prepare_sentence(sentence, antivocab, output_vocab, output_vocab2=None, output_vocab3=None, mapping_file=None, labels=None):
        """
        Prepares an output sentence consisting of the sentence itself along with labels and candidates

        param sentence:
        param antivocab:
        param output_vocab:
        param output_vocab2: optional
        param output_vocab3: optional
        param labels: optional
        param mapping_file: mapping csv optional

        return output: dict with keys: sentence, labels, candidates all list type objects
        """
        records = namedtuple("Training", "id_ lemma pos instance")

        output = {"sentence" : [], "labels" : [], "wndomain_labels" : [], "lex_labels" : [], "candidates": [],  "candidates_wndomain": [],  "candidates_lex": []}
        for entry in sentence:

            id_, lemma, pos, instance = entry

            output_word = utils.map_word_from_dict(lemma, pos, antivocab, output_vocab, instance)
            output_word2 = utils.map_word_from_dict(lemma, pos, antivocab, output_vocab2, instance)
            output_word3= utils.map_word_from_dict(lemma, pos, antivocab, output_vocab3, instance)

            output['sentence'].append(output_word)

            if id_ is None:
                output['labels'].append(output_word)
                output['wndomain_labels'].append(output_word2)
                output['lex_labels'].append(output_word3)
                candidates = [output_word]
                candidates2 = [output_word2]
                candidates3 = [output_word3]

            else:
                if labels is not None:
                    current_label = labels.__next__()
                    assert current_label.id_ == id_, "ID mismatch"

                    sense = current_label.senses[0]

                    wndomain = mapping_file[mapping_file.WordNet==sense]['wordnet_domains'].values[0]
                    wndomain = output_vocab2[wndomain] if wndomain in output_vocab2 else output_vocab2["<UNK>"]

                    lex = mapping_file[mapping_file.WordNet==sense]['lexicographer'].values[0]
                    lex = output_vocab3[lex] if lex in output_vocab3 else output_vocab3["<UNK>"]

                    sense = output_vocab[sense] if sense in output_vocab else output_vocab["<UNK>"]

                    output['labels'].append(sense)
                    output['wndomain_labels'].append(wndomain)
                    output['lex_labels'].append(lex)

                candidates = utils.candidate_synsets(lemma, pos)
                candidates = [utils.map_word_from_dict(c, "X", antivocab, output_vocab, instance=True) for c in candidates]
                candidates2 = [utils.map_word_from_dict(c, "X", antivocab, output_vocab2, instance=True) for c in candidates]
                candidates3 = [utils.map_word_from_dict(c, "X", antivocab, output_vocab3, instance=True) for c in candidates]

            output['candidates'].append(candidates)
            output['candidates_wndomain'].append(candidates2)
            output['candidates_lex'].append(candidates3)

        return output
