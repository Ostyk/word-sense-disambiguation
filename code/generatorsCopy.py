import parsers
import utils
import tensorflow.keras as K
from collections import namedtuple
import numpy as np
from sklearn.utils import shuffle
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.utils.data_utils import Sequence
# class Basic(object):
#     """
#     Generator for Word Sense Disambiguiation performed via a basic sequence tagging
#     """
#     def __init__(self, batch_size, training_file_path, antivocab, output_vocab, PADDING_SIZE = 50, gold_file_path = None):

#         self.batch_size =  batch_size
#         self.training_file_path =  training_file_path
#         self.antivocab =  antivocab
#         self.output_vocab =  output_vocab
#         self.PADDING_SIZE =  PADDING_SIZE
#         self.gold_file_path  =  gold_file_path
#         self.length = 0

#     def __len__(self):
#         return parsers.TrainingParser(self.training_file_path).count() // self.batch_size


#     def __getitem__(self):
#         return self.one_batch(self.batch_size, self.training_file_path, self.antivocab, self.output_vocab, self.PADDING_SIZE, self.gold_file_path)

#     @staticmethod
#     def one_batch(batch_size, training_file_path, antivocab, output_vocab, PADDING_SIZE, gold_file_path):
#         """
#         Batch procesing generator, yields a dict of sentences, candidates and labels if in training mode (determined if gold_file_path is specified)

#         param batch_size:
#         param training_file_path:
#         param antivocab:
#         param output_vocab:
#         param gold_file_path:
#         return: generator object
#         """
#         batch = {"sentences" : [], "candidates" : []}

#         training_data_flow = parsers.TrainingParser(training_file_path )
#         if gold_file_path:
#             gold_data_flow = parsers.GoldParser(gold_file_path)
#             batch.update({"labels" : []})


#         for batch_count, sentence in enumerate(training_data_flow.parse(), start = 1):
#             #length += 1
#             #training mode
#             if gold_file_path:
#                 labels = gold_data_flow.parse()
#                 output = Basic.prepare_sentence(sentence, antivocab, output_vocab, labels)

#                 batch['sentences'].append(output['sentence'])
#                 batch['candidates'].append(output['candidates'])
#                 batch['labels'].append(output['labels'])

#             #evaulation mode
#             else:
#                 output = Basic.prepare_sentence(sentence, antivocab, output_vocab)

#                 batch['sentences'].append(output['sentence'])
#                 batch['candidates'].append(output['candidates'])

#             if int(batch_count)%int(batch_size)==0:

#                 for key in batch.keys():
#                     batch[key] = Basic.apply_padding(batch, key, maxlen = PADDING_SIZE, value = 1)


#                 batch_count = 0

#                 if gold_file_path:
#                     x, y = batch['sentences'], np.expand_dims(batch['labels'], axis=-1)
#                     yield (x, y)
#                 else:
#                     yield batch['sentences']
#                 batch = {"sentences" : [], "candidates" : []}
#                 if gold_file_path:
#                     batch.update({"labels" : []})

#         if batch_count>0:
#             for key in batch.keys():
#                     batch[key] = Basic.apply_padding(batch, key, maxlen = PADDING_SIZE, value = 1)
#             batch_count = 0

#             if gold_file_path:
#                 x, y = batch['sentences'], np.expand_dims(batch['labels'], axis=-1)
#                 yield (x, y)
#             else:
#                 yield shuffle(batch['sentences'])


#     @staticmethod
#     def apply_padding(output, key, maxlen=50, value=1):
#         """
#         Applies padding to output sequences

#         param output: dict
#         param key: key of dict
#         param maxlen: length to pad
#         param value: pad with this value
#         return padded list of lists
#         """
#         x = output[key]
#         if key == 'candidates':
#             for candidate in range(len(x)):
#                 x[candidate] =  x[candidate] + [[value]] * (maxlen-len(x[candidate]))
#             return x
#         else:
#             return K.preprocessing.sequence.pad_sequences(x, truncating='pre', padding='post', maxlen=maxlen, value = value )


#     @staticmethod
#     def prepare_sentence(sentence, antivocab, output_vocab, labels=None):
#         """
#         Prepares an output sentence consisting of the sentence itself along with labels and candidates

#         param sentence:
#         param antivocab:
#         param output_vocab:
#         param labels:

#         return output: dict with keys: sentence, labels, candidates all list type objects
#         """
#         records = namedtuple("Training", "id_ lemma pos instance")

#         output = {"sentence" : [], "labels" : [], "candidates": []}
#         for entry in sentence:

#             id_, lemma, pos, instance = entry

#             output_word = utils.replacement_routine(lemma, pos, antivocab, output_vocab, instance)
#             output['sentence'].append(output_word)

#             if id_ is None:
#                 output['labels'].append(output_word)
#                 candidates = [output_word]

#             else:
#                 if labels is not None:
#                     current_label = labels.__next__()
#                     assert current_label.id_ == id_, "ID mismatch"

#                     sense = current_label.senses[0]
#                     sense = output_vocab[sense] if sense in output_vocab else output_vocab["<UNK>"]
#                     output['labels'].append(sense)
#                 candidates = utils.candidate_synsets(lemma, pos)
#                 candidates = [utils.replacement_routine(c, "X", antivocab, output_vocab, instance=True) for c in candidates]

#             output['candidates'].append(candidates)
#         return output


# class Basic(object):
#     """
#     Word Sense Disambiguiation performed via a basic sequence tagging
#     """
#     def __init__(self, batch_size, training_file_path, antivocab, output_vocab, PADDING_SIZE = 50, gold_file_path = None):

#         self.batch_size =  batch_size
#         self.training_file_path =  training_file_path
#         self.antivocab =  antivocab
#         self.output_vocab =  output_vocab
#         self.PADDING_SIZE =  PADDING_SIZE
#         self.gold_file_path  =  gold_file_path
#         self.length = 0
        
#     def __len__(self):
#         return parsers.TrainingParser(self.training_file_path).count()

def __len__(training_file_path, batch_size):
    return parsers.TrainingParser(training_file_path).count() // batch_size

def get(batch_size, training_file_path, antivocab, output_vocab, PADDING_SIZE = 50, gold_file_path = None):
    """
    Batch procesing generator, yields a dict of sentences, candidates and labels if in training mode (determined if gold_file_path is specified)

    param batch_size:
    param training_file_path:
    param antivocab:
    param output_vocab:
    param gold_file_path:
    return: generator object
    """
    while True:
        batch = {"sentences" : [], "candidates" : []}

        training_data_flow = parsers.TrainingParser(training_file_path )
        if gold_file_path:
            gold_data_flow = parsers.GoldParser(gold_file_path)
            batch.update({"labels" : []})


        for batch_count, sentence in enumerate(training_data_flow.parse(), start = 1):
            #training mode
            if gold_file_path:
                labels = gold_data_flow.parse()
                output = prepare_sentence(sentence, antivocab, output_vocab, labels)

                batch['sentences'].append(output['sentence'])
                batch['candidates'].append(output['candidates'])
                batch['labels'].append(output['labels'])

            #evaulation mode
            else:
                output = prepare_sentence(sentence, antivocab, output_vocab)

                batch['sentences'].append(output['sentence'])
                batch['candidates'].append(output['candidates'])

            if int(batch_count)%int(batch_size)==0:

                for key in batch.keys():
                    batch[key] = apply_padding(batch, key, maxlen = PADDING_SIZE, value = 1)


                batch_count = 0

                if gold_file_path:
                    x, y = batch['sentences'], np.expand_dims(batch['labels'], axis=-1)
                    x, y = shuffle(x, y)
                    yield x, y
                else:
                    yield shuffle(batch['sentences'])
                batch = {"sentences" : [], "candidates" : []}
                if gold_file_path:
                    batch.update({"labels" : []})

        if batch_count>0:
            for key in batch.keys():
                    batch[key] = apply_padding(batch, key, maxlen = PADDING_SIZE, value = 1)
            batch_count = 0

            if gold_file_path:
                x, y = batch['sentences'], np.expand_dims(batch['labels'], axis=-1)
                x, y = shuffle(x, y)
                yield x, y
            else:
                yield shuffle(batch['sentences'])

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

            id_, lemma, pos, instance = entry

            output_word = utils.replacement_routine(lemma, pos, antivocab, output_vocab, instance)
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
                candidates = [utils.replacement_routine(c, "X", antivocab, output_vocab, instance=True) for c in candidates]

            output['candidates'].append(candidates)
        return output