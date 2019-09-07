from nltk.corpus import wordnet as wn
import os
import pandas as pd
from tqdm import tqdm
import json
from numpy import sqrt
import copy
import parsers

def eval_parser(path, batch_size):

    data_flow = parsers.TrainingParser(path)
    sentence_batch = []
    for batch_count, sentence in enumerate(data_flow.parse(), start = 1):
        sentence_batch.append(sentence)

        if int(batch_count)%int(batch_size)==0:
            yield sentence_batch
            sentence_batch = []
            
class WordNet(object):
    """
    Retrieves a WordNet using the nltk package
    """
    def from_sensekey(sensekey: str):
        """
        param senskey:
        """
        synset = wn.lemma_from_key(sensekey).synset()
        return WordNet.from_synset(synset)

    def from_synset(synset):
        """
        param synset: dtype :nltk.corpus.reader.wordnet.Synset
        """
        return "wn:" + str(synset.offset()).zfill(8) + synset.pos()


def json_vocab_reader(path, leftout_path=None):
    """
    Reads a json format vocabulary
    :param path: vocabulary file path
    :param leftout_path: path to leftout words
    :return: vocab dict with special symbols
    """
    vocab_start = {"<UNK>": 0, "<PAD>": 1, "<START>": 2, "<STOP>": 3, "<REPLACEMENT>": 4}
    with open(path, 'r') as file:
        vocab_list = json.load(file)

    vocab = {x: index for index, x in enumerate(vocab_list, start=len(vocab_start))}
    vocab = {**vocab_start, **vocab}

    if leftout_path!=None:
        with open(leftout_path, 'r') as file:
            leftout = json.load(file)
        return vocab, leftout

    else:
        return vocab


def probabilty_of_keeping_word(occurence, words_total, subsampling_rate, type_='mikolov2013'):
    """
    taken from https://nathanrooy.github.io/posts/2018-03-22/word2vec-from-scratch-with-python-and-numpy/
    :param occurence: occurence count of a single word
    :param type_: either mikolov2013 or word2vec official C code
    :param words_total: total number of words
    :param subsampling_rate: subsampling rate given
    :return probabilty of keeping that word
    """
    if type_ == 'mikolov2013':
        return 1.0 - sqrt(subsampling_rate / occurence * words_total)
    elif type_ == 'word2vec':
        return (sqrt(occurence/(subsampling_rate*words_total)) + 1) * (subsampling_rate * words_total / fraction)


def sensekeyToSynsetConverter(sensekey: str):
    """
    Retrieves a WordNet synset from a sensekey using the nltk package'''
    :param sensekey
    :return: wordnet_id
    """
    synset = wn.lemma_from_key(sensekey).synset()

    wordnet_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
    return wordnet_id


def parse_evaluation(gold_file, babelnet2wordnet, babelnet2wndomains, babelnet2lexnames):
    """
    Converts a gold.txt file into corresponding BabelNet, WordNetDomain, and LEX codes
    :param gold_file
    :param babelnet2wordnet: path
    :param babelnet2wndomains: path
    :param babelnet2lexnames: path
    :return: wordnet_id
    """

    mapping = pd.read_table(gold_file, sep = ' ', names = ['sentence_idx', 'sensekey1', 'sensekey2'])
    tqdm.pandas(desc="Sensekey to Wordnet 1")
    mapping['WordNet'] = mapping['sensekey1'].progress_apply(sensekeyToSynsetConverter)

    mapping = mapping.drop(['sensekey2'],axis=1)

    BabelNet = pd.read_table(babelnet2wordnet, sep = '\t', names = ['BabelNet', 'WordNet'])
    WordNetDomain = pd.read_table(babelnet2wndomains, sep = '\t', names = ['BabelNet', 'WordNetDomain'])
    LexicographerNet = pd.read_table(babelnet2lexnames, sep = '\t', names = ['BabelNet', 'LexNames'])

    df = mapping

    df = df.join(BabelNet.set_index('WordNet'), on='WordNet')

    df = df.join(WordNetDomain.set_index('BabelNet'), on='BabelNet')

    df = df.join(LexicographerNet.set_index('BabelNet'), on='BabelNet')

    print("NA is {:.1f}%".format((1-(df.dropna().shape[0] / df.shape[0]) )*100))
    df.WordNetDomain.fillna("factotum", inplace=True)
    df.LexNames.fillna("misc", inplace=True)
    return df

def listdir_fullpath(d):
    '''returns a list of items in a directory with their full path'''
    return [os.path.join(d, f) for f in os.listdir(d)]



#############################################################################################################################
#############################################################################################################################
#####################################      TO REDO THESE FUCNTIONS BELOW                    #################################
#############################################################################################################################
#############################################################################################################################
def vocab_merge(vocab1, vocab2):
    """
    Merges two vocabularies into the first one, keeping the reverse vocabulary consistent.
    :param vocab1: First vocabulary (will contain the merged vocabulary), as Dict str -> int
    :param rev_vocab1: First reverse vocabulary, as List of str
    :param vocab2: Second vocabulary, as Dict str -> int
    :return: (vocab1, rev_vocab1) updated to resemble the merged vocabulary
    """

    v1 = copy.deepcopy(vocab1)

    for key2 in vocab2.keys():
        if key2 not in v1:
            v1[key2] = len(v1)

    return v1

def candidate_synsets(lemma, pos):
    """
    Retrieves the candidate synsets for the given lemma and pos combination.
    :param lemma: Lemma to get the synsets of
    :param pos: POS associated to the lemma
    :return: Candidate synsets having the given lemma and POS, as List; the lemma itself in case there is no match in WordNet
    """

    pos_dictionary = {"ADJ": wn.ADJ, "ADV": wn.ADV, "NOUN": wn.NOUN, "VERB": wn.VERB}   # open classes only

    if pos in [".", "?", ","]: return  "<PUNCT>"
    elif pos == 'NUM': return "<NUM>"
    elif pos == 'SYM': return "<SYM>"

    elif pos in pos_dictionary:
        synsets = wn.synsets(lemma, pos=pos_dictionary[pos])
    else:
        synsets = wn.synsets(lemma)

    if len(synsets) == 0:
        return [lemma]
    return [WordNet.from_synset(syn) for syn in synsets]

def replacement_routine(lemma, pos, antivocab, output_vocab, instance):
    lemma, pos = OOV_handeler(lemma, pos)
    ret_word = None
    if lemma in antivocab:
        ret_word = output_vocab["<REPLACEMENT>"]

    if instance or ret_word is None:
        if lemma in output_vocab:
            ret_word = output_vocab[lemma]
        elif ret_word is None:
            ret_word = output_vocab["<UNK>"]

    return ret_word


def OOV_handeler(lemma, pos):
    if pos in [".", "?", ","]: lemma = "<PUNCT>"
    elif pos == 'NUM': lemma = "<NUM>"
    elif pos == 'SYM': lemma = "<SYM>"

    return lemma, pos
