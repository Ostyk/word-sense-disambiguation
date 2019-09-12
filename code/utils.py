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


def merge_vocabulary(vocab1, vocab2):
    """
    Merges two vocabularies
    :param vocab1:
    :param vocab2:
    :return: merged vocab
    """

    merged = copy.deepcopy(vocab1)
    for key2 in vocab2.keys():
        if key2 not in merged:
            merged[key2] = len(merged)
    return merged

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
    mapping['WordNet'] = mapping['sensekey1'].progress_apply(utils.WordNet.from_sensekey)

    mapping = mapping.drop(['sensekey2'],axis=1)

    BabelNet = pd.read_csv(babelnet2wordnet, sep = '\t', names = ['babelnet', 'WordNet'])
    WordNetDomain = pd.read_csv(babelnet2wndomains, sep = '\t', names = ['babelnet', 'wordnet_domains'])
    LexicographerNet = pd.read_csv(babelnet2lexnames, sep = '\t', names = ['babelnet', 'lexicographer'])

    df = mapping

    df = df.join(BabelNet.set_index('WordNet'), on='WordNet')

    df = df.join(WordNetDomain.set_index('babelnet'), on='babelnet')

    df = df.join(LexicographerNet.set_index('babelnet'), on='babelnet')

    print("NA is {:.1f}%".format((1-(df.dropna().shape[0] / df.shape[0]) )*100))
    df.wordnet_domains.fillna("factotum", inplace=True)
    df.lexicographer.fillna("misc", inplace=True)
    return df

def listdir_fullpath(d):
    '''returns a list of items in a directory with their full path'''
    return [os.path.join(d, f) for f in os.listdir(d)]

def map_word_from_dict(lemma, pos, antivocab, output_vocab, instance):
    """
    :param lemma:
    :param pos:
    :param antivocab:
    :param output_vocab:
    :param instance:
    :return: replaced word
    """
    lemma, pos = OOV_handeler(lemma, pos)

    mapped_word = None
    if lemma in antivocab:
        mapped_word = output_vocab["<REPLACEMENT>"]

    if instance or mapped_word is None:
        if lemma in output_vocab:
            mapped_word = output_vocab[lemma]
        elif mapped_word is None:
            mapped_word = output_vocab["<UNK>"]

    return mapped_word

def candidate_synsets(lemma, pos):
    """
    Used to restrict our attention only to synsets from the entire probability distribution over the output layer
    :param lemma:
    :param pos:
    :return: list(Candidate synsets) or lemma if nothing in Wordnet
    """
    pos_dict = {"ADJ": wn.ADJ, "ADV": wn.ADV, "NOUN": wn.NOUN, "VERB": wn.VERB}

    synsets = wn.synsets(lemma, pos=pos_dict[pos]) if pos in pos_dict else wn.synsets(lemma)
    if len(synsets) == 0:
        return [lemma]
    else:
        return [WordNet.from_synset(x) for x in synsets]

def OOV_handeler(lemma, pos):
    """
    Handles OOV words
    :param lemma: str
    :param pos: str
    :return: lemma, pos
    """
    if pos in [".", "?", ","]: lemma = "<PUNCT>"
    elif pos == 'NUM': lemma = "<NUM>"
    elif pos == 'SYM': lemma = "<SYM>"

    return lemma, pos
