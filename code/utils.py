from nltk.corpus import wordnet as wn
import os
import pandas as pd
from tqdm import tqdm

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
    return df

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


