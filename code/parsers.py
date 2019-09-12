from lxml import etree
from collections import namedtuple
from tqdm import tqdm, tnrange, tqdm_notebook
import json
import random
import utils
import numpy as np

class TrainingParser(object):
    """
    Class to parse the XML training file
    """
    def __init__(self, xml_file):
        self.xml_file = xml_file
        self.records = namedtuple("Training", "id_ lemma pos instance")

    def count(self):
        """
        Parses only for the sake of counting the number of sentences
        return sum: of the sentences in the xml file
        """
        return sum(1 for _ in etree.iterparse(self.xml_file, tag="sentence"))

    def parse(self):
        """
        Parses the archived training XML file in Raganato's format
        :param archived_xml: path to XML file
        :return: sentence generator
        """
        for event, sentence in etree.iterparse(self.xml_file, tag="sentence"):
            to_generator = []
            if event == 'end':
                for elem in sentence:
                    item = self.records(id_ = elem.get("id") if elem.tag == 'instance' else None,
                                        lemma = elem.get("lemma"),
                                        pos = elem.get("pos"),
                                        instance = True if elem.tag == 'instance' else False)
                    to_generator.append(item)
                yield to_generator
            sentence.clear()

    def create_vocab(self, input_vocab_path, pos_vocab_path, left_out_vocab_path, subsampling_rate=1e-3, min_count=5):
        """
        Creates  two vocabularies: Input and POS
        :param input_vocab_path: path to save input_vocab
        :param pos_vocab_path:  path to save pos_vocab
        :param left_out_vocab_path: path to save left out words
        :param subsampling_rate: subsampling rate of the vocab http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        :param min_count: occurence of word
        :return: None, JSON dumps to given paths
        """
        input_vocab, pos_vocab = {}, {}
        words_total, count = 0, 0
        not_subsampled = set()
        subsampling = True if min_count!=0 else False

        for sentence in tqdm(self.parse()):
            count += 1

            for word in sentence:
                words_total += 1

                lemma, pos = word.lemma, word.pos

                if word.instance == True:
                    not_subsampled.add(lemma)
                # handling OOV
                lemma, pos = utils.OOV_handeler(lemma, pos)

                input_vocab[lemma] =  input_vocab.get(lemma, 0) + 1
                pos_vocab[pos] = pos_vocab.get(pos, 0) + 1

        print("{} sentences parsed with {} words".format(count, len(input_vocab)))
        #print(not_subsampled)
        #print(not_subsampled)

        #sort input vocab
        input_vocab = dict(sorted(input_vocab.items(), key=lambda x: x[1], reverse=True))
        #Subsampling
        input_vocab_to_file = []
        left_out_vocab = set()

        for (word, occurence) in input_vocab.items():
            if word in not_subsampled or occurence >= min_count:
                prob = utils.probabilty_of_keeping_word(occurence, words_total, subsampling_rate, type_='mikolov2013') if subsampling else 0
                if word in not_subsampled or random.uniform(0, 1) >= prob:
                    input_vocab_to_file.append(word)
                else:
                    left_out_vocab.add(word)

        print("subsampled input vocab: {}\nPOS vocab:{}\nleft out vocab:{}".format(len(input_vocab_to_file),
                                                                                   len(pos_vocab),
                                                                                   len(left_out_vocab)))
        with open(left_out_vocab_path, 'w') as f:
            json.dump(list(left_out_vocab), f)

        with open(input_vocab_path, 'w') as f:
            json.dump(input_vocab_to_file, f)

        pos_vocab = sorted(pos_vocab.items(), key=lambda k: k[1], reverse=True)
        pos_vocab = [i[0] for i in pos_vocab]
        with open(pos_vocab_path, 'w') as f:
            json.dump(pos_vocab, f)



class GoldParser(object):
    """
    Class to parse the gold.txt file
    """
    def __init__(self, file):
        self.file = open(file, 'r')
        self.records = namedtuple("Gold", "id_ senses")

    def parse(self):
        """
        Parses the archived gold text file
        :return: sentence generator
        """

        for line in self.file:
            line = line.strip()
            line = line.split(" ")
            item = self.records(id_ = line[0],
                                senses = list(set(utils.WordNet.from_sensekey(i) for i in line[1:])))

            yield item
