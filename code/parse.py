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

    def create_vocab(self, input_vocab_path, pos_vocab_path, subsampling_rate=1e-4, min_count=5):
        """
        Creates  two vocabularies: Input and POS
        :param input_vocab_path: path to save input_vocab
        :param pos_vocab_path:  path to save pos_vocab
        :param subsampling_rate: subsampling rate of the vocab http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        :param min_count: occurence of word
        :return: None, JSON dumps to given paths
        """
        input_vocab, pos_vocab = {}, {}
        words_total, count = 0, 0
        not_subsampled = set()

        for sentence in tqdm(self.parse()):
            count += 1

            for word in sentence:
                lemma = word.lemma
                pos = word.pos

                if word.instance == True:
                    not_subsampled.add(lemma)
                # handling OOV
                if pos in [".", "PUNCT"]: lemma = "<PUNCT>"
                elif pos == 'NUM': lemma = "<NUM>"
                elif pos == 'SYM': lemma = "<SYM>"



                input_vocab[lemma] =  input_vocab.get(lemma, 0) + 1
                pos_vocab[pos] = pos_vocab.get(pos, 0) + 1

                words_total += 1

            if count % 10_000 == 0:
                print("{} sentences parsed thus far, with {} tags".format(count, len(input_vocab)))




        #sort input vocab
        input_vocab = dict(sorted(input_vocab.items(), key=lambda x: x[1], reverse=True))

        #Subsampling
        #http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        #filter it to not touch words that cannot be subsampled or by min_count
        input_vocab = {word: occurence for word, occurence in input_vocab.items()
                       if word in not_subsampled
                       or occurence >= min_count}

        #probabilty_of_keeping_word = lambda occurence: 1.0 - np.sqrt(subsampling_rate / occurence * words_total)
        def probabilty_of_keeping_word(occurence):
            fraction = occurence/words_total
            return ( np.sqrt( fraction / subsampling_rate ) + 1 ) * ( subsampling_rate / fraction )

        input_vocab_to_file = []
        left_out_vocab = set()

        for (word, occurence) in input_vocab.items():
            if word in not_subsampled or random.uniform(0, 1) >= probabilty_of_keeping_word(occurence):
                input_vocab_to_file.append(word)
            else:
                left_out_vocab.add(word)

        #new_vocab = [ (word) for (word, occurence) in filtered_vocab.items() if word in untouchable or uniform(0, 1) >= prob(occurence)]

        print("input vocab: {}\nPOS vocab:{}".format( len(input_vocab_to_file), len(pos_vocab) ) )
        with open(input_vocab_path, 'w') as f:
            json.dump(input_vocab_to_file, f)

        with open(pos_vocab_path, 'w') as f:
            json.dump(pos_vocab, f)



if __name__ == '__main__':

#     Training = TrainingParser('../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor+OMSTI/semcor+omsti.data.xml')

#     Training.create_vocab(input_vocab_path = "../resources/extracted-data/semcor+omsti.input.vocab.json",
#                           pos_vocab_path = "../resources/extracted-data/semcor+omsti.pos.vocab.json")
    pass
