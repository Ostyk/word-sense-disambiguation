from lxml import etree
from collections import namedtuple
from tqdm import tqdm, tnrange, tqdm_notebook
import json

import utils

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
            
    def create_vocab(self, input_vocab_path, pos_vocab_path):
        """
        Creates  two vocabularies: Input and POS
        :param input_vocab_path: path to save input_vocab
        :param pos_vocab_path:  path to save pos_vocab
        :return: None, JSON dumps to given paths
        """
        input_vocab, pos_vocab = {}, {}
        words_total, count = 0, 0

        for sentence in tqdm(self.parse()):
            count += 1

            for word in sentence:
                lemma = word.lemma
                pos = word.pos

                # handling OOV
                if pos in [".", "PUNCT"]: lemma = "<PUNCT>"
                elif pos == 'NUM': lemma = "<NUM>"
                elif pos == 'SYM': lemma = "<SYM>"

                input_vocab[lemma] =  input_vocab.get(lemma, 0) + 1
                pos_vocab[pos] = pos_vocab.get(pos, 0) + 1

                words_total += 1

            if count % 10_000 == 0:
                print("{} sentences parsed thus far, with {} tags".format(count, len(input_vocab)))
                break
                
        with open(input_vocab_path, 'w') as f:
            json.dump(input_vocab, f)
        
        with open(pos_vocab_path, 'w') as f:
            json.dump(pos_vocab, f)

            
            
if __name__ == '__main__':
    
    Training = TrainingParser('../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor+OMSTI/semcor+omsti.data.xml')
    
    Training.create_vocab(input_vocab_path = "../resources/extracted-data/semcor+omsti.input.vocab.txt",
                          pos_vocab_path = "../resources/extracted-data/semcor+omsti.pos.vocab.txt")
    pass