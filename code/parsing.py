#!/usr/bin/env python
# coding: utf-8

# In[4]:


from tqdm import tqdm, tnrange
import json
import pandas as pd
import os

import utils
import parsers


# In[ ]:





# # Parse training XML file

# In[5]:


Training = parsers.TrainingParser('../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml')

Training.create_vocab(input_vocab_path = "../resources/semcor.input.vocab.json",
                      pos_vocab_path = "../resources/semcor.pos.vocab.json",
                      left_out_vocab_path = "../resources/semcor.leftout.vocab.json",
                      subsampling_rate=1e-4,
                      min_count=5)


# # converting eval datasets

# In[6]:


dir_ = "../resources/WSD_Evaluation_Framework/Evaluation_Datasets"
eval_datasets = [i for i in os.listdir(dir_) if i.startswith("se")]
eval_datasets


# In[7]:


for name in eval_datasets:
    print("Dataset: {}".format(name))
    
    path = os.path.join(dir_, name)
    gold_file = [i for i in os.listdir(path) if i.endswith('gold.key.txt')][0]
    gold_file = os.path.join(path, gold_file)
    print("using {}".format(gold_file))

    df = utils.parse_evaluation(gold_file = gold_file,
                                babelnet2wordnet = '../resources/babelnet2wordnet.tsv',
                                babelnet2wndomains = '../resources/babelnet2wndomains.tsv',
                                babelnet2lexnames = '../resources/babelnet2lexnames.tsv')
    base = gold_file.split(".gold.key.txt")[0]

    df[['sentence_idx', 'babelnet']].to_csv(base+".gold.babelnet.txt", header=None, index=None, sep=' ')
    df[['sentence_idx', 'wordnet_domains']].to_csv(base+".gold.wordnet_domains.txt", header=None, index=None, sep=' ')
    df[['sentence_idx', 'lexicographer']].to_csv(base+".gold.lexicographer.txt", header=None, index=None, sep=' ')
    
    


# # Gold output vocab (training file semcor)

# In[ ]:


# df = utils.parse_evaluation(gold_file = "../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt",
#                             babelnet2wordnet = '../resources/babelnet2wordnet.tsv',
#                             babelnet2wndomains = '../resources/babelnet2wndomains.tsv',
#                             babelnet2lexnames = '../resources/babelnet2lexnames.tsv')


# In[ ]:


# for net in ['WordNet', 'BabelNet', 'WordNetDomain', 'LexNames']:
#     output_vocab = df[net].dropna().unique()
#     output_path = "../resources/semcor.vocab.{}.json".format(net)
#     print(output_path)
# #     with open(output_path, 'w') as f:
# #         f.write('\n'.join(output_vocab))
#     with open(output_path, 'w') as f:
#         json.dump(list(output_vocab), f)


# # Create mapping file between synset types to be used for all purposes

# In[8]:


def create_mapping(output_path = "../resources/mapping.csv",
                   babelnet2wordnet = '../resources/babelnet2wordnet.tsv',
                   babelnet2wndomains = '../resources/babelnet2wndomains.tsv',
                   babelnet2lexnames = '../resources/babelnet2lexnames.tsv'):
    """
    creates a mapping csv
    :param output_path: path
    :param babelnet2wordnet: path
    :param babelnet2wordnet: path
    :param babelnet2wordnet: path
    :return None: saves output csv to output_path
    """
    
    BabelNet = pd.read_csv(babelnet2wordnet, sep = '\t', names = ['babelnet', 'WordNet'])
    WordNetDomain = pd.read_csv(babelnet2wndomains, sep = '\t', names = ['babelnet', 'wordnet_domains'])
    LexicographerNet = pd.read_csv(babelnet2lexnames, sep = '\t', names = ['babelnet', 'lexicographer'])
    
    df = BabelNet.join(WordNetDomain.set_index('babelnet'), on='babelnet')
    df = df.join(LexicographerNet.set_index('babelnet'), on='babelnet')
    
    df.wordnet_domains.fillna("factotum", inplace=True)
    df.lexicographer.fillna("misc", inplace=True)
    
    df.to_csv(output_path, index = False)


# In[9]:


create_mapping()


# In[ ]:





# In[ ]:





# In[ ]:




