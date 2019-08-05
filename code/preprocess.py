import tarfile
import os
import xml.etree.ElementTree as etree
import pandas as pd
from nltk.corpus import wordnet as wn
from tqdm import tqdm, tnrange, tqdm_notebook
import csv
import json

def sensekeyToSynsetConverter(sensekey: str):
    '''retrieves a WordNet synset from a sensekey using the nltk package'''
    synset = wn.lemma_from_key(sensekey).synset()
    
    synset_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
    return synset_id

archived_xml = '../resources/WSD_Evaluation_Framework/semcor+omsti.data.xml'
mapping_file = '../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor+OMSTI/semcor+omsti.gold.key.txt'

mapping = pd.read_table(mapping_file, sep = ' ', names = ['sentence_idx', 'sensekey1', 'sensekey2'])

tqdm.pandas(desc="my bar!")
# converting from sensekey to synset id for the two columns
mapping['sensekey1'] = mapping['sensekey1'].progress_apply(sensekeyToSynsetConverter)

tqdm.pandas(desc="my bar!")
# using notnull() instead of dropna because dropna() does not work on particular columns
mapping['sensekey2'][mapping['sensekey2'].notnull()] = mapping['sensekey2'][mapping['sensekey2'].notnull()].progress_apply(sensekeyToSynsetConverter)
context = etree.iterparse(archived_xml, events=("start", "end"))

with open('../resources/parsed_corpora.csv', 'w', encoding='utf-8') as file:
    
    csv_writer =  csv.writer(file)
    csv_writer.writerow(('id', 'X', 'y','sensekeyCount'))
    
    for idx, (event, elem) in enumerate(tqdm(context)):
        
        if elem.tag == 'sentence' and event == 'start':
            sentence_id = elem.get("id")
            X, y, senseCount = [], [], 0

        if elem.tag == "wf" and event == 'start':
            X.append(elem.text)
            y.append(elem.text)

        if elem.tag == "instance" and event == 'start':
            # get mapping from idx
            m = mapping[mapping['sentence_idx']== elem.get("id")]
            X.append(elem.text)

            #get sensekeys from mapping row
            l = [m['sensekey1'].iloc[0], m['sensekey2'].iloc[0]]
            cleanedList = [x for x in l if str(x) != 'nan'] #gets rid of NaN's
            senseCount += len(cleanedList)
            y.append(cleanedList)

        if elem.tag == 'sentence' and event == 'end':
            csv_writer.writerow([sentence_id, X, y, senseCount])

        if (idx+1)%10**6==0:
            print("{:2f}% complete".format((idx/32946488)*100))

        elem.clear()
del context
