from nltk.corpus import wordnet as wn

def sensekeyToSynsetConverter(sensekey: str):
    """
    Retrieves a WordNet synset from a sensekey using the nltk package'''
    :param sensekey
    :return: wordnet_id
    """
    synset = wn.lemma_from_key(sensekey).synset()
    
    wordnet_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
    return wordnet_id