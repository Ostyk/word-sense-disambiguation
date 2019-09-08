#!/usr/bin/env python
# coding: utf-8

# In[1]:


import predict
import os
import subprocess


# In[2]:


dir_ = "../resources/WSD_Evaluation_Framework/Evaluation_Datasets"
eval_datasets = sorted([i for i in os.listdir(dir_) if i.startswith("se")])
resources_path = '../resources'
del eval_datasets[1]
eval_datasets


# In[3]:


bashCommand = "sudo javac ../resources/WSD_Evaluation_Framework/Evaluation_Datasets/Scorer.java"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
output,error


# In[4]:


scores = {"babelnet": {}, 'wordnet_domains': {}, 'lexicographer': {}}
for name in eval_datasets:
    for key in list(scores.keys()):
        scores[key].update({name:None})
scores


# In[5]:


os.getcwd()


# In[7]:


perform_predictions = False

for name in eval_datasets:
    print("Dataset: {}\n".format(name))
    path = os.path.join(dir_, name)
    xml_file = [i for i in os.listdir(path) if i.endswith('.xml')][0]
    xml_file = os.path.join(path, xml_file)
    print(xml_file)
    print("_"*50)
    if perform_predictions:
        predict.predict_babelnet(input_path =   '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.data.xml'.format(name, name),
                                 output_path = '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.pred.babelnet.txt'.format(name, name),
                                 resources_path = resources_path)
    ########################################################
    os.chdir("../resources/WSD_Evaluation_Framework/Evaluation_Datasets/")
    bashCommand = "sudo java Scorer {}/{}.gold.babelnet.txt {}/{}.pred.babelnet.txt".format(name, name, name, name)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    scores['babelnet'][name] = float(output.decode("UTF-8").split("\n")[0].split("\t")[1].split("%")[0])
    print("babelnet: {}".format(name))
    for i in output.decode("UTF-8").split("\n"):
        print(i)
    os.chdir("../../../code")
    ########################################################
    print("_"*50)
    if perform_predictions:
        predict.predict_wordnet_domains(input_path =   '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.data.xml'.format(name, name),
                                        output_path = '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.pred.wordnet_domains.txt'.format(name, name),
                                        resources_path = resources_path)
    ########################################################
    os.chdir("../resources/WSD_Evaluation_Framework/Evaluation_Datasets/")
    bashCommand = "sudo java Scorer {}/{}.gold.wordnet_domains.txt {}/{}.pred.wordnet_domains.txt".format(name, name, name, name)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    scores['wordnet_domains'][name] = float(output.decode("UTF-8").split("\n")[0].split("\t")[1].split("%")[0])
    print("wordnet_domains: {}".format(name))
    for i in output.decode("UTF-8").split("\n"):
        print(i)
    os.chdir("../../../code")
    ########################################################
    print("_"*50)
    if perform_predictions:
        predict.predict_lexicographer(input_path =   '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.data.xml'.format(name, name),
                                      output_path = '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.pred.lexicographer.txt'.format(name, name),
                                      resources_path = resources_path)
    
    ########################################################
    os.chdir("../resources/WSD_Evaluation_Framework/Evaluation_Datasets/")
    bashCommand = "sudo java Scorer {}/{}.gold.lexicographer.txt {}/{}.pred.lexicographer.txt".format(name, name, name, name)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    scores['lexicographer'][name] = float(output.decode("UTF-8").split("\n")[0].split("\t")[1].split("%")[0])
    print("lexicographer: {}".format(name))
    for i in output.decode("UTF-8").split("\n"):
        print(i)
    os.chdir("../../../code")
    ########################################################
    print("_"*50)
    print("_"*50)
    print("_"*50)


# In[8]:


import pandas as pd


# In[9]:


pd.DataFrame(scores)


# In[ ]:




