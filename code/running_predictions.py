
# coding: utf-8

# In[ ]:


import predict
import os
import subprocess


# In[ ]:


#!ls ../resources/models/best_model/ -asl


# In[ ]:


from collections import namedtuple


# In[ ]:


dir_ = "../resources/WSD_Evaluation_Framework/Evaluation_Datasets"
eval_datasets = sorted([i for i in os.listdir(dir_) if i.startswith("se")])
resources_path = '../resources'
del eval_datasets[1]
eval_datasets


# In[ ]:


bashCommand = "sudo javac ../resources/WSD_Evaluation_Framework/Evaluation_Datasets/Scorer.java"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
output,error


# In[ ]:


scores = {"babelnet": {}, 'wordnet_domains': {}, 'lexicographer': {}}
for name in eval_datasets:
    for key in list(scores.keys()):
        scores[key].update({name:None})
scores


# In[ ]:


record = namedtuple("predictions", "build perform")

Basic_model = record(True, True) # task determines if its basic or Multitask

MFS_baseline = record(False, False)

task = 'Multitask'
#assert not (perform_predictions and MFS_baseline)


# In[ ]:


for name in eval_datasets:
    print("Dataset: {}\n".format(name))
    path = os.path.join(dir_, name)
    xml_file = [i for i in os.listdir(path) if i.endswith('.xml')][0]
    xml_file = os.path.join(path, xml_file)
    print(xml_file)
    print("_"*50)
    if Basic_model.build:
        predict.predict_babelnet(input_path =   '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.data.xml'.format(name, name),
                                 output_path = '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.pred.babelnet.{}.txt'.format(name, name, task),
                                 resources_path = resources_path)
    if Basic_model.perform:
        bashCommand = "sudo java Scorer {}/{}.gold.babelnet.txt {}/{}.pred.babelnet2.txt".format(name, name, name, name)
    if MFS_baseline.build:
        predict.MFS_predict_writer(input_path =   '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.data.xml'.format(name, name),
                                   output_path = '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/MFS.{}.pred.babelnet.{}.txt'.format(name, name, task),
                                   resources_path = resources_path,
                                   prediction_type = 'babelnet')
    if MFS_baseline.perform:
        bashCommand = "sudo java Scorer {}/{}.gold.babelnet.txt {}/MFS.{}.pred.babelnet.{}.txt".format(name, name, name, name, task)
    ########################################################
    os.chdir("../resources/WSD_Evaluation_Framework/Evaluation_Datasets/")
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    scores['babelnet'][name] = float(output.decode("UTF-8").split("\n")[0].split("\t")[1].split("%")[0])
    print("babelnet: {}".format(name))
    for i in output.decode("UTF-8").split("\n"):
        print(i)
    os.chdir("../../../code")
    
    ########################################################
    ########################################################
    ########################################################
    
    print("_"*50)
    if Basic_model.build:
        predict.predict_wordnet_domains(input_path =   '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.data.xml'.format(name, name),
                                        output_path = '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.pred.wordnet_domains.{}..txt'.format(name, name, task),
                                        resources_path = resources_path)
    if Basic_model.perform:
        bashCommand = "sudo java Scorer {}/{}.gold.wordnet_domains.txt {}/{}.pred.wordnet_domains2.txt".format(name, name, name, name)
    
    if MFS_baseline.build:
        predict.MFS_predict_writer(input_path =   '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.data.xml'.format(name, name),
                                   output_path = '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/MFS.{}.pred.wordnet_domain.{}.txt'.format(name, name, task),
                                   resources_path = resources_path,
                                   prediction_type = 'wordnet_domains')
    if MFS_baseline.perform:
        bashCommand = "sudo java Scorer {}/{}.gold.wordnet_domains.txt {}/MFS.{}.pred.wordnet_domains.{}.txt".format(name, name, name, name, task)    
        
        
    ########################################################
    os.chdir("../resources/WSD_Evaluation_Framework/Evaluation_Datasets/")
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    scores['wordnet_domains'][name] = float(output.decode("UTF-8").split("\n")[0].split("\t")[1].split("%")[0])
    print("wordnet_domains: {}".format(name))
    for i in output.decode("UTF-8").split("\n"):
        print(i)
    os.chdir("../../../code")
    
    ########################################################
    ########################################################
    ########################################################
    print("_"*50)
    if Basic_model.build:
        predict.predict_lexicographer(input_path =   '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.data.xml'.format(name, name),
                                      output_path = '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.pred.lexicographer.{}..txt'.format(name, name, task),
                                      resources_path = resources_path)
    if Basic_model.perform:
        bashCommand = "sudo java Scorer {}/{}.gold.lexicographer.txt {}/{}.pred.lexicographer2.txt".format(name, name, name, name)
    
    
    if MFS_baseline.build:
        predict.MFS_predict_writer(input_path =   '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.data.xml'.format(name, name),
                                   output_path = '../resources/WSD_Evaluation_Framework/Evaluation_Datasets/{}/MFS.{}.pred.lexicographer.{}..txt'.format(name, name, task),
                                   resources_path = resources_path,
                                   prediction_type = 'lexicographer')
    if MFS_baseline.perform:
        bashCommand = "sudo java Scorer {}/{}.gold.lexicographer.txt {}/MFS.{}.pred.lexicographer.{}..txt".format(name, name, name, name, task)    
        
    
    ########################################################
    os.chdir("../resources/WSD_Evaluation_Framework/Evaluation_Datasets/")
   
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


# In[ ]:


import pandas as pd


# In[ ]:


# MFS_scores = pd.DataFrame(scores)
# MFS_scores


# In[ ]:


# basicModel_scores = pd.DataFrame(scores)
# basicModel_scores


# In[ ]:


MultiTask_scores = pd.DataFrame(scores)
MultiTask_scores

