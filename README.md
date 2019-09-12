# NLP Homework 3

## Submission skeleton
```
- code
  |__ predict.py # This file contains all the functions you have to implement.
- resources # mappings are in here. you should place any additional resource (e.g. trained models) in here
  |__ babelnet2lexnames.tsv  # bnids to lexnames
  |__ babelnet2wndomains.tsv # bnids to WordNet domains labels
  |__ babelnet2wordnet       # bnids to WordNet offsets
- README.md # this file
- Homework_3_nlp.pdf # the slides presenting this homework
- report.pdf	# your report
```

## Instructions
Place all your code in the `code` folder. You can create other files.
Place any additional resources needed for the functions in `predict.py` (such as the weights of your trained models) in the `resources` folder.
Place your report as `report.pdf` in the root folder.
Follow the slides for any additional information.


## Code structure inside [code](code) folder
- [parsers.py](code/parsers.py)
  - Train(XML) and Gold parsers
- [utils.py](code/utils.py)
  - a whole lot helper functions used in other files
- [models.py](code/utils.py)
  - provides three models including MFS, Basic, and Multitask
- [train.py](code/utils.py)
  - trains a Basic Model using the generator [generatorPrototype.py](code/generatorPrototype.py)
- [trainMultitask.py](code/trainMultitask.py)
  - trains a Multitask Model using the generator [generatorMultitask.py](code/generatorMultitask.py)
- [predict.py](code/predict.py)
  - predict functions for balenet, wnDomain, and lex. Default model used is Multitask
- [running_predictions.py](code/running_predictions.py)
  - This is only used with the WSD_evaluation framework placed inside the (resources) folder. Uses the java Scorer.
  - .py version of [running_predictions.ipynb](code/gridSearch_visualisation.ipynb)
- [parsing.py](code/parsing.py)
  - parsing implementation, includes vocab creation of both lemmas and various senses. Mapping file.
  - .py version of [parsing.ipynb](code/parsing.ipynb)
