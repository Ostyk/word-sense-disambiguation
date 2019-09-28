# Word sense disambigutation

## explanation to follow

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
