import models
import utils
from tensorflow.random import set_random_seed
set_random_seed(42)
import tensorflow.keras as K

def trainBasicModel(training_file_path, gold_file_path,
                    training_file_path_dev, gold_file_path_dev,
                    fine_senses_vocab_path,
                    input_vocab_path, input_antivocab_path,
                    embedding_size = 32,
                    batch_size = 64,
                    LEARNING_RATE = 0.01,
                    N_EPOCHS = 10,
                    PADDING_SIZE = 50,
                    print_model = True):
    
    #loading dict
    senses = utils.json_vocab_reader(fine_senses_vocab_path)
    inputs, antivocab = utils.json_vocab_reader(input_vocab_path, input_antivocab_path)
    output_vocab = utils.vocab_merge(senses, inputs)
    reverse_output_vocab =  dict((v, k) for k, v in output_vocab.items())
    
    K.backend.clear_session()
    model = models.Basic(test=2)
    
    
    BasicModelNetwork = model.build(vocab_size = len(output_vocab),
                                    embedding_size = embedding_size,
                                    hidden_size = 32,
                                    PADDING_SIZE = PADDING_SIZE,
                                    LEARNING_RATE = LEARNING_RATE,
                                    INPUT_DROPOUT = 0.2,
                                    LSTM_DROPOUT = 0.45,
                                    RECURRENT_DROPOUT = 0.35,
                                    N_EPOCHS = N_EPOCHS)

    if print_model:
        BasicModelNetwork.summary()
        
        
    train_generator = model.prepare_sentence_batch(batch_size = batch_size,
                                                    training_file_path = training_file_path,
                                                    gold_file_path = gold_file_path,
                                                    antivocab = antivocab,
                                                    output_vocab = output_vocab,
                                                    PADDING_SIZE = PADDING_SIZE)
    
    validation_generator = model.prepare_sentence_batch(batch_size = batch_size,
                                                         training_file_path = training_file_path_dev,
                                                         gold_file_path = gold_file_path_dev,
                                                         antivocab = antivocab,
                                                         output_vocab = output_vocab,
                                                         PADDING_SIZE = PADDING_SIZE)
    
        
#     BasicModelNetwork.fit_generator(generator, 
#                                     steps_per_epoch=None,
#                                     epochs=1, 
#                                     verbose=3,
#                                     callbacks=None,
#                                     validation_data=None,
#                                     validation_steps=None,
#                                     validation_freq=1,
#                                     class_weight=None,
#                                     max_queue_size=10,
#                                     workers=-1, 
#                                     use_multiprocessing=False,
#                                     shuffle=False,
#                                     initial_epoch=0)
    
    
    
# if __name__ == '__main__':
#     trainBasicModel(training_file_path = ,
#                     gold_file_path,
#                     training_file_path_dev,
#                     gold_file_path_dev,
#                     fine_senses_vocab_path = '../resources/semcor.vocab.WordNet.json',
#                     input_vocab_path = '../resources/semcor.input.vocab.json',
#                     input_antivocab_path = '../resources/semcor.leftout.vocab.json',
#                     embedding_size = 32,
#                     batch_size = 64,
#                     LEARNING_RATE = 0.01,
#                     N_EPOCHS = 10,
#                     PADDING_SIZE = 50
#                     print_model = True):
        
    