import tensorflow.keras as K


def KerasModel(vocab_size, embedding_size, hidden_size, PADDING_SIZE, LEARNING_RATE, INPUT_DROPOUT, LSTM_DROPOUT,RECURRENT_DROPOUT, N_EPOCHS):
    print("Creating KERAS model")

    input = K.layers.Input(shape=(None,))
    embeddings = K.layers.Embedding(vocab_size,
                                    embedding_size,
                                    mask_zero=True,
                                    name = 'embedding')(input)


    BI_LSTM = (K.layers.Bidirectional(
               K.layers.LSTM(hidden_size, dropout = LSTM_DROPOUT,
                             recurrent_dropout = RECURRENT_DROPOUT,
                             return_sequences=True,
                             kernel_regularizer=K.regularizers.l2(0.01),
                             activity_regularizer=K.regularizers.l1(0.01)
                            ), name = 'Bi-directional_LSTM'))(embeddings)

    predictions = K.layers.TimeDistributed(K.layers.Dense(vocab_size, activation='softmax'))(BI_LSTM)

    model = K.models.Model(inputs=[input], outputs=predictions)

    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = K.optimizers.Adam(lr=LEARNING_RATE, decay = 0.001/N_EPOCHS, amsgrad=False),
                  metrics = ['acc'])

    return model
