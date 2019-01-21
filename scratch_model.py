# prepare english tokenizer
tweet_tokenizer = create_tokenizer(dataset[:, 0])
tweet_vocab_size = min(len(tweet_tokenizer.word_index), 5000) + 1
tweet_length = max_length(dataset[:, 0])
print('Tweet Vocabulary Size: %d' % tweet_vocab_size)
print('Tweet Max Length: %d' % (tweet_length))
# prepare german tokenizer
response_tokenizer = create_tokenizer(dataset[:, 1])
response_vocab_size = min(len(response_tokenizer.word_index), 5000) + 1
response_length = max_length(dataset[:, 1])
print('Response Vocabulary Size: %d' % response_vocab_size)
print('Response Max Length: %d' % (response_length))
del dataset
# prepare training data
trainX = encode_sequences(tweet_tokenizer, tweet_length, train[:, 0])
trainY = encode_sequences(response_tokenizer, response_length, train[:, 1])
trainY = encode_output(trainY, response_vocab_size)
# prepare validation data
testX = encode_sequences(tweet_tokenizer, tweet_length, test[:, 0])
testY = encode_sequences(response_tokenizer, response_length, test[:, 1])
testY = encode_output(testY, response_vocab_size)

batch_trainX = trainX[1:1000]
batch_trainY = trainY[1:1000]
batch_trainY = encode_output(batch_trainY, response_vocab_size)

# define model
model = define_model(tweet_vocab_size, response_vocab_size, tweet_length, response_length, 256)
#model = define_non_seq_model(tweet_vocab_size, response_vocab_size, tweet_length, response_length, 256)
adam = Adam(lr=0.0001, decay=0.00001)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
#model(model, to_file='model.png', show_shapes=True)
# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(batch_trainX, batch_trainY, epochs=64, batch_size=256, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)
