from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load dataset
raw_dataset = load_clean_sentences('tweet-response.pkl')

# reduce dataset size
n_sentences = 375000
dataset = raw_dataset[:n_sentences, :]
# random shuffle
shuffle(dataset)
# split into train/test
test, train = dataset[:2000], dataset[2000:]
# save
save_clean_data(dataset, 'data/tweet-response-both.pkl')
save_clean_data(train, 'data/tweet-response-train.pkl')
save_clean_data(test, 'data/tweet-response-test.pkl')
