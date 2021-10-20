# documentation https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# create a small corpus
documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

# next we will tokenize the text, remove stopwords, and words that appear once

from pprint import pprint  # pretty-printer
from collections import defaultdict

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]

pprint(texts)

# next we will convert words to vectors using document representation called "bag-of-words"
# each document is represented by one vector
# each vector represents question-answer pair

from gensim import corpora
dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester.dict')  # store the dictionary, for future reference
print(dictionary)

# mapping between questions and IDs is called dictionary
print(dictionary.token2id)

# we can also convert tokenized documents to vectors
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)  # the word "interaction" does not appear in the dictionary and is ignored

# result of the code above is [(0, 1), (1, 1)]
# where (0,1) is for human, and (1,1) is for computer (both words appear in the dictionary)
# interaction didn't appear in the text, thus not part of the dictionary

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
print(corpus)

from smart_open import open  # for transparently opening remote files

# we are going to open document that is stored online

class MyCorpus:
    def __iter__(self):
        for line in open('https://radimrehurek.com/mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

# Gensim accepts any corpus (no need for it to be pandas, list or array)
# the document can we created on the fly
 # doesn't load the corpus into memory!
corpus_memory_friendly = MyCorpus()
print(corpus_memory_friendly)

# load one vector into memory at a time
for vector in corpus_memory_friendly:
    print(vector)


# construct dictionary without loading text into memory
# collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in open('https://radimrehurek.com/mycorpus.txt'))
# remove stop words and words that appear only once
stop_ids = [
    dictionary.token2id[stopword]
    for stopword in stoplist
    if stopword in dictionary.token2id
]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print(dictionary)