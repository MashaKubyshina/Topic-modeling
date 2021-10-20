# documentation https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html#sphx-glr-auto-examples-core-run-core-concepts-py

import pprint
import gensim

# import the document
document = "Human machine interface for lab abc computer applications"

# import corpus, a collection of documents
# corpus will be used for training the model
# the model is trained to lok for common themes and topics
# gensim is focused on unsupervised models (documents are not tagged by a human)
# corpus is also used to find similarity queries (semantic similarities and clusters)

text_corpus = [
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

# Next step is to pre-process the corpus (remove common words, tokenize)
# Create a set of frequent words

stoplist = set('for a of the and to in'.split(' '))

# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in text_corpus]

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
pprint.pprint(processed_corpus)

# next we need to associate each word in the corpus with a unique integer ID
# for this we use dictionary class

from gensim import corpora

dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)

# next we are going to convert words to vectors
# and documents as vectors of features (a phrase can be single feature)
# single feature can be thought as a question-answer pair
# we know it as "dense vector" because it contains answers to questions, including zero values
# gensim omits zero values, it is known as sparse vector or bag-of-words
# if vectors of two documents are similar this means the documents are similar as well

# we use dictionary to turn tokenized documents into 12-dimensional vectors
# we have 12 unique words

pprint.pprint(dictionary.token2id)

# test the vectorization on a new document that is not in the corpus
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)

# next we convert our entire original corpus into a list of vectors
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
pprint.pprint(bow_corpus)

# now we will move to training the model on our corpus
from gensim import models

# we will use Tfidf model from gensim
# tf-idf transforms vectors from the bag-of-words representation
# to a vector space where frequency counts are weighted according to
# rarity of each word in the corpus

# train the model
tfidf = models.TfidfModel(bow_corpus)

# transform the "system minors" string
words = "system minors".lower().split()
print(tfidf[dictionary.doc2bow(words)])
# in the output we will see that ID corresponding to "system"
# (occurred 4 times) was weighted down

from gensim import similarities

# now we can transform the whole corpus
index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)

# and query the similarity of query document against every document of the corpus
query_document = 'system engineering'.split()
query_bow = dictionary.doc2bow(query_document)
sims = index[tfidf[query_bow]]
print(list(enumerate(sims)))

# the document with highest similarity score will be placed at the top

for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
    print(document_number, score)