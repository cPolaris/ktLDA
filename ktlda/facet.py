"""
Adapt to sklearn-like API for client usage
"""

from ktlda.model import lda_cython
import numpy as np
from collections import Counter
from tqdm.autonotebook import tqdm


class KtLDA:
    """
    LDA topic model via Collapsed Gibbs Sampling
    """

    def __init__(self, n_components, alpha=0.5, beta=0.5, iterations=10, max_vocab=5000, random_state=None):
        """

        :param n_components: number of topics
        :param alpha: Dirichlet parameter for document-topic distribution
        :param beta: Dirichlet parameter for topic and word distribution
        :param iterations: number of iterations
        :param max_vocab: number of most frequent words to preserve. Less frequent words
                          will be dropped
        :param random_state: sets the seed of numpy random number generator if not None
        """
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.max_vocab = max_vocab
        if random_state:
            np.random.seed(random_state)
        self.model = None
        self.doc_topic_dist = None
        self.topic_word_dist = None

    def _convert_docs(self, documents, max_vocab=5000):
        """
        Convert documents to acceptable input to our algorithm
        :return: (document matrix, vocabulary dict)
        """
        print('word counting')
        wordcount = Counter()
        for doc in tqdm(documents):
            wordcount += Counter(doc.split())

        vocab, _ = zip(*wordcount.most_common(max_vocab))
        dicts = dict([(w, idx) for idx, w in enumerate(vocab)])

        print('transform doc while removing least frequent words')
        docs = []
        vocabset = set(vocab)
        for i, d in tqdm(enumerate(documents)):
            docs.append([])
            for j, w in enumerate(d.split()):
                if w in vocabset:
                    docs[i].append(dicts[w])

        doc_mat = np.zeros([len(docs), len(max(docs, key=lambda x: len(x)))], dtype=int)
        doc_mat.fill(-1)
        for i, words in enumerate(docs):
            for j, w in enumerate(words):
                doc_mat[i, j] = w
        return doc_mat, dicts

    def fit(self, documents):
        doc_mat, dicts = self._convert_docs(documents, self.max_vocab)
        self.model = lda_cython(K=self.n_components, alpha=self.alpha, beta=self.beta, docs=doc_mat,
                                vocab_len=len(dicts),
                                iteration=self.iterations)
        self.model.inference()
        self.doc_topic_dist = self.model.topics_in_document_distribution()
        self.topic_word_dist = self.model.words_in_topic_distributions()
