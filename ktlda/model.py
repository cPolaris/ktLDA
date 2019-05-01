import numpy as np
from cgibbs_inf import inf


class lda_cython:
    """
    Input parameters:
        K: number of topics
        alpha: Parameter for Dirichlet distribution for document topic distribution.
        beta: Parameter for Dirichlet distribution for topic word distribution.
        docs: Documents encoded with index.
        vocab_len: Length of vocabulary.
        iteration: Number of Iteration.
    """

    def __init__(self, K=25, alpha=0.5, beta=0.5, docs=None, vocab_len=None, iteration=10):
        """Initialize the parameters"""

        self.K, self.α, self.β, self.docs, self.vocab_len, self.iteration = K, alpha, beta, docs, vocab_len, iteration
        self.docs_len = len(self.docs)

        self.topic_doc_word = np.zeros(docs.shape, dtype=int)
        self.topic_doc_word.fill(-1)

        self.cnt_doc_topic = np.zeros((self.docs_len, self.K)) + alpha
        self.cnt_topic_word = np.zeros((self.K, vocab_len)) + beta
        self.cnt_topic = np.zeros(self.K) + vocab_len * beta

        for d_i, doc in enumerate(docs):
            for idx, do in enumerate(doc):
                if do < 0:
                    continue
                i = int(do)
                z = np.random.randint(0, K)

                # Randomly assigned topic
                self.topic_doc_word[d_i, idx] = z

                # Increase the counters accordingly
                self.cnt_doc_topic[d_i, z] += 1
                self.cnt_topic_word[z, i] += 1
                self.cnt_topic[z] += 1

        print(self.topic_doc_word)

    def inference(self):
        """
        Cython Optimized inference function.
        Pseudocode Available in the report.
        """

        inf(self.K, self.α, self.β, self.docs, self.vocab_len, self.iteration, self.topic_doc_word, self.cnt_doc_topic,
            self.cnt_topic_word, self.cnt_topic)

    def topics_in_document_distribution(self):
        """Output the distribution of topics in documents."""

        topic_distribution = np.zeros(self.cnt_doc_topic.shape)
        for idx, row in enumerate(self.cnt_doc_topic):
            topic_distribution[idx] = self.cnt_doc_topic[idx] / self.cnt_doc_topic[idx].sum()
        return topic_distribution

    def words_in_topic_distributions(self):
        """Output the distribution of words in topics."""

        return self.cnt_topic_word / self.cnt_topic.reshape(self.cnt_topic_word.shape[0], 1)
