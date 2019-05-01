import numpy as np


def prepare_documents(documents):
    """
    :param documents:
    :return: (doc_lda, V). doc_lda: documents represented by word index.
        V: word to index mapping
    """
    docs = []

    for doc in documents:
        docs.append(doc.split())

    dicts = []
    for d in docs:
        for w in d:
            dicts.append(w)

    V = dicts = list(set(dicts))
    dicts = dict([(w, idx) for idx, w in enumerate(dicts)])

    for i, d in enumerate(docs):
        for j, w in enumerate(d):
            docs[i][j] = dicts[docs[i][j]]

    doc_lda = docs

    b = np.zeros([len(doc_lda), len(max(doc_lda, key=lambda x: len(x)))], dtype=int)
    b.fill(-1)
    for i, j in enumerate(doc_lda):
        b[i][0:len(j)] = j

    doc_lda = b
    return doc_lda, V


def ldagen(vocab, alpha, beta, lam, ndoc=10, rnd_seed=663):
    """
    Performs the following generative process of LDA:
    - sample document length: N ~ Poisson(ξ)
    - sample document-topic distribution: θ ~ Dir(α)
    - for each word w_n:
        - sample topic: k ~ Categorical(θ)
        - sample word: w ~ Categorical(β_k)

    :param vocab: vocabulary. A list of words.
    :param alpha: param for Dir distribution of document-topic. Shape (num_D, num_T)
    :param beta: param for Dir distribution of topic-word. Shape (num_T, len(vocab))
    :param lam: param for Poisson distribution
    :param rnd_seed: seed for random generator
    :return docs: documents
    """
    np.random.seed(rnd_seed)
    docs = []
    for i in range(ndoc):
        curr_doc = []
        num_words = np.random.poisson(lam)
        theta = np.random.dirichlet(alpha)
        for _ in range(num_words):
            k = np.argmax(np.random.multinomial(1, theta))
            w = None
            curr_doc.append(w)
    return docs


if __name__ == '__main__':
    vocab = ['good', 'great', 'bad', 'terrible']
    alpha = []
    ldagen(vocab, )
