import unittest

from ktlda.facet import KtLDA

TOY_DOCS = ['sunny hot heat frost icy icy',
            'sunny sunny hot heat sunny hot sunny hot',
            'sunny sunny hot hot hot',
            'sunny hot heat heat heat sunny hot heat',
            'icy frost icy frost frost icy icy icy frost frost icy icy',
            'frost frost icy frost frost icy',
            'frost icy icy icy icy icy frost icy']


class TestModel(unittest.TestCase):
    def test_correct(self):
        lda = KtLDA(n_components=2)
        lda.fit(TOY_DOCS)
        print(lda.doc_topic_dist)
        print(lda.topic_word_dist)


if __name__ == '__main__':
    unittest.main()
