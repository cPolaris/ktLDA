import unittest

from ktlda.utils import prepare_documents

TOY_DOCS = ['sunny hot heat frost icy icy',
            'sunny sunny hot heat sunny hot sunny hot',
            'sunny sunny hot hot hot',
            'sunny hot heat heat heat sunny hot heat',
            'icy frost icy frost frost icy icy icy frost frost icy icy',
            'frost frost icy frost frost icy',
            'frost icy icy icy icy icy frost icy']


class TestUtils(unittest.TestCase):
    def test_correct(self):
        print(prepare_documents(TOY_DOCS))


if __name__ == '__main__':
    unittest.main()
