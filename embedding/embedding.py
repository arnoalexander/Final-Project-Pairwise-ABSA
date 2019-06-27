import gensim


class FastTextEmbedding:

    """
    Wrapper class for gensim FastText
    """

    def __init__(self, *args, **kwargs):
        self.model = gensim.models.FastText(*args, **kwargs)

    def build_vocab(self, *args, **kwargs):
        self.model.build_vocab(*args, **kwargs)

    def train(self, *args, **kwargs):
        self.model.train(*args, **kwargs)


if __name__ == '__main__':
    document = [["the", "quick", "brown", "fox"], ["jumps", "over", "a", "lazy", "dog"]]
    test_word = "test"
    embedding = FastTextEmbedding(min_count=1, size=10, min_n=3)
    embedding.build_vocab(document)
    try:
        print(embedding.model.wv[test_word])
    except KeyError:
        print("not found")
