from nltk.tokenize import RegexpTokenizer
import re


class Preprocessor:

    @classmethod
    def tokenize(cls, sentence):

        """
        Tokenize a sentence into list of token
        """

        if sentence is None:
            return []
        else:
            result = sentence.lower()
            result = re.sub(r'(\.+|\,+|\!+|\?+|\-+|\/+|\&+|\:+|\(+|\)+)', lambda x: ' ' + x.group()[0] + ' ', result)

            tokenizer = RegexpTokenizer(r'[\w\.\,\!\?\-\/\&\:\(\)]+')
            result = tokenizer.tokenize(result)

            return result


if __name__ == '__main__':
    sent = input('Sentence: ')
    print(Preprocessor.tokenize(sent))
