"""
Read tsv file containing sentences (list of tokens with labels)

Sentence with n tokens is represented with n rows. k-th row represents k-th token and consists of 1 or 2 string
separated by tab.
*   First string is the token itself
*   Second string (optional) is label of token. The possible labels are ASPECT and SENTIMENT with BIO encoding: B-ASPECT, I-ASPECT,
    B-SENTIMENT, I-SENTIMENT, O

The result is document representation in dict/json format
"""

import os
import definition


class Reader:

    """
    Reader utility class for NER tagging task
    """

    # constants
    IDX_TOKEN = 0
    IDX_LABEL = 1

    @classmethod
    def read_file(cls, path, with_target=True):
        """
        Read a tsv file with predefined format
        """
        infile = open(path, mode='r')
        data = infile.read()
        return cls.read_string(data=data, with_target=with_target)

    @classmethod
    def read_string(cls, data, with_target=True):
        """
        Read a string with predefined format
        """
        results = []
        sentence = None
        lines = data.splitlines()
        for idx, line in enumerate(lines):
            if line == '':
                if sentence is not None:
                    results.append(cls._parse_sentence(sentence=sentence, with_target=with_target))
                    sentence = None
            else:
                if sentence is None:
                    sentence = []
                line = line.split('\t')
                sentence.append(line)
                if idx == len(lines) - 1:
                    results.append(cls._parse_sentence(sentence=sentence, with_target=with_target))
                    sentence = None

        return results

    @classmethod
    def _parse_sentence(cls, sentence, with_target=True):
        result = {'token': [element[cls.IDX_TOKEN] for element in sentence]}

        if not with_target:
            return result

        result['label'] = [element[cls.IDX_LABEL] for element in sentence]

        return result


if __name__ == '__main__':
    print(Reader.read_file(os.path.join(definition.DATA_LABELLED, 'sample-small.txt'))[0])
