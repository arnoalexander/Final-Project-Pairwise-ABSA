"""
Read tsv file containing sentences (list of tokens with labels and pairing)

Sentence with n tokens is represented with n rows. k-th row represents k-th token and consists of 2 or 3 string
separated by tab.
*   First string is the token itself
*   Second string is label of token. The possible labels are ASPECT and SENTIMENT with BIO encoding: B-ASPECT, I-ASPECT,
    B-SENTIMENT, I-SENTIMENT, O
*   Third string (optional) is only possible for token labelled with B-SENTIMENT. It is an integer x to show that this
    sentiment is related to x-th aspect of this sentence. If the sentiment have no relation with any aspect, this string
    would not exist (there are only 2 strings in this row)
"""

import definition


class Reader:

    """
    Utility class
    """

    # constants
    IDX_TOKEN = 0
    IDX_LABEL = 1
    IDX_PAIR = 2
    STATE_O = 0
    STATE_ASPECT = 1
    STATE_SENTIMENT = 2

    @classmethod
    def read_file(cls, path):
        """
        Read a tsv file with predefined format
        """
        infile = open(path, mode='r')
        data = infile.read()
        return cls.read_string(data)

    @classmethod
    def read_string(cls, data):
        """
        Read a string with predefined format
        """
        results = []
        sentence = None
        lines = data.splitlines()
        for idx, line in enumerate(lines):
            if line == '':
                if sentence is not None:
                    results.append(cls._parse_sentence(sentence))
                    sentence = None
            else:
                if sentence is None:
                    sentence = []
                line = line.split('\t')
                sentence.append(line)
                if idx == len(lines) - 1:
                    results.append(cls._parse_sentence(sentence))
                    sentence = None

        return results

    @classmethod
    def _parse_sentence(cls, sentence):
        result = {'token': [element[cls.IDX_TOKEN] for element in sentence],
                  'label': [element[cls.IDX_LABEL] for element in sentence],
                  'aspect': [{'start': index,
                              'length': 1}
                             for index, label in enumerate([element[cls.IDX_LABEL] for element in sentence])
                             if label == 'B-ASPECT'],
                  'sentiment': [{'start': index,
                                 'length': 1,
                                 'index_aspect': []}
                                for index, label in enumerate([element[cls.IDX_LABEL] for element in sentence])
                                if label == 'B-SENTIMENT']}

        for index, element in enumerate(result['aspect']):
            index_token = element['start'] + 1
            while index_token < len(sentence) and sentence[index_token][cls.IDX_LABEL] == 'I-ASPECT':
                result['aspect'][index]['length'] += 1
                index_token += 1
        for index, element in enumerate(result['sentiment']):
            index_token = element['start'] + 1
            while index_token < len(sentence) and sentence[index_token][cls.IDX_LABEL] == 'I-SENTIMENT':
                result['sentiment'][index]['length'] += 1
                index_token += 1
            if len(sentence[element['start']]) >= 3:
                splitted_pair = sentence[element['start']][cls.IDX_PAIR].split(',')
                result['sentiment'][index]['index_aspect'] = [int(index_aspect) for index_aspect in splitted_pair]

        return result


if __name__ == '__main__':
    print(Reader.read_file(definition.DATA_PAIRED_SMALLSAMPLE))
