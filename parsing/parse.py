import numpy as np
import time


def parse_sentence_padd(dictionary, sentence, size=None, pad_in_front=False):
    """
    :param dictionary: the dictionary to be used for parsing. It has to have a dictionary.rare_string and dictionary.padding_string
    :param sentence: the sentence to parse
    :param size: the max size, above this number of tokens the sentence is truncated, shorter and it's padded
    :param pad_in_front: boolean that decides whether padding needs to be done in front or at the end
    :return: a tuple with the encoded tokens (lowercased) and a set of features for the casing of the letters
    """
    tokens = dictionary.tokenize(sentence, lowercase=False)
    encoded_tokens = dictionary.encode(sentence, size=size, pad_in_front=False)
    extra_features = np.zeros((len(encoded_tokens), 3))
    for i in xrange(len(encoded_tokens)):
        if i >= len(tokens):
            continue
        token = tokens[i]
        if token == dictionary.rare_string or token == dictionary.padding_string:
            continue

        feats = extra_features[i, :]
        if token[0].isupper():
            feats[0] = 1
        if token.isupper():
            feats[1] = 1
        for token_pos in xrange(len(token)):
            if token[token_pos].isupper():
                feats[2] = 1
                break

    return encoded_tokens, extra_features

def parse_sentence_padd_batch(dictionary, sentence,batch_size, size=None, pad_in_front=False):
    """
    :param dictionary: the dictionary to be used for parsing. It has to have a dictionary.rare_string and dictionary.padding_string
    :param sentence: the sentence to parse
    :param size: the max size, above this number of tokens the sentence is truncated, shorter and it's padded
    :param pad_in_front: boolean that decides whether padding needs to be done in front or at the end
    :return: a tuple with the encoded tokens (lowercased) and a set of features for the casing of the letters
    """
    encoded_tokens = np.zeros(shape=(batch_size,size))
    extra_features =np.zeros(shape=(batch_size,size,3))
    for i_batch in xrange(batch_size):
        tokens = dictionary.tokenize(sentence[1][i_batch], lowercase=False)
        encoded_tokens[i_batch,:] = dictionary.encode(sentence[1][i_batch], size=size)
        for i in xrange(len(encoded_tokens[i_batch])):
            if i >= len(tokens):
                continue
            token = tokens[i]
            if token == dictionary.rare_string or token == dictionary.padding_string:
                continue

            feats = extra_features[i_batch,i, :]
            if token[0].isupper():
                feats[0] = 1
            if token.isupper():
                feats[1] = 1
            for token_pos in xrange(len(token)):
                if token[token_pos].isupper():
                    feats[2] = 1
                    break
    return encoded_tokens, extra_features
