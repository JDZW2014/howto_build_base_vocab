"""
date：          2021-03-18
Description : compute idf
auther : wcy
"""
# import modules
import sys
from math import *

__all__ = ["frequent_pattern_mining"]


def compute_idf(raw_text_input='rawText.txt', stopwords_output='stopwordsFromText.txt'):
    in_docs = {}
    occurrence = {}
    docs_n = 0
    tokens_n = 0
    for line in open(raw_text_input, 'r'):
        docs_n += 1

        inside = 0
        chars = []
        tokens = {}
        for ch in line:
            if ch == '(':
                inside += 1
            elif ch == ')':
                inside -= 1
            elif inside == 0:
                if ch.isalpha():
                    chars.append(ch.lower())
                elif ch == '\'':
                    chars.append(ch)
                else:
                    if len(chars) > 0:
                        token = ''.join(chars)
                        tokens_n += 1
                        if token in occurrence:
                            occurrence[token] += 1
                        else:
                            occurrence[token] = 1
                        tokens[token] = True
                    chars = []
        if len(chars) > 0:
            token = ''.join(chars)
            tokens_n += 1
            if token in occurrence:
                occurrence[token] += 1
            else:
                occurrence[token] = 1
            tokens[token] = True
            chars = []
        for token in tokens:
            if token in in_docs:
                in_docs[token] += 1
            else:
                in_docs[token] = 1

    # print 'tokens = ', tokensN
    # print 'docs = ', docsN

    rank = []
    for token, occur in occurrence.items():
        tf = occur / float(tokens_n)
        idf = max(log(docs_n / float(in_docs[token])), 1e-10)
        rank.append((token, tf * idf))
    sorted_x = sorted(rank, key=lambda x: -x[1])

    out = open(stopwords_output, 'w')
    for token, key in sorted_x:
        out.write(str(token) + ',' + str(key) + '\n')
    out.close()


if __name__ == "__main__":
    compute_idf(sys.argv[1:])
