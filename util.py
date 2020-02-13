# Instructor: Karl Stratos

import copy
import os
import math
import nltk
import numpy as np
import random

from collections import Counter
from transformers import BertTokenizer, RobertaTokenizer


WINDOW = 3  # Let's limit window size to be <= 3.


class LogLinearLanguageModel:

    def __init__(self, model, vocab, unk_symbol, feature_extractor, f2i, fcache,
                 x2ys, init=0.001, lr=0.01, check_interval=500, seed=42):
        self.model = model
        self.token_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
        self.token_to_idx[unk_symbol] = 0
        self.feature_extractor = feature_extractor
        self.f2i = f2i
        self.fcache = fcache
        self.x2ys = x2ys

        # Seed needed for reproducibility.
        random.seed(seed)
        np.random.seed(seed)

        self.w = np.random.uniform(-init, init, len(f2i))
        self.lr = lr
        self.check_interval = check_interval
        self.best_val_ppl = float('inf')

    def train(self, corpus, val_corpus, epochs):
        best_w = None
        num_bad_epochs = 0
        for ep in range(epochs):
            loss = self.do_epoch(corpus, val_corpus)
            val_ppl = self.test(val_corpus)
            print('Epoch {:3d} | avg loss {:8.4f} | running train ppl {:8.4f} '
                  '| val ppl {:8.4f}'.format(ep + 1, loss, np.exp(loss),
                                             val_ppl),
                  end='    ')

            if val_ppl < self.best_val_ppl:
                self.best_val_ppl = val_ppl
                print('***new best val ppl***')
                best_w = copy.deepcopy(self.w)
                num_bad_epochs = 0
            else:
                self.lr /= 4.0
                print('quartered learning rate to %g' % self.lr)
                num_bad_epochs += 1

            if num_bad_epochs >= 7:
                break

        np.save(self.model, best_w)

    def do_epoch(self, corpus, val_corpus):
        positions = list(range(WINDOW - 1, len(corpus)))
        random.shuffle(positions)
        total_loss = 0.0  # - sum_i ln q(y_i|x_i)

        for ex_num, position in enumerate(positions):
            x = corpus[position-WINDOW+1:position]
            y = corpus[position]
            # Compute the probability over the vocabulary given x.
            q = self.compute_probs(x)
            phi = np.zeros(len(self.token_to_idx))
            # print(len(self.fcache),len(self.w),len(self.f2i))

            # print(tuple(x)+tuple([y]), y_idx)
            # for ind_y in y_idx:
            #     phi[ind_y] = 1
            for _y in self.x2ys[tuple(x)]:
                val = 0
                if _y is y:
                    val = 1
                exp_val = 0
                y_idx = self.fcache[tuple(x)+tuple([_y])]
                feat = self.feature_extractor(tuple(x)+tuple([y]))
                for ind_y in y_idx:
                    # exp_val += q[ind_y] * 1
                    self.w[ind_y] -= self.lr * (q[self.token_to_idx[_y]]- val)

            # TODO: Implement the gradient update w = w - lr * grad_w J(w).
            # The update must be sparse. Do not work with the whole vector w.
            # Use caches self.fcache and self.x2ys.
            # for _y in self.x2ys[tuple(x)]:
            #     c = x.copy()
            #     c.append(y)
            #     y_idx = self.fcache[tuple(c)]
            #     for ind_y in y_idx:
            #         self.w[ind_y] -= self.lr * np.log(q[self.token_to_idx[_y]])
            # print(y, self.token_to_idx[y], q[self.token_to_idx[y]])
            # print(math.log(q[self.token_to_idx[y]]))
            # print()
            total_loss -= math.log(q[self.token_to_idx[y]])
            if (ex_num + 1) % self.check_interval == 0:
                print('%d/%d examples, avg loss %g' %
                      (ex_num + 1, len(positions), total_loss / (ex_num + 1)))

        return total_loss / len(positions)

    def compute_probs(self, x):
        # TODO: Calculate NumPy score vector q_ s.t. q_[ind(y)] = w' phi(x, y).
        q_ = np.zeros(len(self.token_to_idx))
        for y in self.x2ys[tuple(x)]:
            y_idx = self.fcache[tuple(x)+tuple([y])]
            for ind_y in y_idx:
                q_[self.token_to_idx[y]] += self.w[ind_y]
        return softmax(q_)

    def test(self, corpus):
        logprob = 0.

        for i in range(WINDOW - 1, len(corpus)):
            x = corpus[i-WINDOW+1:i]
            y = corpus[i]
            q = self.compute_probs(x)
            logprob += math.log(q[self.token_to_idx[y]])

        logprob /= len(corpus[:-WINDOW+1])
        ppl = np.exp(-logprob)

        return ppl

    def load(self):
        self.w = np.load(self.model)

    def topK_feats(self, K):
        K = min(K, len(self.f2i))
        i2f = [None for _ in range(len(self.f2i))]
        for f in self.f2i:
            i2f[self.f2i[f]] = f
        topk_inds = self.w.argsort()[-K:][::-1]
        topk = [(i, i2f[i], self.w[i]) for i in topk_inds]
        return topk


def basic_features2(window):
    return {'c-1=%s^w=%s' % (window[-2], window[-1]): True,
            'c-2=%s^w=%s' % (window[-3], window[-1]): True,
            'c-2=%s^c-1=%s^w=%s' % (window[-3], window[-2], window[-1]): True}


def basic_features1(window):
    return {'c-1=%s^w=%s' % (window[-2], window[-1]): True}


def basic_features1_suffix3(window):
    # a={'c-1=%s^w=%s' % (window[-2], window[-1]): True,
    #          'c-1s1=%s^w=%s' % (window[-2][-1], window[-1]): True,
    #          'c-1s2=%s^w=%s' % (window[-2][-2:], window[-1]): True,
    #          'c-1s3=%s^w=%s' % (window[-2][-3:], window[-1]): True}
    # print(window,a)
    # exit()
    return  {'c-1=%s^w=%s' % (window[-2], window[-1]): True,
             'c-1s1=%s^w=%s' % (window[-2][-1], window[-1]): True,
             'c-1s2=%s^w=%s' % (window[-2][-2:-1], window[-1]): True,
             'c-1s3=%s^w=%s' % (window[-2][-3:-1], window[-1]): True}  # TODO: Implement


def extract_features(training_corpus, feature_extractor):
    f2i = {}
    fcache = {}
    num_feats_cached = 0
    x2ys = {}

    window_types = {}
    for i in range(WINDOW - 1, len(training_corpus)):
        x = training_corpus[i-WINDOW+1:i]
        if not tuple(x) in x2ys:
            x2ys[tuple(x)] = {}
        y = training_corpus[i]
        x2ys[tuple(x)][y] = True

        window = tuple(training_corpus[i-WINDOW+1:i+1])

        inds = []
        for feature in feature_extractor(window):
            if not feature in f2i:
                f2i[feature] = len(f2i)
            inds.append(f2i[feature])

        if not window in fcache:
            fcache[window] = inds
            num_feats_cached += len(inds)

    return f2i, fcache, num_feats_cached, x2ys


def softmax(v):
    v_max = np.max(v)
    out = np.exp(v - v_max)
    out /= np.sum(out)
    return out  # TODO: Implement


class Tokenizer:

    def __init__(self, tokenize_type='basic', lowercase=False):
        self.tokenize_type = tokenize_type
        self.lowercase = lowercase

        if self.tokenize_type == 'wp':
            self.wptok = BertTokenizer.from_pretrained('bert-base-cased')

        if self.tokenize_type == 'bpe':
            self.bpetok = RobertaTokenizer.from_pretrained('roberta-base')

    def tokenize(self, string):
        if self.lowercase:
            string = string.lower()

        if self.tokenize_type == 'basic':
            tokens = string.split()
        elif self.tokenize_type == 'nltk':
            tokens = nltk.tokenize.word_tokenize(string)
        elif self.tokenize_type == 'wp':
            tokens = self.wptok.tokenize(string)
        elif self.tokenize_type == 'bpe':
            tokens = self.bpetok.tokenize(string)
        else:
            raise ValueError('Unknown tokenization type.')

        return tokens

    def count_ngrams(self, toks, n=3):
        ngram_counts = [Counter() for _ in range(n)]

        for i in range(len(toks)):
            for j in range(n):
                if i - j >= 0:
                    ngram = tuple([toks[k] for k in range(i - j, i + 1)])
                    #ngram = None  # TODO: define
                    ngram_counts[j][ngram] += 1

        return ngram_counts

    def threshold(self, toks, vocab, unk_symbol):
        V = set(vocab)
        assert unk_symbol not in V
        return [w if w in V else unk_symbol for w in toks]
