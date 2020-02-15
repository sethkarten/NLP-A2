# Instructor: Karl Stratos

import argparse
import util
from scipy.optimize import minimize_scalar
import numpy as np

best_perp = float('-inf')
best_lr = None

def main(args):
    tokenizer = util.Tokenizer(tokenize_type=args.tok, lowercase=True)

    train_toks = tokenizer.tokenize(open(args.train_file).read())
    num_train_toks = int(args.train_fraction * len(train_toks))
    print('-' * 79)
    print('Using %d tokens for training (%g%% of %d)' %
          (num_train_toks, 100 * args.train_fraction, len(train_toks)))
    train_toks = train_toks[:int(args.train_fraction * len(train_toks))]
    val_toks = tokenizer.tokenize(open(args.val_file).read())
    num_val_toks = int(args.val_fraction * len(val_toks))
    print('Using %d tokens for validation (%g%% of %d)' %
          (num_val_toks, 100 * args.val_fraction, len(val_toks)))
    val_toks = val_toks[:int(args.val_fraction * len(val_toks))]

    train_ngram_counts = tokenizer.count_ngrams(train_toks)


    # Get vocab and threshold.
    print('Using vocab size %d (excluding UNK) (original %d)' %
          (min(args.vocab, len(train_ngram_counts[0])),
           len(train_ngram_counts[0])))
    vocab = [tup[0] for tup, _ in train_ngram_counts[0].most_common(args.vocab)]
    train_toks = tokenizer.threshold(train_toks, vocab, args.unk)
    val_toks = tokenizer.threshold(val_toks, vocab, args.unk)


    if args.features == 'basic1':
        feature_extractor = util.basic_features1
    elif args.features == 'basic1suffix3':
        feature_extractor = util.basic_features1_suffix3  # TODO: Implement
    elif args.features == 'basic2':
        feature_extractor = util.basic_features2
    else:
        raise ValueError('Unknown feature extractor type.')

    # We'll cheat and cache features for validation data to make things faster
    # for this assignment. The correct thing to do here would be
    #
    #f2i, fcache, num_feats_cached, x2ys \
    #      util.extract_features(train_toks, feature_extractor)
    #
    f2i, fcache, num_feats_cached, x2ys \
        = util.extract_features(train_toks + val_toks, feature_extractor)

    print('%d feature types extracted' % len(f2i))
    print('%d feature values cached for %d window types' %
          (num_feats_cached, len(fcache)))

    m = Train(args, vocab, feature_extractor, f2i, fcache, x2ys, train_toks, val_toks)
    out = minimize_scalar(m.train, bounds=(0.5,1), bracket=[0.5,1])
    print(out)
    # lower = 0.5, upper = 1
    # while True:

    # val_ppl = lr, args.seed)
    # if val_ppl < best_perp:
    #     best_perp = val_ppl
    #     best_lr = lr

class Train:
    def __init__(self, args, vocab, feature_extractor, f2i, fcache, x2ys, train_toks, val_toks):
        self.args = args
        self.vocab = vocab
        self.feature_extractor = feature_extractor
        self.f2i = f2i
        self.fcache = fcache
        self.x2ys = x2ys
        self.train_toks = train_toks
        self.val_toks = val_toks

    def train(self, lr):
        print(lr)
        # The language model assumes a truncated vocab and a feature definition.
        lm = util.LogLinearLanguageModel(self.args.model, self.vocab, self.args.unk,
                                         self.feature_extractor, self.f2i, self.fcache, self.x2ys,
                                         init=self.args.init, lr=lr,
                                         check_interval=self.args.check_interval,
                                         seed=self.args.seed)

        lm.train(self.train_toks, self.val_toks, self.args.epochs)

        val_ppl = lm.test(self.val_toks)
        # print('Optimized Perplexity: %f' %(val_ppl))
        return val_ppl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.npy',
                        help='model file [%(default)s]')
    parser.add_argument('--test', action='store_true',
                        help='do not train, load trained model?')
    parser.add_argument('--train_file', type=str,
                        default='data/gigaword_subset.train',
                        help='corpus for training [%(default)s]')
    train_file = 'data/gigaword_subset.train'
    parser.add_argument('--val_file', type=str,
                        default='data/gigaword_subset.val',
                        help='corpus for validation [%(default)s]')
    parser.add_argument('--features', type=str, default='basic1',
                        help='feature extractor type [%(default)s]')
    parser.add_argument('--init', type=float, default=0.001,
                        help='init range [%(default)g]')
    parser.add_argument('--lr', type=float, default=0.5,
                        help='learning rate [%(default)g]')
    parser.add_argument('--tok', type=str, default='nltk',
                        choices=['basic', 'nltk', 'wp', 'bpe'],
                        help='tokenizer type [%(default)s]')
    parser.add_argument('--vocab', type=int, default=10000,
                        help='max vocab size [%(default)d]')
    parser.add_argument('--train_fraction', type=float, default=0.1,
                        help='use this fraction of training data [%(default)g]')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='use this fraction of val data [%(default)g]')
    parser.add_argument('--K', type=int, default=10,
                        help='K in top K [%(default)d]')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs [%(default)d]')
    parser.add_argument('--check_interval', type=int, default=10000,
                        metavar='CH',
                        help='number of updates for a check [%(default)d]')
    parser.add_argument('--unk', type=str, default='<?>',
                        help='unknown token symbol [%(default)s]')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [%(default)d]')
    args = parser.parse_args()
    main(args)
