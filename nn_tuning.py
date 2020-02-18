from assignment2_nlm import *
import math
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import util

from collections import OrderedDict

def main(args):
    tokenizer = util.Tokenizer(tokenize_type='nltk', lowercase=True)

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

    best_val_ppl0 = float("inf")
    best_val_ppl1 = float("inf")
    best_lr0 = -1
    best_lr1 = -1
    for nlayers in [0,1]:
        for dim in [1,5,10,100,200]:
            for lr in [0.00001,0.00003,0.0001,0.0003,0.001]:
                lm = FFLM(args.model, vocab, args.unk, args.init, lr,
                    args.check_interval, args.seed, args.nhis, dim, dim,
                    nlayers, 16)

                lm.train_epochs(train_toks, val_toks, 10)
                val_ppl = lm.best_val_ppl
                if val_ppl < best_val_ppl0 and nlayers == 0:
                    best_val_ppl0, best_lr0 = val_ppl, lr
                elif val_ppl < best_val_ppl1 and nlayers == 1:
                    best_val_ppl1, best_lr1 = val_ppl, lr
                print(str(dim), str(lr), str(val_ppl))
                with open("nn_data"+str(nlayers)+"C.txt", "a+") as f:
                    f.write(str(dim)+","+str(lr)+","+str(val_ppl)+"\n")
    epochs = 1000
    dim = 30
    batch_size = 16
    for nlayers, lr in zip([0,1], [best_lr0, best_lr1]):
        lm = FFLM(args.model, vocab, args.unk, args.init, lr,
            args.check_interval, args.seed, args.nhis, dim, dim,
            nlayers, batch_size)

        lm.train_epochs(train_toks, val_toks, epochs)
        val_ppl = lm.best_val_ppl
        print(str(dim), str(lr), str(val_ppl))
        with open("nn_data_cmpC.txt", "a+") as f:
            f.write(str(nlayers)+","+str(lr)+","+str(val_ppl)+"\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.pt',
                        help='model file [%(default)s]')
    parser.add_argument('--test', action='store_true',
                        help='do not train, load trained model?')
    parser.add_argument('--train_file', type=str,
                        default='data/gigaword_subset.val',
                        help='corpus for training [%(default)s]')
    parser.add_argument('--val_file', type=str,
                        default='data/gigaword_subset.val',
                        help='corpus for validation [%(default)s]')
    parser.add_argument('--init', type=float, default=0.0,
                        help='init range (default if 0) [%(default)g]')
    parser.add_argument('--lr', type=float, default=0.00003,
                        help='learning rate [%(default)g]')
    parser.add_argument('--vocab', type=int, default=1000,
                        help='max vocab size [%(default)d]')
    parser.add_argument('--nhis', type=int, default=3,
                        help='number of previous words to condition on '
                        '[%(default)d]')
    parser.add_argument('--wdim', type=int, default=30,
                        help='word embedding dimension [%(default)d]')
    parser.add_argument('--hdim', type=int, default=30,
                        help='hidden state dimension [%(default)d]')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers [%(default)d]')
    parser.add_argument('--B', type=int, default=16,
                        help='batch size [%(default)d]')
    parser.add_argument('--train_fraction', type=float, default=0.1,
                        help='use this fraction of training data [%(default)g]')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='use this fraction of val data [%(default)g]')
    parser.add_argument('--K', type=int, default=10,
                        help='K in top K [%(default)d]')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs [%(default)d]')
    parser.add_argument('--check_interval', type=int, default=2000,
                        metavar='CH',
                        help='number of updates for a check [%(default)d]')
    parser.add_argument('--unk', type=str, default='<?>',
                        help='unknown token symbol [%(default)s]')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [%(default)d]')
    args = parser.parse_args()
    main(args)
