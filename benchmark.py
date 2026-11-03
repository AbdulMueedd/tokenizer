"""Benchmark the BPE tokenizer across merge counts and training corpus sizes.

Usage:
    python benchmark.py CORPUS_FILE TEST_FILE

Slices CORPUS_FILE into training sets of 500/1000/1500/2000 words, runs each
against K = 50/100/150/200, averages the timings over several runs, and writes
three comparison plots.
"""

import sys
import time

import matplotlib.pyplot as plt

import tokenizer as bpe

KS = [50, 100, 150, 200]
SIZES = [500, 1000, 1500, 2000]
RUNS = 5


def main():
    if len(sys.argv) != 3:
        print("Usage: python benchmark.py CORPUS_FILE TEST_FILE")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8", errors="ignore") as f:
        corpus = bpe.preprocess(f.read())
    with open(sys.argv[2], encoding="utf-8", errors="ignore") as f:
        test = bpe.preprocess(f.read())

    train_t, tok_t, ntok = {}, {}, {}

    for n in SIZES:
        words = corpus[:n]
        for k in KS:
            total_train = total_tok = 0.0
            for _ in range(RUNS):
                start = time.perf_counter()
                vocab, merges = bpe.bpe_learn(words, k)
                total_train += time.perf_counter() - start

                start = time.perf_counter()
                tokens = bpe.bpe_segment(test, merges)
                total_tok += time.perf_counter() - start

            train_t[n, k] = total_train / RUNS
            tok_t[n, k] = total_tok / RUNS
            ntok[n, k] = len(tokens)

            print("size=%d K=%d train=%.4fs tokenize=%.4fs test_tokens=%d"
                  % (n, k, train_t[n, k], tok_t[n, k], len(tokens)))
            print("   last 10 merges:", vocab[-10:])
            print("   sample:", " ".join(tokens[:20]))

    plots = (
        (train_t, "Training time (s)", "plot_training_time.png"),
        (tok_t, "Tokenization time (s)", "plot_tokenization_time.png"),
        (ntok, "Tokens in test output", "plot_token_count.png"),
    )
    for data, ylabel, filename in plots:
        plt.figure()
        for n in SIZES:
            plt.plot(KS, [data[n, k] for k in KS], marker="o", label="%d words" % n)
        plt.xlabel("K (merges)")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.savefig(filename, dpi=150)


if __name__ == "__main__":
    main()
