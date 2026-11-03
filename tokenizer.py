"""Byte Pair Encoding tokenizer: learner and segmenter, implemented from scratch.

Usage:
    python tokenizer.py K TRAIN_FILE TEST_FILE
"""

import sys
import os
import re
import time

STOP = "_"                       # stop (end of word) token
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + \
          [chr(c) for c in range(ord("a"), ord("z") + 1)]
DEFAULT_K = 5

VOCAB_FILE = "vocab.txt"
RESULT_FILE = "result.txt"


def preprocess(text):
    """Treat the file as a single sentence, keep only letters of V, split on spaces."""
    text = re.sub(r"\s+", " ", text)          # newlines, tabs etc. become a space
    text = re.sub(r"[^A-Za-z ]", "", text)    # drop punctuation, digits, anything outside V
    return text.split()


def merge_symbols(symbols, a, b):
    """Replace every adjacent (a, b) in a tuple of symbols with the merged symbol a+b."""
    out = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def bpe_learn(words, k):
    """BPE learner. Returns (final vocabulary list, ordered list of merges)."""
    # Corpus as a dictionary: word (tuple of symbols ending with STOP) -> frequency
    corpus = {}
    for w in words:
        key = tuple(w) + (STOP,)
        corpus[key] = corpus.get(key, 0) + 1

    vocab = LETTERS + [STOP]
    merges = []

    for _ in range(k):
        # Count all adjacent symbol pairs, weighted by word frequency
        pair_counts = {}
        for symbols, freq in corpus.items():
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
        if not pair_counts:
            break                              # every word is already a single token

        # Most frequent pair (ties: the pair seen first in the corpus wins)
        best, best_count = None, 0
        for pair, count in pair_counts.items():
            if count > best_count:
                best, best_count = pair, count

        a, b = best
        merges.append(best)
        vocab.append(a + b)

        # Apply the merge everywhere in the corpus
        new_corpus = {}
        for symbols, freq in corpus.items():
            merged = merge_symbols(symbols, a, b)
            new_corpus[merged] = new_corpus.get(merged, 0) + freq
        corpus = new_corpus

    return vocab, merges


def bpe_segment(words, merges):
    """BPE segmenter. Replays the learned merges, in order, on each word."""
    cache = {}
    tokens = []
    for w in words:
        if w not in cache:
            symbols = tuple(w) + (STOP,)
            for a, b in merges:
                if len(symbols) == 1:
                    break
                symbols = merge_symbols(symbols, a, b)
            cache[w] = symbols
        tokens.extend(cache[w])
    return tokens


def main():
    if len(sys.argv) != 4:
        print("ERROR: Not enough/too many/illegal input arguments.")
        print("Usage: python " + os.path.basename(sys.argv[0]) + " K TRAIN_FILE TEST_FILE")
        sys.exit(1)

    k_arg, train_name, test_name = sys.argv[1], sys.argv[2], sys.argv[3]

    folder = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(folder, train_name)
    test_path = os.path.join(folder, test_name)
    for name, path in ((train_name, train_path), (test_name, test_path)):
        if not os.path.isfile(path):
            print("ERROR: File " + name + " does not exist in the program folder.")
            sys.exit(1)

    try:
        k = int(k_arg)
        if k < 1:
            k = DEFAULT_K
    except ValueError:
        k = DEFAULT_K

    with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
        train_words = preprocess(f.read())
    with open(test_path, "r", encoding="utf-8", errors="ignore") as f:
        test_words = preprocess(f.read())

    start = time.perf_counter()
    vocab, merges = bpe_learn(train_words, k)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    tokens = bpe_segment(test_words, merges)
    token_time = time.perf_counter() - start

    with open(os.path.join(folder, VOCAB_FILE), "w", encoding="utf-8") as f:
        f.write("\n".join(vocab) + "\n")
    with open(os.path.join(folder, RESULT_FILE), "w", encoding="utf-8") as f:
        f.write(" ".join(tokens) + "\n")

    print("Number of merges: " + str(k))
    print("Training file name: " + train_name)
    print("Test file name: " + test_name)
    print("Training time: " + str(train_time) + " seconds")
    print("Tokenization time: " + str(token_time) + " seconds")
    print("Tokenization result: " + " ".join(tokens[:20]))
    if len(tokens) > 20:
        print("Tokenized text is longer than 20 tokens")


if __name__ == "__main__":
    main()
