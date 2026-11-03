# Byte Pair Encoding Tokenizer

A from-scratch implementation of the Byte Pair Encoding (BPE) subword tokenization
algorithm, including both the **Learner** (vocabulary induction) and the
**Segmenter** (applying a learned vocabulary to new text).

No tokenization libraries are used. The only dependencies are the Python standard
library (`sys`, `os`, `re`, `time`), plus `matplotlib` for the optional benchmark
plots.

## Background

BPE builds a subword vocabulary from raw text without any linguistic knowledge.
Starting from individual characters, it repeatedly finds the most frequent adjacent
symbol pair in the corpus and merges it into a single new symbol. After K merges,
the vocabulary contains the original characters plus K learned multi-character
tokens.

The segmenter then replays those merges, **in the order they were learned**, against
unseen text. Order matters: BPE is not longest-match lookup against the vocabulary,
and replaying merges out of order produces different segmentations.

A stop token (`_`) is appended to every word so that word-final subwords stay
distinct from word-internal ones. This is what lets the model learn `the_` as a
unit separate from the letters `t`, `h`, `e`.

## Usage

```
python tokenizer.py K TRAIN_FILE TEST_FILE
```

- `K` is the number of merges to perform (a positive integer).
- `TRAIN_FILE` is the corpus the learner induces its vocabulary from.
- `TEST_FILE` is the text the trained segmenter tokenizes.

Both input files must live in the same directory as the script. If `K` is not a
positive integer it falls back to 5.

Example:

```
python tokenizer.py 100 train.txt test.txt
```

### Output

Printed to stdout: the merge count, input filenames, training time, tokenization
time, and the first 20 tokens of the result.

Written to disk:

- `vocab.txt` — the final vocabulary, one token per line, in merge order
- `result.txt` — the full tokenized test text, space separated

### Preprocessing

Each input file is treated as a single sentence. Everything outside the initial
vocabulary (A-Z, a-z, space) is stripped, including punctuation and digits, and
whitespace is collapsed before splitting on spaces.

## Benchmarks

`benchmark.py` sweeps K across 50/100/150/200 and training corpus size across
500/1000/1500/2000 words, averaging 5 runs per configuration and emitting three
plots.

```
python benchmark.py CORPUS_FILE TEST_FILE
```

## Findings

Measured on *Alice's Adventures in Wonderland* as the training corpus, with an
unrelated passage from *Pride and Prejudice* as the test text.

**Training cost is linear in both K and corpus size.** At 2000 training words,
training goes from 0.023s at K=50 to 0.081s at K=200. Each merge requires a full
pass recounting every adjacent pair in the corpus, so cost is roughly
O(K x corpus).

**Segmentation cost depends on K and not on corpus size.** All four corpus sizes
produce nearly identical segmentation times, from 0.0013s at K=50 to 0.0039s at
K=200. The segmenter replays K merges per unique word; how much data those merges
were learned from is irrelevant once training is done.

**Compression improves with K, with diminishing returns.** Test output falls from
638 tokens at K=50 to 482 at K=200 for a 2000-word training corpus, and the curve
is visibly flattening. Early merges capture the highest frequency pairs, so each
additional merge buys less than the one before it.

**Merges recover morphology in a predictable order.** At K=50 the vocabulary is
almost entirely frequent bigrams: `th`, `in`, `er`, `ou`, `el`, `ur`. By K=100,
whole function words appear with the stop token attached (`the_`, `of_`, `and_`,
`that_`, `not_`, `had_`, `but_`) alongside genuine suffixes (`ing_`, `ed_`, `er_`,
`es_`, `ly_`). By K=150 to 200 content words and prefixes emerge: `nothing_`,
`thought_`, `little_`, `con`, `beg`. None of this is supplied to the algorithm;
it falls out of frequency alone.

**The vocabulary is a fingerprint of the training corpus, not of English.**
`Alice_` is a learned token by K=100. Training on *Pride and Prejudice* instead
yields `Mr_`, `Miss_`, and `Aust`. This shows up clearly at segmentation time:
tokenizing Austen text with an Alice-trained vocabulary leaves frequent function
words intact while rare words shatter toward individual characters, for example
`M ic ha el m as_` and `N e th er f i el d_`.

**More training data generalizes better at the same K.** At K=50, a 2000-word
corpus yields 638 test tokens against 676 for a 500-word corpus. Merges learned
from more text transfer more reliably to unseen input.

A practical note: source text must be genuinely plain. An early run used a file
saved as RTF with a `.txt` extension, and the markup control words were frequent
enough to win merge slots outright, producing junk tokens like `xtxt`. Boilerplate
has the same effect: Project Gutenberg's license header sat inside the smaller
training slices and spent merges on `Gutenberg_`, `eBook_`, and `States_`.
