# turboBPE

A fast, minimal implementation of the Byte Pair Encoding (BPE) algorithm used in LLM tokenization built on top of [Andrej Karpathy's minbpe](https://github.com/karpathy/minbpe), with a C-accelerated backend that makes training absurdly fast.

The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings. It was popularized for LLMs by the [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and the associated [code release](https://github.com/openai/gpt-2) from OpenAI. Today, every major LLM (GPT, Llama, Mistral, etc.) uses some form of this algorithm to train its tokenizer.

turboBPE keeps the same clean, hackable spirit of minbpe while replacing the inner training and encoding loops with a compiled C extension giving you the readability you want and the speed you need.

The core idea: instead of minting one token per iteration and recomputing stats from scratch each time, turbobpe computes stats **once**, picks the top-N non-overlapping pairs, and merges them all in a single pass. Same correctness guarantees, significantly fewer stat sweeps.

---

## Why turboBPE?

Because training a 10K vocab tokenizer on a 4 MB corpus in **~10 seconds** instead of **~6 hours** changes what's practical.

Here's what that looks like on real benchmarks (run on my development machine your numbers will vary, but the order of magnitude holds):

### Training

| Dataset | Vocab | minbpe | turboBPE | Speedup |
|---|---|---|---|---|
| `taylorswift.txt` (~182 KB) | 10K | ~782 sec | ~1.3 sec | **~600×** |
| 4 MB corpus | 10K | ~22,220 sec (~6h 10m) | ~10.5 sec | **~2,100×** |

The speedup grows as vocab size increases — the larger the merge table, the more turboBPE pulls ahead.

### Tokenization (Encoding)

| Dataset | minbpe | turboBPE | Speedup |
|---|---|---|---|
| `taylorswift.txt` | ~1.07 sec | ~0.085 sec | **~12×** |
| 4 MB corpus | ~29 sec | ~1.8 sec | **~16×** |

---

## How does it work?

The high-level algorithm is identical to minbpe: repeatedly find the most frequent adjacent token pair across the corpus and merge it into a new token, until you hit your target vocabulary size.

The key innovation in turboBPE is **batch merging**, controlled by the `batch_size` parameter. Classical BPE performs exactly one merge per training round find the top pair, merge it everywhere, rescan, repeat. That rescan is expensive. turboBPE instead picks the top-N pairs per round (where N is `batch_size`) and merges all of them in a single pass, as long as they don't form a chain conflict. A chain conflict is when two pairs share a token boundary for example, merging `(A, B)` and `(B, C)` in the same pass is unsafe because the first merge consumes the `B` that the second merge depends on. turboBPE detects and resolves these conflicts automatically, keeping only the safe subset for each batch.

This means with `batch_size=10`, you're doing roughly 10× fewer full passes over the corpus, and the C-accelerated backend maintains running pair statistics incrementally so counts stay accurate across the batch without a full rescan.

*Fun fact:* Theoretically, `batch_size` can be set to anything. `find_overlapping_cases` + `filter_top_pairs` will always carve out a safe subset regardless of how many candidates you throw at it the overlap filter is the safety net, not the batch size.

> `get_stats` is the bottleneck of BPE training. Every merge you squeeze into a single stat sweep is a sweep you don't pay for.

The encoding path uses a linked-list structure under the hood for in-place merges, avoiding the repeated list copies that make naive BPE encoding slow.

The Python-facing API is unchanged from minbpe you don't need to think about any of this to use it. But `batch_size` is there if you want to tune.

---

## Installation

```bash
pip install turbobpe
```

---

## Quick Start

```python
from turbobpe import RegexTokenizer

tokenizer = RegexTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)

tokenizer.encode("hello world")        # string -> list of token ids
tokenizer.decode([1000, 2000, 3000])   # list of token ids -> string

tokenizer.save("tok32k")               # writes tok32k.model and tok32k.vocab
tokenizer.load("tok32k.model")         # load it back later
```

The `.model` file is what you need for loading. The `.vocab` file is a human-readable view of what each token looks like useful for debugging and visualization, not for loading.

### Tuning `batch_size`

`batch_size` controls how many token pairs are merged per training round. The default is `10`, which is a good starting point for most use cases.

```python
tokenizer.train(very_long_training_string, vocab_size=32768, batch_size=10)
```

A higher `batch_size` means fewer training rounds and faster wall-clock time, but each batch may be doing slightly more approximate work pairs that appear in the same round aren't strictly ordered against each other the way they would be in classical single-merge BPE. In practice the resulting vocabulary is nearly identical and the tradeoff is almost always worth it. If you need the most conservative, merge-order-faithful behavior possible, set `batch_size=1` to fall back to classical one-merge-per-round BPE (and accept the speed penalty).

---

## Special Tokens

Register special tokens after training. If your vocab size is 32768, the last merge token has id 32767, so your first special token should be 32768:

```python
from turbobpe import RegexTokenizer

tokenizer = RegexTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)
tokenizer.register_special_tokens({"<|endoftext|>": 32768})
tokenizer.encode("<|endoftext|>hello world", allowed_special="all")
```

The `allowed_special` parameter follows tiktoken convention: `"all"`, `"none"`, `"none_raise"` (default raises if a special token appears in the text), or a custom set. This default is intentional — silently tokenizing special tokens in attacker-controlled input is a footgun.

---

## Comparison with Other Tokenizers

### vs. minbpe (Python)

minbpe is the clean, educational reference implementation this project builds on. If you want to understand BPE from scratch, read minbpe first. turboBPE is what you reach for when you actually want to train on non-trivial data. The API is nearly identical, so switching is a one-line import change. The tradeoff: turboBPE requires a compiled C extension, so `pip install turbobpe` needs a working C compiler at build time (or a compatible pre-built wheel).

### vs. tiktoken (OpenAI)

tiktoken is OpenAI's production tokenizer, written in Rust, and it is fast encoding is very competitive. The key difference is that **tiktoken does not support training**. It ships with pre-trained vocabularies (GPT-2, GPT-4, etc.) and is designed for inference only. turboBPE lets you train your own tokenizer from scratch on your own data, which tiktoken simply doesn't do. If you need GPT-4 compatible tokenization and don't care about training, use tiktoken. If you're building your own model or need a custom vocab, turboBPE is the tool.

### vs. Hugging Face Tokenizers

The Hugging Face `tokenizers` library is a production-grade, Rust-backed tokenizer with broad format support, padding, truncation, and tight integration with the `transformers` ecosystem. It is fast and full-featured. If you're building a production NLP pipeline on top of HuggingFace models, their tokenizer is probably the right choice it has years of engineering behind it and handles edge cases you haven't thought of yet. turboBPE is leaner and more direct: it does BPE, it does it fast, and it stays out of your way. If you want to understand what's happening, extend the code, or train a tokenizer outside the HuggingFace ecosystem, turboBPE is a much simpler entry point. The HuggingFace library is also not trivially auditable turboBPE's Python layer is a few hundred lines of clean code you can read in an afternoon.

**Summary table:**

| | turboBPE | minbpe | tiktoken | HuggingFace |
|---|:---:|:---:|:---:|:---:|
| Training | ✅ Fast | ✅ Slow | ❌ | ✅ Fast |
| Encoding | ✅ Fast | ⚠️ Slow | ✅ Fast | ✅ Fast |
| Custom vocab | ✅ | ✅ | ❌ | ✅ |
| Special tokens | ✅ | ✅ | ✅ | ✅ |
| Readable codebase | ✅ | ✅ | ⚠️ | ⚠️ |
| HF ecosystem | ❌ | ❌ | ❌ | ✅ |
| Requires C build | ✅ | ❌ | ❌ (Rust) | ❌ (Rust) |

---

## Acknowledgements

This project would not exist without [Andrej Karpathy's minbpe](https://github.com/karpathy/minbpe), which is the clearest exposition of BPE I've come across. The stats idea was shaped in part by the [fast_minbpe](https://github.com/yanivle/fast_minbpe). Both are worth reading.

---

## License

MIT

---

*Train faster, scale better. Build your own tokenizer.*