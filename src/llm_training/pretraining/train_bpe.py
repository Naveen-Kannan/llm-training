import os
import regex as re
from collections import defaultdict
from .pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool


def _words_from_corpus(txt: str, special_tokens: list[str]) -> list[str]:
    # split on special tokens
    split_pat = "|".join(re.escape(s) for s in special_tokens)
    split_pat = f"({split_pat})"

    split_corpus = re.split(split_pat, txt)

    # pre-tokenize regex
    pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    words = []
    special_set = set(special_tokens)
    for corpus in split_corpus:
        if not corpus or corpus in special_set:
            continue
        words.extend(re.findall(pat_str, corpus))
    return words


def _get_word_freqs(words: list[str]) -> dict[tuple[bytes, ...], int]:
    # deduplicate words, track frequency
    word_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)
    for w in words:
        key = tuple(bytes([b]) for b in w.encode("utf-8"))
        word_freqs[key] += 1
    return dict(word_freqs)


def _init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    # vocab init, special tokens
    vocab = {i: bytes([i]) for i in range(256)}

    special_bytes = [s.encode("utf-8") for s in special_tokens]
    for i in range(len(special_bytes)):
        vocab[256 + i] = special_bytes[i]
    return vocab


def _get_pair_counts(word_freqs: dict[tuple[bytes, ...], int]):
    # pair counts weighted by word frequency; inverted index for merge updates
    pair_counts = defaultdict(int)
    pairs_to_keys = defaultdict(set)

    for seq, freq in word_freqs.items():
        for i in range(len(seq) - 1):
            p = (seq[i], seq[i + 1])
            pair_counts[p] += freq
            pairs_to_keys[p].add(seq)
    return pair_counts, pairs_to_keys


def _build_new_repr(seq, pair):
    a, b = pair
    out = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(seq[i] + seq[i + 1])
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)


def _get_best_pair(pair_counts):
    best_pair, best_val = None, None

    for pair, count in pair_counts.items():
        val = (count, pair)
        if best_val is None or val > best_val:
            best_val = val
            best_pair = pair

    return best_pair


def _apply_merge(word_freqs, pair_counts, pairs_to_keys, pair):
    keys_to_modify = pairs_to_keys[pair].copy()

    for old_key in keys_to_modify:
        old_freq = word_freqs.pop(old_key)
        new_key = _build_new_repr(old_key, pair)

        for i in range(len(old_key) - 1):
            left, right = old_key[i], old_key[i + 1]
            p = (left, right)
            pair_counts[p] -= old_freq
            if pair_counts[p] <= 0:
                del pair_counts[p]
            pairs_to_keys[p].discard(old_key)
            if not pairs_to_keys[p]:
                del pairs_to_keys[p]

        for i in range(len(new_key) - 1):
            left, right = new_key[i], new_key[i + 1]
            p = (left, right)
            pair_counts[p] += old_freq
            pairs_to_keys[p].add(new_key)

        word_freqs[new_key] = word_freqs.get(new_key, 0) + old_freq

    pairs_to_keys[pair] = set()


def _worker_chunk(input_path: str, special_tokens: list[str], start: int, end: int):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        txt = chunk_bytes.decode("utf-8")

        return _get_word_freqs(_words_from_corpus(txt, special_tokens))


# old func, before ram limit lol
def _merge_freq_dicts(freq_dicts):
    merged = defaultdict(int)
    for d in freq_dicts:
        for k, v in d.items():
            merged[k] += v
    return dict(merged)


def _worker_chunk_task(args: tuple[str, list[str], int, int]):
    return _worker_chunk(*args)


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    file_size = os.path.getsize(input_path)
    if file_size < 1_000_000:  # ~1 MB
        # serial
        with open(input_path, "r") as f:
            txt = f.read()
        words = _words_from_corpus(txt, special_tokens)
        word_freqs = _get_word_freqs(words)
    else:
        # parallelize; many chunks so peak RAM ~= n_workers * chunk_size, not full file
        end_of_text_bytes = special_tokens[0].encode("utf-8")
        n_cores = os.cpu_count() or 1
        n_chunks = max(n_cores * 40, 32)
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, n_chunks, end_of_text_bytes)
            boundaries = list(zip(boundaries[:-1], boundaries[1:]))

        # pre-token counts

        tasks = [(input_path, special_tokens, s, e) for s, e in boundaries]
        merged = defaultdict(int)
        with Pool(n_cores) as p:
            for d in p.imap_unordered(_worker_chunk_task, tasks, chunksize=1):
                for k, v in d.items():
                    merged[k] += v

        word_freqs = dict(merged)
    ########################################

    vocab = _init_vocab(special_tokens)
    merges = []
    pair_counts, pairs_to_keys = _get_pair_counts(word_freqs)
    token_count = len(vocab)
    while token_count < vocab_size:
        if not pair_counts:
            break
        best_pair = _get_best_pair(pair_counts)

        new_token = best_pair[0] + best_pair[1]

        vocab[token_count] = new_token
        token_count += 1

        merges.append(best_pair)

        _apply_merge(word_freqs, pair_counts, pairs_to_keys, best_pair)

    return vocab, merges
