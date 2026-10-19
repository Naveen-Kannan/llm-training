import pickle
import regex as re

# pre-token
PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges

        self.inverse_vocab = {v: k for k, v in vocab.items()}

        # merge pair -> priority rank
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}

        self.special_tokens = special_tokens or []

        # # add special tokens to vocab if missing
        # jk not needed

        self.special_pat = None

        # regex to split on special tokens
        if self.special_tokens:
            st = sorted(self.special_tokens, key=len, reverse=True)
            pat = "|".join(re.escape(s) for s in st)
            self.special_pat = re.compile(f"({pat})")

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, tokens):
        # repeatedly merge
        while len(tokens) > 1:
            best_pair = None
            best_rank = float("inf")  # golf

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:  # fully tokenized
                break

            # merge, similar to train_bpe.py
            a, b = best_pair
            merged = a + b
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def _encode_chunk(self, text):
        # pre-tokenize
        ids = []
        pretokens = re.findall(PAT_STR, text)

        for word in pretokens:
            # can optimize encode/decode?
            tokens = [bytes([b]) for b in word.encode("utf-8")]

            merged_tokens = self._apply_merges(tokens)
            ids.extend(self.inverse_vocab[t] for t in merged_tokens)

        return ids

    def encode(self, text):
        # still fully serial ... should be fine
        if self.special_pat is None:
            return self._encode_chunk(text)

        ids = []
        file_chunks = self.special_pat.split(text)
        special_set = set(self.special_tokens)
        for part in file_chunks:
            if not part:
                continue

            # can optimize the encode/decode?
            if part in special_set:
                ids.append(self.inverse_vocab[part.encode("utf-8")])
            else:
                ids.extend(self._encode_chunk(part))

        return ids

    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids):
        raw = b"".join(self.vocab[id] for id in ids)
        return raw.decode("utf-8", errors="replace")
