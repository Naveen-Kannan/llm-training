import hashlib
import os
from pathlib import Path
import unicodedata
import re
import mmh3
import random
from collections import defaultdict, deque


def dedup(input_file_dirs, output_dir):
    hashes = set()
    non_unique_hashes = set()
    for input_file_dir in input_file_dirs:
        with open(input_file_dir, "r") as f:
            for line in f:
                # get hash of line
                hash = hashlib.sha256(line.encode()).digest()
                if hash in hashes:
                    non_unique_hashes.add(hash)
                else:
                    hashes.add(hash)

    os.makedirs(output_dir, exist_ok=True)
    for input_file_dir in input_file_dirs:
        out_path = Path(output_dir) / Path(input_file_dir).name
        with open(input_file_dir, "r") as f:
            with open(out_path, "a") as fo:
                for line in f:
                    hash = hashlib.sha256(line.encode()).digest()
                    if hash not in non_unique_hashes:
                        fo.write(line)


def minhash_dedup(
    input_file_dirs: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_dir: os.PathLike,
):
    # init
    file_to_sig = dict()

    for input_file_dir in input_file_dirs:
        with open(input_file_dir, "r") as f:
            # get text
            text = f.read()
            # lowercase text
            text = text.lower()  # applyis to all
            # normalized text unicode NFD.
            text = unicodedata.normalize("NFD", text)
            # remove punc
            filtered_text = ""
            for char in text:
                if unicodedata.category(char)[0] != "P":
                    if unicodedata.category(char) != "Mn":
                        filtered_text += char

            # normalize spaces?
            normalized_text = re.sub(r"\s+", " ", filtered_text)

            normalized_words = normalized_text.split()
            # compute signature
            signature = []
            for i in range(num_hashes):
                minhash = float("inf")
                for j in range(len(normalized_words) - ngrams + 1):
                    ngram = "".join(normalized_words[j : j + ngrams])
                    minhash = min(minhash, mmh3.hash(ngram, seed=i))
                signature.append(minhash)

            file_to_sig[input_file_dir] = signature

    band_to_candidates = defaultdict(set)
    # lsh on bands
    for i in range(num_bands):
        for file_dir, signature in file_to_sig.items():
            idx = num_hashes // num_bands
            band = tuple(signature[i * idx : (i + 1) * idx])
            band_to_candidates[(i, band)].add(file_dir)

    adj = defaultdict(set)
    get_ngrams = defaultdict(set)
    for band, candidates in band_to_candidates.items():
        # note: change to be true jaccard not appx
        i, band = band
        if len(candidates) == 1:
            continue

        for candidate in candidates:
            with open(candidate, "r") as f:
                # get text
                text = f.read()
                # lowercase text
                text = text.lower()  # applyis to all
                # normalized text unicode NFD.
                text = unicodedata.normalize("NFD", text)
                # remove punc
                filtered_text = ""
                for char in text:
                    if unicodedata.category(char)[0] != "P":
                        if unicodedata.category(char) != "Mn":
                            filtered_text += char

                # normalize spaces?
                normalized_text = re.sub(r"\s+", " ", filtered_text)

                normalized_words = normalized_text.split()

                for j in range(len(normalized_words) - ngrams + 1):
                    ngram = "".join(normalized_words[j : j + ngrams])
                    get_ngrams[candidate].add(ngram)

        for candidate1 in candidates:
            for candidate2 in candidates:
                if candidate1 == candidate2:
                    continue

                jacc = len(get_ngrams[candidate1].intersection(get_ngrams[candidate2])) / len(
                    get_ngrams[candidate1].union(get_ngrams[candidate2])
                )

                # # compute jaccard similarity
                # this way was way nicer imo
                # sig1, sig2 = file_to_sig[candidate1], file_to_sig[candidate2]
                # same = 0
                # for i in range(len(sig1)):
                #     if sig1[i] == sig2[i]:
                #         same += 1
                # jacc = same / len(sig1)
                if jacc >= jaccard_threshold:
                    adj[candidate1].add(candidate2)
                    adj[candidate2].add(candidate1)

        # i, band = band
        # if len(candidates) == 1:
        #     continue

        # for candidate1 in candidates:
        #     for candidate2 in candidates:
        #         if candidate1 == candidate2:
        #             continue

        #         # compute jaccard similarity
        #         sig1, sig2 = file_to_sig[candidate1], file_to_sig[candidate2]
        #         same = 0
        #         for i in range(len(sig1)):
        #             if sig1[i] == sig2[i]:
        #                 same += 1
        #         jacc = same / len(sig1)
        #         if jacc >= jaccard_threshold:
        #             adj[candidate1].add(candidate2)
        #             adj[candidate2].add(candidate1)

    # calm lil bfs
    final_docs = []
    visited = set()
    for file in input_file_dirs:
        if file in visited:
            continue

        cluster = [file]
        next_nodes = deque()
        visited.add(file)
        for node in adj[file]:
            next_nodes.append(node)

        while next_nodes:
            curr = next_nodes.popleft()
            if curr in visited:
                continue

            cluster.append(curr)
            visited.add(curr)
            for node in adj[curr]:
                next_nodes.append(node)
        # add one random doc to final_docs
        final_docs.append(random.choice(cluster))

    os.makedirs(output_dir, exist_ok=True)

    for doc in final_docs:
        out_path = Path(output_dir) / Path(doc).name
        with open(doc, "r") as f:
            with open(out_path, "a") as fo:
                for line in f:
                    fo.write(line)
