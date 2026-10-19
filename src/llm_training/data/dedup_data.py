import concurrent.futures
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import hashlib
import unicodedata
import re
import mmh3
import random
from collections import defaultdict, deque


def dedup_lines(lines):
    hashes = set()
    non_unique_hashes = set()
    # go by lines
    for line in lines:
        # get hash of line
        hash = hashlib.sha256(line.encode()).digest()
        if hash in hashes:
            non_unique_hashes.add(hash)
        else:
            hashes.add(hash)

    out_lines = []
    for line in lines:
        hash = hashlib.sha256(line.encode()).digest()
        if hash not in non_unique_hashes:
            out_lines.append(line)
    return out_lines


def minhash_dedup_lines(
    lines: list[str],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
):
    # init
    file_to_sig = dict()
    # go by lines
    for file_dir, text in enumerate(lines):
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

        file_to_sig[file_dir] = signature

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
            # get text
            text = lines[candidate]
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
                if jacc >= jaccard_threshold:
                    adj[candidate1].add(candidate2)
                    adj[candidate2].add(candidate1)

    # calm lil bfs
    final_docs = []
    visited = set()
    for file in range(len(lines)):
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

    # assemble list
    return [lines[doc] for doc in final_docs]


# def process_single_file(in_path: str, out_path: str):
#     name = Path(in_path).stem
#     work = Path("data/tmp_dedup") / name
#     sd = work / "split"
#     ed = work / "exact"
#     md = work / "minhash"
#     sd.mkdir(parents=True)

#     # split into files
#     with open(in_path) as fi:
#         for i, line in enumerate(fi):
#             line = line.strip()
#             if not line: # should never happen
#                 continue
#             with open(sd / f"{i}.txt", "w") as fo:
#                 fo.write(line)

#     # hav eto ssemble lists
#     sf = [sd / p for p in os.listdir(sd)]
#     # exact line within file
#     dedup(sf, str(ed))

#     ef = [ed / p for p in os.listdir(ed)]
#     # bands = 5
#     minhash_dedup(ef, num_hashes=20, num_bands=4, ngrams=5, jaccard_threshold=0.8, output_dir=str(md))

#     with open(out_path, "w") as fo:
#         for p in os.listdir(md):
#             with open(md / p) as fi:
#                 t = fi.read().strip()
#             # potentially removing
#             if t:
#                 fo.write(t + "\n")

#     shutil.rmtree(work)
#     return out_path


def process_single_file(in_path: str, out_path: str):
    with open(in_path) as fi:
        lines = [l.strip() for l in fi if l.strip()]

    # in memory version
    lines = dedup_lines(lines)
    lines = minhash_dedup_lines(lines, num_hashes=32, num_bands=8, ngrams=5, jaccard_threshold=0.8)

    with open(out_path, "w") as fo:
        for line in lines:
            fo.write(line + "\n")
    return out_path


if __name__ == "__main__":
    # Set up the executor
    num_cpus = len(os.sched_getaffinity(0))
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
    shutil.rmtree("data/filtered_deduped", ignore_errors=True)
    # shutil.rmtree("data/tmp_dedup", ignore_errors=True)
    os.makedirs("data/filtered_deduped", exist_ok=True)
    # os.makedirs("data/tmp_dedup", exist_ok=True)

    in_files = list(Path("data/filtered_quality").glob("*.warc.wet.gz"))
    futures = []
    for f in in_files:
        # For each filepath, submit a job to the executor and get a future back
        future = executor.submit(process_single_file, str(f), str(Path("data/filtered_deduped") / f.name))
        # Store the futures
        futures.append(future)
    # Iterate over the completed futures as they finish, using a progress bar
    # to keep track of progress.
    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(in_files),
    ):
        out = future.result()
        print(f"Output file written: {out}")
