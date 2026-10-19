import concurrent.futures
import gzip
from pathlib import Path
import os
from collections import Counter
import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm
from llm_training.data.common import get_shared_assets_path
from llm_training.data.gopher_filters import gopher_filters
from llm_training.data.identify_harmful import identify_nsfw, identify_toxic_speech
from llm_training.data.mask_pii import email_mask, ip_mask, phone_mask
from llm_training.data.train_classifier import classify_quality
import shutil
import random

local = Path("/shared-data") if Path("/shared-data").exists() else get_shared_assets_path()

# nsfw_classifier = fasttext.load_model(str(local / "classifiers/dolma_fasttext_nsfw_jigsaw_model.bin"))
# toxic_classifier = fasttext.load_model(str(local / "classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin"))
quality_classifier = fasttext.load_model("data/quality_classifier.bin")


def process_single_wet_file(input_path: str, output_path: str):
    workor_totals = Counter()
    with open(input_path) as f:
        with open(output_path, "w") as fo:
            for line in f:
                workor_totals["total"] += 1
                text = line.strip()
                label, score = classify_quality(text, quality_classifier)
                if label == "cc" and score >= 0.9999:
                    # only actually do this with a 15% chance.
                    if random.random() > 0.15:
                        workor_totals["quality"] += 1
                        continue
                fo.write(text + "\n")
                workor_totals["final"] += 1
    return output_path, workor_totals


if __name__ == "__main__":
    # Set up the executor
    num_cpus = len(os.sched_getaffinity(0))
    # num_cpus = os.cpu_count()
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
    shutil.rmtree("data/filtered_quality", ignore_errors=True)
    os.makedirs("data/filtered_quality", exist_ok=True)
    inp_dir = Path("data/filtered")
    # wet_filepaths = []
    # for path in os.listdir(inp_dir):
    #     wet_filepaths.append(inp_dir / path)
    wet_filepaths = list((inp_dir).glob("*.warc.wet.gz"))
    out_dir = Path("data/filtered_quality")
    futures = []
    for wet_filepath in wet_filepaths:
        # For each warc.wet.gz filepath, submit a job to the executor and get a future back
        wet_filename = str(Path(wet_filepath).name)
        future = executor.submit(process_single_wet_file, wet_filepath, os.path.join(out_dir, wet_filename))
        # Store the futures
        futures.append(future)
    # Iterate over the completed futures as they finish, using a progress bar
    # to keep track of progress.
    totals = Counter()
    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(wet_filepaths),
    ):
        output_file, workor_totals = future.result()
        totals.update(workor_totals)
        print(f"Output file written: {output_file}")
    print(dict(totals))
