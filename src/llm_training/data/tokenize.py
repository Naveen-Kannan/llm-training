import multiprocessing
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import os

if __name__ == "__main__":
    input_path = "data/filtered_deduped"
    output_path = "data/tokens.bin"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def tokenize_line_and_add_eos(line):
        return tokenizer.encode(line) + [tokenizer.eos_token_id]

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    chunksize = 100
    # results = []

    # for result in tqdm(
    #     pool.imap(tokenize_line_and_add_eos, lines, chunksize=chunksize),
    #     total=len(lines),
    #     desc="Tokenizing lines"
    # ):
    #     results.append(result)
    # pool.close()
    # pool.join()

    # # Flatten the list of ids and convert to numpy array
    # all_ids = [token_id for sublist in results for token_id in sublist]
    # print(f"Tokenized and encoded {input_path} into {len(all_ids)} tokens")
    # ids_array = np.array(all_ids, dtype=np.uint16)
    # ids_array.tofile(output_path)

    tot = 0
    # incremental write, per file
    with open(output_path, "wb") as out:
        for file in os.listdir(input_path):
            with open(os.path.join(input_path, file)) as f:
                lines = f.readlines()
            for result in tqdm(
                pool.imap(tokenize_line_and_add_eos, lines, chunksize=chunksize),
                total=len(lines),
                desc=f"Tokenizing {file}",
            ):
                np.array(result, dtype=np.uint16).tofile(out)
                tot += len(result)
    pool.close()
    pool.join()
    print(f"Tokenized and encoded {input_path} into {tot} tokens")
