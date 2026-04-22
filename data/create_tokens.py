import gzip
import os 
import pickle
from posixpath import split
from typing import List, Tuple
from tqdm import tqdm
from configs.data_paths import DataPaths
from models.SPACE_GROUPS import SPACE_GROUPS
from models.tokenizier import CIFTokenizer


def tokenize_dataset(
    input_file: str,
    output_file: str,
    add_special_tokens: bool = True,
    single_spaces: bool = True,
) -> None:
    """
    Reads a gzip-pickled List[Tuple[str, str]] of (id, cif_text),
    tokenizes each CIF string, and writes a gzip-pickled
    List[Tuple[str, List[int]]] of (id, token_ids).

    Args:
        input_file:          Path to input .pkl.gz (from preprocess_folder)
        output_file:         Path to output .pkl.gz
        add_special_tokens:  If True, wraps each sequence with <bos> and <eos>
        single_spaces:       If True, collapses multiple spaces/tabs into one
    """
    print(f"Loading {input_file} ...")
    with gzip.open(input_file, "rb") as f:
        data: List[Tuple[str, str]] = pickle.load(f)
    print(f"  Loaded {len(data)} CIF entries.")

    tokenizer = CIFTokenizer(add_special_tokens=add_special_tokens)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    tokenized: List[Tuple[str, List[int]]] = []
    skipped = 0
    unk_id = tokenizer.unk_token_id
    total_unk = 0
    total_tokens = 0

    for cif_id, cif_text in tqdm(data, desc="Tokenizing"):
        try:
            token_ids = tokenizer.tokenize_and_encode(
                cif_text,
                single_spaces=single_spaces,
                add_special_tokens=add_special_tokens,
            )

            n_unk = token_ids.count(unk_id)
            total_unk += n_unk
            total_tokens += len(token_ids)

            tokenized.append((cif_id, token_ids))

        except Exception as e:
            print(f"  [WARN] Skipping {cif_id}: {e}")
            skipped += 1


    n = len(tokenized)
    if n > 0:
        lengths = [len(ids) for _, ids in tokenized]
        print(f"\n--- Tokenization complete ---")
        print(f"  Encoded:       {n} CIFs")
        print(f"  Skipped:       {skipped} CIFs")
        print(f"  Total tokens:  {total_tokens:,}")
        print(f"  Avg length:    {total_tokens / n:.1f} tokens/CIF")
        print(f"  Min length:    {min(lengths)}")
        print(f"  Max length:    {max(lengths)}")
        print(f"  <unk> tokens:  {total_unk:,} ({100 * total_unk / total_tokens:.2f}%)")
        if total_unk / total_tokens > 0.01:
            print("  [WARN] >1% unknown tokens — consider expanding your vocabulary.")


    print(f"\nSaving to {output_file} ...")
    with gzip.open(output_file, "wb") as f:
        pickle.dump(tokenized, f)
    print("Done.")


def inspect(input_file: str, n: int = 3) -> None:
    """
    Quick sanity check: print the first n entries from a tokenized .pkl.gz,
    decoded back to text so you can verify correctness.
    """
    tokenizer = CIFTokenizer(add_special_tokens=True)

    with gzip.open(input_file, "rb") as f:
        data: List[Tuple[str, List[int]]] = pickle.load(f)

    print(f"File contains {len(data)} entries.\n")
    for cif_id, token_ids in data[:n]:
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"=== {cif_id} ===")
        print(f"  Token count : {len(token_ids)}")
        print(f"  First 20 IDs: {token_ids[:20]}")
        print(f"  Decoded (first 300 chars):\n{decoded[:300]}")
        print()



if __name__ == "__main__":
    DATA_DIR = DataPaths().processed_data_dir

    # tokenize the train data 
    input_train_file = DataPaths().train_data_file
    output_train_file = DataPaths().train_tokens_file
    os.makedirs(DataPaths().tokens_folder, exist_ok=True)

    tokenize_dataset(
            input_file=input_train_file,
            output_file=output_train_file,
            add_special_tokens=True,
            single_spaces=True,
        )

    input_val_file = DataPaths().val_data_file
    output_val_file = DataPaths().val_tokens_file
    tokenize_dataset(
            input_file=input_val_file,
            output_file=output_val_file,
            add_special_tokens=True,
            single_spaces=True,
        )
    
    input_test_file = DataPaths().test_data_file
    output_test_file = DataPaths().test_tokens_file
    tokenize_dataset(
            input_file=input_test_file, 
            output_file=output_test_file,
            add_special_tokens=True,
            single_spaces=True,
        )

    print(f"\n--- Sanity check: {split} (first 3 entries decoded) ---\n")
    inspect(output_train_file, n=3)