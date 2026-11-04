# python CS585_P02B_A20597307.py alone 10

import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

EMBED_FILE = "CS585_P02A_A20597307_EMBED.TXT"
DEFAULT_K = 5

# load embeddings from CS585_P02A file
def load_embeddings(embed_path):
    V = []
    vectors = []
    with open(embed_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            V.append(parts[0].strip())
            vectors.append([float(x) for x in parts[1:]])
    return V, np.array(vectors)

# check input parameters
def check_input_params():
    # two arguments
    if len(sys.argv) != 3:
        print("Error: Not enough/too many/illegal input arguments.")
        print("Usage: python CS585_P02B_A20597307.py WORD K")
        sys.exit(1)

    word = sys.argv[1]

    # K must be an integer in [1, 10], otherwise use the default
    try:
        K = int(sys.argv[2])
        if K < 1 or K > 10:
            raise ValueError
    except ValueError:
        print(f"Warning: K = '{sys.argv[2]}' is out of range. Using K={DEFAULT_K}")
        K = DEFAULT_K

    # embedding file must exist in the same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embed_path = os.path.join(script_dir, EMBED_FILE)

    if not os.path.isfile(embed_path):
        print(f"ERROR: Embedding file '{EMBED_FILE}' does not exist "
              f"in directory '{script_dir}'.")
        sys.exit(1)

    # WORD must be non-empty and contain only Latin letters
    if not re.fullmatch(r"[a-zA-Z]+", word):
        print(f"ERROR: WORD '{word}' is empty or contains non-Latin characters.")
        sys.exit(1)

    # V is all lowercase
    word = word.lower()

    # WORD must be in V
    V, vectors = load_embeddings(embed_path)
    if word not in V:
        print(f"ERROR: WORD '{word}' is not in the vocabulary.")
        sys.exit(1)

    return word, K, V, vectors

# find nearest vectors
def find_nearest(word, K, V, vectors):
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1e-10
    unit = vectors / norms[:, None]

    idx = V.index(word)
    similarities = unit @ unit[idx]
    similarities[idx] = -np.inf

    K = min(K, len(V) - 1)
    nearest = np.argsort(-similarities)[:K]
    return idx, nearest, similarities[nearest]

def main():
    word, K, V, vectors = check_input_params()
    idx, nearest, similarities = find_nearest(word, K, V, vectors)
    print(idx, nearest, similarities)




if __name__ == "__main__":
    main()