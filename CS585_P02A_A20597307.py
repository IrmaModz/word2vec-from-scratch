import sys
import os
import numpy as np
import re
import time

# Hyperparameters
OUTPUT_FILE = "CS585_P02A_A20597307_EMBED.TXT"
WINDOW_SIZE = 2
DEFAULT_D = 50
LEARNING_RATE = 0.025
EPOCHS = 5
NEG_SAMPLES = 2

# check input parameters
def check_input_params():   
    # accept two arguments only
    if len(sys.argv)!=3:
        print("Error: Not enough/too many/illegal input arguments.")
        sys.exit(1)

    # D should be a positive integer, if it is out of range  then should be Default value
    try:
        D=int(sys.argv[1])
        if D<=0:
            raise ValueError("D must be positive")
    except ValueError:
        print(f"Error: D is out of range. Using D={DEFAULT_D}")
        D=DEFAULT_D

    # Corpus file must exists
    corpus_file=sys.argv[2]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(script_dir, corpus_file)

    if not os.path.isfile(corpus_path):
        print(f"ERROR: Input file '{corpus_file}' does not exist "
              f"in directory '{script_dir}'.")
        sys.exit(1)
         
    return D, corpus_path


# clean file
def clean_file(file_name):
    # read the file (encoding that handles non-Latin characters)
    # replace (not just delete) non-Latin-letter characters so words stay separated
    # lowercase entire contents
    with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text=text.lower()
    return text

# create tokens + vocabulary
def create_vocabulary(text):
    # Split the text into wordss 
    # Create a list of unique words
    # Sort the unique words alphabetically
    # Give each word a number (index)
    # Keep the original word order for training 
    tokens = text.split()
    V = sorted(set(tokens))
    word_to_index = {word: i for i, word in enumerate(V)}
    return tokens, V, word_to_index

# build training data
def build_training_pairs(tokens, word_to_index):
    # Go through the words in their original order
    # Select one word as the target
    # Find nearby words using WINDOW_SIZE
    # Create a pair for each target and nearby word
    # Store the word indices
    pairs = []
    n = len(tokens)

    for i in range(n):
        target = word_to_index[tokens[i]]

        for offset in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
            if offset == 0:
                continue
            j = i + offset
            if j < 0 or j >= n:
                continue
            context = word_to_index[tokens[j]]
            pairs.append((target, context))

    return pairs

# initialize weights
def init_weights(vocab_size, D):
    # input (target) embedding matrix: vocab_size x D
    # output (context) matrix: as explained in class
    rng = np.random.default_rng(42)
    W_in = rng.uniform(-0.5, 0.5, size=(vocab_size, D)) / D
    W_out = rng.uniform(-0.5, 0.5, size=(vocab_size, D)) / D

    return W_in, W_out

# build negative probabilies
def build_neg_probs(tokens, word_to_index):
    counts = np.zeros(len(word_to_index))
    for word in tokens:
        counts[word_to_index[word]] += 1
    return counts / counts.sum()

# calculate sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# train
def train(pairs, W_in, W_out, neg_probs):
    # Calculate dot products for positive and negative samples
    # Apply sigmoid to calculate probabilities
    # Calculate negative sampling loss
    # Update input and output weights using gradient descent

    rng = np.random.default_rng(42)
    pairs = np.array(pairs)
    vocab_size = W_in.shape[0]

    for epoch in range(EPOCHS):
        order = rng.permutation(len(pairs))
        negatives = rng.choice(vocab_size, size=(len(pairs), NEG_SAMPLES), p=neg_probs)

        for n, idx in enumerate(order):
            t_idx, c_idx = pairs[idx]
            neg_idx = negatives[n]

            t = W_in[t_idx].copy()
            c_pos = W_out[c_idx].copy()
            c_negs = W_out[neg_idx].copy()

            score_pos = sigmoid(t @ c_pos)
            score_negs = sigmoid(c_negs @ t)

            grad_c_pos = (score_pos - 1) * t
            grad_c_negs = score_negs[:, None] * t
            grad_t = (score_pos - 1) * c_pos + score_negs @ c_negs

            W_out[c_idx] -= LEARNING_RATE * grad_c_pos
            np.add.at(W_out, neg_idx, -LEARNING_RATE * grad_c_negs)
            W_in[t_idx] -= LEARNING_RATE * grad_t

    return W_in, W_out

# save results
def save_embedding(V, embeddings, file_name):
    # one line per word, in V's (alphabetical) order
    # format: word, v1, v2, ..., vD   (space after each comma)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, file_name)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, word in enumerate(V):
            values = ", ".join(f"{x:.6f}" for x in embeddings[i])
            f.write(f"{word}, {values}\n")
    return

def main():
    D, corpus_path = check_input_params()

    print("Modzgvrishvili, Irma, A20597307 solution:")
    print(f"Number of dimensions: {D}")
    print(f"Training file name: {os.path.basename(corpus_path)}")
    print(f"Embedding file name: {OUTPUT_FILE}")

    text = clean_file(corpus_path)
    tokens, V, word_to_index = create_vocabulary(text)
    if not tokens:
            print("ERROR: Corpus contains no valid Latin-alphabet words.")
            sys.exit(1)
    pairs = build_training_pairs(tokens, word_to_index)
    neg_probs = build_neg_probs(tokens, word_to_index)

    start = time.perf_counter()
    W_in, W_out = init_weights(len(V), D)
    W_in, W_out = train(pairs, W_in, W_out, neg_probs)
    training_time = time.perf_counter() - start

    save_embedding(V, W_in, OUTPUT_FILE)

    print()
    print(f"Training time: {training_time:.2f} seconds")

if __name__ == "__main__":
    main()