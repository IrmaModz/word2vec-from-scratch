'''
python CS585_P02A_A20597307.py 2 CORPUS_FILE.txt
Irma Modgvrishvili
A20597307
'''
import sys
import os
import numpy as np
import re

# Hyperparameters
OUTPUT_FILE = "CS585_P02A_A20597307_EMBED.TXT"
WINDOW_SIZE = 2
DEFAULT_D = 50


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

# train
def train(pairs, W_in, W_out, vocab_size):
    # for each epoch:
    #   for each (target, context) pair:
    #      Get the target word vector
    #      Calculate scores using dot products
    #      Convert scores into probabilities using softmax  
    #      Compare the prediction with the correct word
    #      Calculate loss  
    #      Calculate gradients
    #     update weights using LEARNING_RATE
    return W_in, W_out

# save results
def save_embedding(V, embeddings, file_name):
    # one line per word, in V's (alphabetical) order
    # format: word, v1, v2, ..., vD   (space after each comma)
    return

def main():
    D, corpus_path = check_input_params()
    text = clean_file(corpus_path)
    print(text)
    print("-----------")

    tokens, V, word_to_index = create_vocabulary(text)
    print(tokens, V, word_to_index)
    print("-----------")

    pairs = build_training_pairs(tokens, word_to_index)
    print(pairs)

    W_in, W_out = init_weights(len(V), D)
    print("-----------")
    print(W_in, W_out)
    #W_in, W_out = train(pairs, W_in, W_out, len(V))
    #save_embedding(V, W_in, OUTPUT_FILE)

    return


if __name__ == "__main__":
    main()
    