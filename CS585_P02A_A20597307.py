'''
python CS585_P02A_A20597307.py 20 CORPUS_FILE
Irma Modgvrishvili
A20597307
'''
import sys
import os
import numpy as np

DEFAULT_D = 50

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

if __name__ == "__main__":
    print(check_input_params())