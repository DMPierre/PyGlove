import subprocess
import sys
import os
from tempfile import NamedTemporaryFile


class Glove(object):
    """
    This is the main class for the Glove model.
    Args:
    dict(params_dict): dict of accepted parameters for Glove.
    int(verbose): the verbosity amount.

    Returns:
    int(0)
    """

    def __init__(self, params_dict, verbose=2):
        # Set GloVe hyperparameters.
        for k in params_dict.keys():
            self.__setattr__("{}".format(k), str(params_dict[k]))
        self.logs = []

    def train(self):
        with open(self.input, "r") as f_input, open("./GloVe/vocab.txt", "w") as f_output:
            args = ["./GloVe/build/vocab_count"]
            args.append("-min-count")
            args.append(self.min_count)
            args.append("-verbose")
            args.append(self.verbose)
            p = subprocess.run(args, stdin=f_input, stdout=f_output, encoding="utf-8")

        # Calling subprocess for cooccurrence.
        with open(self.input, "r") as f_input, open("./GloVe/cooccurrence.bin", "w") as cooc_output:
            args = ["./GloVe/build/cooccur"]
            args.append("-vocab-file")
            args.append("./GloVe/vocab.txt")
            args.append("-verbose")
            args.append(self.verbose)
            args.append("-window-size")
            args.append(self.window)
            p = subprocess.run(args, stdin=f_input, stdout=cooc_output, encoding="utf-8")

        # Calling subprocess for shuffling coocurrences.
        with open("./GloVe/cooccurrence.bin", "rb") as cooc_input, open("./GloVe/cooccurrence.shuf.bin", "wb") as shuff_output:
            args = ["./GloVe/build/shuffle"]
            args.append("-verbose")
            args.append(self.verbose)
            p = subprocess.run(args, stdin=cooc_input, stdout=shuff_output, encoding="utf-8")

        # Final subprocess for running glove.
        args = ["./GloVe/build/glove"]
        args.append("-save-file")
        args.append(self.output)
        args.append("-threads")
        args.append(self.threads)
        args.append("-input-file")
        args.append("./GloVe/cooccurrence.shuf.bin")
        args.append("-x-max")
        args.append(self.x_max)
        args.append("-iter")
        args.append(self.iter)
        args.append("-vector-size")
        args.append(self.size)
        args.append("-binary")
        args.append(self.binary)
        args.append("-vocab-file")
        args.append("./GloVe/vocab.txt")
        args.append("-verbose")
        args.append(self.verbose)
        p = subprocess.run(args, encoding="utf-8")
        return 0
