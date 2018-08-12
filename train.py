#!/usr/bin/env python3
import subprocess
import sys
import os
import argparse
import time

# Import Glove
from model import Glove


class Train(Glove):
    """
    Class for training.
    Args:
    dict(params): dict containing common GloVe hyperparams
    str(input): path to input file
    str(output): path to output file
    """

    def __init__(self, params, input, output):
        print("Training with parameters: {}.\n".format(
            " ".join(["{}={}".format(k, v) for k, v in params.items()])))

        self.params = params
        self.input = input
        self.output = output

    def train(self):
        t1 = time.time()
        self.model = Glove(self.params)
        self.process = self.model.train()
        t2 = time.time()
        print("\nFinished training. Time taken: {} mins.\n".format(str((t2 - t1) / 60)[:5]))
        return self.process


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    parser.add_argument("-d", "--dimension", type=int)

    # Common Glove parameters
    parser.add_argument("-w", "--window", default=5, type=int)
    parser.add_argument("-min", "--min_count", default=2, type=int)
    parser.add_argument("-it", "--iter", default=50, type=int)
    parser.add_argument("-ths", "--threads", default=40, type=int)
    parser.add_argument("-xmax", "--xmax", default=10, type=int)
    parser.add_argument("-v", "--verbose", default=2, type=int)
    parser.add_argument("-bin", "--binary", default=0, type=int)

    args = parser.parse_args()

    params = {"size": args.dimension, "window": args.window, "input": args.input,
              "output": args.output, "min_count": args.min_count, "iter": args.iter,
              "threads": args.threads, "x_max": args.xmax, "binary": args.binary,
              "verbose": args.verbose}

    glove = Train(params, args.input, args.output)
    sys.exit(glove.train())
