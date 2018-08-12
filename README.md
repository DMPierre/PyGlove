#### PyGlove: An easy to use python wrapper for the GloVe C-library.


The python wrapper accepts all of the used GloVe hyperparameters.


```
params = {"size": dimension, "window": window, "input": input,
          "output": output, "min_count": min_count, "iter": iter,
          "threads": threads, "x_max": xmax, "binary": binary,
          "verbose": verbose}
```


Among the most commonly tweaked:


- ```size``` is the embedding dimension.
- ```window``` is the window size.
- ```input``` is the path to the txt input file.
- ```output``` will be the name of the output file containing the vectors.
- ```min_count``` is the size of the minimum required count for words to be included in the vocabulary.
- ```iter``` is the number of iterations to be run.


Example use:

```
$ ./train.py -i sampledata/sample.txt -o vectors -d 128
```

Output:


```
in -0.026202 -0.033490 0.009888 -0.048939 -0.032035 [...]
```
