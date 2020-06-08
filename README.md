# dark_web_authorship

EECE5644 final project documentation. Applies LSTM and RNN neural networks to authorship classification in dark web marketplaces using Twitter GloVe vector representaions. 

Credit goes to Gwern Branwen for the dataset, which may be accessed at https://www.gwern.net/DNM-archives.  Here, a subset of the data has been prepared in order to avoid excessive downloads.


## Installation

Install the stanford gloVe twitter and wikipedia pretrained using wget. Note, these will take a very long time to install.
```wget http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip```
```wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip ```

Unzip the files:

```unzip glove.twitter.27B.zip```
```unzip glove.6B.zip```

The `parse_file` code will prepare the pages in a given directory. Pre-prepared files are provided in the `files` folder.

The environment for the project can be installed using conda:

```
$ conda install -f environment.yml
```

```
$ conda activate silk-road-author-id
```



## File naming scheme:
### train/test in data prefixed, validate in validate prefixed
data_{GLOVE_TYPE}_{N_AUTHORS}_{EMBEDDING_VECTOR_SIZE}_{INPUT_SIZE}_df.pickle
<br>
validate_{GLOVE_TYPE}_{N_AUTHORS}_{EMBEDDING_VECTOR_SIZE}_{INPUT_SIZE}_df.pickle