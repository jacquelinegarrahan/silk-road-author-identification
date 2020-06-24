# Silk Road Author Identification

This is the source code for my EECE5644 final project (Summer 2018), which aims to apply LSTM and RNN neural networks to classifying sequence representations by author in Silk Road forums. The code contains notebooks for preparing and running both models types, as well as building datasets. Vector representations of the forum posts were built using the Stanford gloVe vector representations. The BuildDataset notebook includes utility functions for building embedding matrices from the Stanford vector representations and preparing datasets. The dataset used for this project was compiled by Gwern Branwen for the dataset, and can be accessed at https://www.gwern.net/DNM-archives.

## Installation

Install the stanford gloVe twitter and wikipedia pretrained using wget. Note, these will take a very long time to install.
<br>
`$ wget http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip`
<br>
`$ wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip `

Unzip the files:

`$ unzip glove.twitter.27B.zip`
`$ unzip glove.6B.zip`

The `parse_file` code will prepare the pages in a given directory. Pre-prepared files are provided in the `files` folder.

The environment for the project can be installed using conda:

`$ conda env install -f environment.yml`

`$ conda activate silk-road-author-id`

Once activated, install the environment using ipykernel:

` $ python -m ipykernel install --user --name=silk-road-author-id`

Launch the notebooks:

` $ jupyter notebook `

## File naming scheme:
data frames:
`{GLOVE_TYPE}_{N_AUTHORS}_{EMBEDDING_VECTOR_SIZE}_{INPUT_SIZE}_df.pickle`
embedding matrices: 
`{GLOVE_TYPE}_{N_AUTHORS}_{EMBEDDING_VECTOR_SIZE}_{INPUT_SIZE}_embedding.pickle`
