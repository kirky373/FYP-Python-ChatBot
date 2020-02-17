import os

MAX_LENGTH = 10  # Maximum sentence length to consider
teacher_forcing_ratio = 1.0
save_dir = 'data/save'
corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)