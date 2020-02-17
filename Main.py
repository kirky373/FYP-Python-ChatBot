######################################################################
# Conversational models are a hot topic in artificial intelligence
# research. Chatbots can be found in a variety of settings, including
# customer service applications and online helpdesks. These bots are often
# powered by retrieval-based models, which output predefined responses to
# questions of certain forms. In a highly restricted domain like a
# company’s IT helpdesk, these models may be sufficient, however, they are
# not robust enough for more general use-cases. Teaching a machine to
# carry out a meaningful conversation with a human in multiple domains is
# a research question that is far from solved. Recently, the deep learning
# boom has allowed for powerful generative models like Google’s `Neural
# Conversational Model <https://arxiv.org/abs/1506.05869>`__, which marks
# a large step towards multi-domain generative conversational models.
#
#
# **Program Highlights**
#
# -  Handle loading and preprocessing of `Cornell Movie-Dialogs
#    Corpus <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__
#    dataset
# -  Implement a sequence-to-sequence model with `Luong attention
#    mechanism(s) <https://arxiv.org/abs/1508.04025>`__
# -  Jointly train encoder and decoder models using mini-batches
# -  Implement greedy-search decoding module
# -  Interact with trained chatbot
#
# **Acknowledgements**
#
# This Program borrows code from the following sources:
#
# 1) Yuan-Kuei Wu’s pytorch-chatbot implementation:
#    https://github.com/ywk991112/pytorch-chatbot
#
# 2) Sean Robertson’s practical-pytorch seq2seq-translation example:
#    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation
#
# 3) FloydHub’s Cornell Movie Corpus preprocessing code:
#    https://github.com/floydhub/textutil-preprocess-cornell-movie-corpus
#


######################################################################
# Preparations
# ------------
#
# To start, Download the data ZIP file
# `here <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__
# and put in a ``data/`` directory under the current directory.
#
#

import torch

from evaluate import GreedySearchDecoder, evaluateInput
from load import voc
from train import encoder, decoder

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting
evaluateInput(encoder, decoder, searcher, voc)
#print(device) #Debug