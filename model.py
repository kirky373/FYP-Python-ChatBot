import torch
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden

    ######################################################################
    # Decoder
    # ~~~~~~~
    #
    # The decoder RNN generates the response sentence in a token-by-token
    # fashion. It uses the encoder’s context vectors, and internal hidden
    # states to generate the next word in the sequence. It continues
    # generating words until it outputs an *EOS_token*, representing the end
    # of the sentence. A common problem with a vanilla seq2seq decoder is that
    # if we rely soley on the context vector to encode the entire input
    # sequence’s meaning, it is likely that we will have information loss.
    # This is especially the case when dealing with long input sequences,
    # greatly limiting the capability of our decoder.
    #
    # To combat this, `Bahdanau et al. <https://arxiv.org/abs/1409.0473>`__
    # created an “attention mechanism” that allows the decoder to pay
    # attention to certain parts of the input sequence, rather than using the
    # entire fixed context at every step.
    #
    # At a high level, attention is calculated using the decoder’s current
    # hidden state and the encoder’s outputs. The output attention weights
    # have the same shape as the input sequence, allowing us to multiply them
    # by the encoder outputs, giving us a weighted sum which indicates the
    #
    # `Luong et al. <https://arxiv.org/abs/1508.04025>`__ improved upon
    # Bahdanau et al.’s groundwork by creating “Global attention”. The key
    # difference is that with “Global attention”, we consider all of the
    # encoder’s hidden states, as opposed to Bahdanau et al.’s “Local
    # attention”, which only considers the encoder’s hidden state from the
    # current time step. Another difference is that with “Global attention”,
    # we calculate attention weights, or energies, using the hidden state of
    # the decoder from the current time step only. Bahdanau et al.’s attention
    # calculation requires knowledge of the decoder’s state from the previous
    # time step. Also, Luong et al. provides various methods to calculate the
    # attention energies between the encoder output and decoder output which
    # are called “score functions”:
    #
    # where :math:`h_t` = current target decoder state and :math:`\bar{h}_s` =
    # all encoder states.
    #
    # Overall, the Global attention mechanism can be summarized by the
    # following figure. Note that we will implement the “Attention Layer” as a
    # separate ``nn.Module`` called ``Attn``. The output of this module is a
    # softmax normalized weights tensor of shape *(batch_size, 1,
    # max_length)*.
    #

    # Luong attention layer
class Attn(nn.Module):
        def __init__(self, method, hidden_size):
            super(Attn, self).__init__()
            self.method = method
            if self.method not in ['dot', 'general', 'concat']:
                raise ValueError(self.method, "is not an appropriate attention method.")
            self.hidden_size = hidden_size
            if self.method == 'general':
                self.attn = nn.Linear(self.hidden_size, hidden_size)
            elif self.method == 'concat':
                self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
                self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        def dot_score(self, hidden, encoder_output):
            return torch.sum(hidden * encoder_output, dim=2)

        def general_score(self, hidden, encoder_output):
            energy = self.attn(encoder_output)
            return torch.sum(hidden * energy, dim=2)

        def concat_score(self, hidden, encoder_output):
            energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
            return torch.sum(self.v * energy, dim=2)

        def forward(self, hidden, encoder_outputs):
            # Calculate the attention weights (energies) based on the given method
            if self.method == 'general':
                attn_energies = self.general_score(hidden, encoder_outputs)
            elif self.method == 'concat':
                attn_energies = self.concat_score(hidden, encoder_outputs)
            elif self.method == 'dot':
                attn_energies = self.dot_score(hidden, encoder_outputs)

            # Transpose max_length and batch_size dimensions
            attn_energies = attn_energies.t()

            # Return the softmax normalized probability scores (with added dimension)
            return F.softmax(attn_energies, dim=1).unsqueeze(1)

        def score(self, hidden, encoder_output):
            # hidden [1, 512], encoder_output [1, 512]
            if self.method == 'dot':
                energy = hidden.squeeze(0).dot(encoder_output.squeeze(0))
                return energy

            elif self.method == 'general':
                energy = self.attn(encoder_output)
                energy = hidden.squeeze(0).dot(energy.squeeze(0))
                return energy

            elif self.method == 'concat':
                energy = self.attn(torch.cat((hidden, encoder_output), 1))
                energy = self.v.squeeze(0).dot(energy.squeeze(0))
                return energy

######################################################################
# Now that we have defined our attention submodule, we can implement the
# actual decoder model. For the decoder, we will manually feed our batch
# one time step at a time. This means that our embedded word tensor and
# GRU output will both have shape *(1, batch_size, hidden_size)*.
#
# **Computation Graph:**
#
#    1) Get embedding of current input word.
#    2) Forward through unidirectional GRU.
#    3) Calculate attention weights from the current GRU output from (2).
#    4) Multiply attention weights to encoder outputs to get new "weighted sum" context vector.
#    5) Concatenate weighted context vector and GRU output using Luong eq. 5.
#    6) Predict next word using Luong eq. 6 (without softmax).
#    7) Return output and final hidden state.
#
# **Inputs:**
#
# -  ``input_step``: one time step (one word) of input sequence batch;
#    shape=\ *(1, batch_size)*
# -  ``last_hidden``: final hidden layer of GRU; shape=\ *(n_layers x
#    num_directions, batch_size, hidden_size)*
# -  ``encoder_outputs``: encoder model’s output; shape=\ *(max_length,
#    batch_size, hidden_size)*
#
# **Outputs:**
#
# -  ``output``: softmax normalized tensor giving probabilities of each
#    word being the correct next word in the decoded sequence;
#    shape=\ *(batch_size, voc.num_words)*
# -  ``hidden``: final hidden state of GRU; shape=\ *(n_layers x
#    num_directions, batch_size, hidden_size)*
#

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden