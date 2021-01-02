import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    cdf = 0.5 * (
        1.0 + torch.tanh((math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )
    return x * cdf


def get_sinusoid_encoding_table(
    sequence_length,
    hidden_dim,
):
    def get_angle(pos, index):
        return pos / np.power(10000, 2 * (index // 2) / hidden_dim)

    def get_position_angle_vector(pos):
        return [get_angle(pos, i) for i in range(hidden_dim)]

    sinusoid_table = np.array([get_position_angle_vector(pos) for pos in range(sequence_length)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    
    return sinusoid_table


def get_attention_pad_mask(key, pad_token):
    """
    key: [batch_size, sequence_length]
    """
    batch_size = key.size()[0]
    sequence_length = key.size()[1]

    return key.data.eq(pad_token).unsqueeze(1).expand(batch_size, sequence_length, sequence_length)


def get_attention_decoder_mask(batch_size, sequence_length):
    mask = 1 - torch.triu(torch.ones(sequence_length, sequence_length)).long().T
    mask = mask.unsqueeze(0)
    return mask.expand(batch_size, sequence_length, sequence_length)