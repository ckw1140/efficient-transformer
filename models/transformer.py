import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import (
    gelu,
    get_attention_decoder_mask,
    get_attention_pad_mask,
    get_sinusoid_encoding_table,
)

VOCAB_SIZE = 8008
PAD_TOKEN = 0


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        head_dim,
        dropout_prob,
    ):
        super(MultiheadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.W_q = nn.Linear(hidden_dim, num_heads * head_dim)
        self.W_k = nn.Linear(hidden_dim, num_heads * head_dim)
        self.W_v = nn.Linear(hidden_dim, num_heads * head_dim)
        self.W_o = nn.Linear(num_heads * head_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
    ):
        """
        query, key, value: [batch_size, sequence_length, hidden_dim]
        """
        batch_size = query.size()[0]

        # Q, K, V: [batch_size, num_heads, sequence_length, hidden_dim]
        Q = self.split_heads(self.W_q(query), batch_size)
        K = self.split_heads(self.W_k(key), batch_size)
        V = self.split_heads(self.W_v(value), batch_size)

        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # scaled_dot_product: [batch_size, num_heads, sequence_length, sequence_length]
        scaled_dot_product = torch.matmul(Q, K.transpose(2, 3)) / (self.head_dim ** 0.5)
        scaled_dot_product.masked_fill(attention_mask, -1e9)

        # attention_prob: [batch_size, num_heads, sequence_length, sequence_length]
        attention_prob = F.softmax(scaled_dot_product, dim=-1)
        attention_prob = self.dropout(attention_prob)

        # attention_result: [batch_size, num_heads, sequence_length, head_dim]
        attention_result = torch.matmul(attention_prob, V)

        # attention_result: [batch_size, sequence_length, hidden_dim]
        attention_result = attention_result.transpose(1, 2).contiguous()
        attention_result = attention_result.view(batch_size, -1, self.hidden_dim)

        outputs = self.W_o(attention_result)
        return outputs, attention_prob


    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim,
        feed_forward_dim,
        dropout_prob,
    ):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(hidden_dim, feed_forward_dim)
        self.linear2 = nn.Linear(feed_forward_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        inputs,
    ):
        outputs = self.linear1(inputs)
        outputs = gelu(outputs)
        outputs = self.linear2(outputs)
        return self.dropout(outputs)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        head_dim,
        feed_forward_dim,
        dropout_prob,
        layer_norm_eps,
    ):
        super(EncoderLayer, self).__init__()

        self.multihead_attention = MultiheadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout_prob=dropout_prob,
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            feed_forward_dim=feed_forward_dim,
            dropout_prob=dropout_prob,
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

    def forward(
        self,
        inputs,
        attention_mask,
    ):
        residual = inputs
        outputs, attention_prob = self.multihead_attention(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=attention_mask,
        )
        outputs = self.layer_norm1(outputs + residual)

        residual = outputs
        outputs = self.feed_forward(outputs)
        outputs = self.layer_norm2(outputs + residual)

        return outputs, attention_prob


class Encoder(nn.Module):
    def __init__(
        self,
        sequence_length,
        num_layers,
        hidden_dim,
        num_heads,
        head_dim,
        feed_forward_dim,
        dropout_prob,
        layer_norm_eps,
    ):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, hidden_dim)
        self.positional_encoding = nn.Embedding.from_pretrained(
            torch.FloatTensor(
                get_sinusoid_encoding_table(
                    sequence_length=sequence_length + 1,
                    hidden_dim=hidden_dim,
                )
            ),
            freeze=True,
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    feed_forward_dim=feed_forward_dim,
                    dropout_prob=dropout_prob,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        inputs,
    ):
        """
        inputs: [batch_size, sequence_length]
        """
        batch_size = inputs.size()[0]
        sequence_length = inputs.size()[1]

        # position: [batch_size, sequence_length]
        position = (
            torch.arange(
                sequence_length,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            .expand(
                batch_size,
                sequence_length,
            )
            .contiguous()
        )
        position = position + 1

        # outputs: [bacth_size, sequence_length, hidden_dim]
        outputs = self.embedding(inputs) + self.positional_encoding(position)

        # attention_mask: [batch_size, sequence_length, sequence_length]
        attention_mask = get_attention_pad_mask(
            key=inputs,
            pad_token=PAD_TOKEN,
        )
        
        attention_probs = []

        for layer in self.layers:
            outputs, attention_prob = layer(outputs, attention_mask)
            attention_probs.append(attention_prob)

        return outputs, attention_probs


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        head_dim,
        feed_forward_dim,
        dropout_prob,
        layer_norm_eps,
    ):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiheadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout_prob=dropout_prob,
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dec_enc_attention = MultiheadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout_prob=dropout_prob,
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            feed_forward_dim=feed_forward_dim,
            dropout_prob=dropout_prob,
        )
        self.layer_norm3 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

    def forward(
        self,
        dec_inputs,
        enc_outputs,
        self_attention_mask,
        dec_enc_attention_mask,
    ):
        residual = dec_inputs
        outputs, self_attention_prob = self.self_attention(
            query=dec_inputs,
            key=dec_inputs,
            value=dec_inputs,
            attention_mask=self_attention_mask,
        )
        outputs = self.layer_norm1(outputs + residual)

        residual = outputs
        outputs, dec_enc_attention_prob = self.dec_enc_attention(
            query=dec_inputs,
            key=enc_outputs,
            value=enc_outputs,
            attention_mask=dec_enc_attention_mask,
        )
        outputs = self.layer_norm2(outputs + residual)

        residual = outputs
        outputs = self.feed_forward(outputs)
        outputs = self.layer_norm3(outputs + residual)

        return outputs, self_attention_prob, dec_enc_attention_prob


class Decoder(nn.Module):
    def __init__(
        self,
        sequence_length,
        num_layers,
        hidden_dim,
        num_heads,
        head_dim,
        feed_forward_dim,
        dropout_prob,
        layer_norm_eps,
    ):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, hidden_dim)
        self.positional_encoding = nn.Embedding.from_pretrained(
            torch.FloatTensor(
                get_sinusoid_encoding_table(
                    sequence_length=sequence_length + 1,
                    hidden_dim=hidden_dim,
                )
            ),
            freeze=True,
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    feed_forward_dim=feed_forward_dim,
                    dropout_prob=dropout_prob,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        dec_inputs,
        enc_inputs,
        enc_outputs,
    ):
        """
        inputs: [batch_size, sequence_length]
        """
        batch_size = dec_inputs.size()[0]
        sequence_length = dec_inputs.size()[1]

        # position: [batch_size, sequence_length]
        position = (
            torch.arange(
                sequence_length,
                device=dec_inputs.device,
                dtype=dec_inputs.dtype,
            )
            .expand(
                batch_size,
                sequence_length,
            )
            .contiguous()
        )
        position = position + 1

        # outputs: [bacth_size, sequence_length, hidden_dim]
        outputs = self.embedding(dec_inputs) + self.positional_encoding(position)

        # self_attention_mask: [batch_size, sequence_length, sequence_length]
        self_attention_pad_mask = get_attention_pad_mask(
            key=dec_inputs,
            pad_token=PAD_TOKEN,
        ).to(dec_inputs.device)

        self_attention_decoder_mask = get_attention_decoder_mask(
            batch_size=batch_size,
            sequence_length=sequence_length,
        ).to(dec_inputs.device)
        
        self_attention_mask = torch.gt(self_attention_pad_mask + self_attention_decoder_mask, 0)

        # dec_enc_attention_mask: [batch_size, sequence_length, sequence_length]
        dec_enc_attention_mask = get_attention_pad_mask(
            key=enc_inputs,
            pad_token=PAD_TOKEN,
        )
        
        self_attention_probs = []
        dec_enc_attention_probs = []

        for layer in self.layers:
            outputs, self_attention_prob, dec_enc_attention_prob = layer(
                dec_inputs=outputs,
                enc_outputs=enc_outputs,
                self_attention_mask=self_attention_mask,
                dec_enc_attention_mask=dec_enc_attention_mask,
            )
            self_attention_probs.append(self_attention_prob)
            dec_enc_attention_probs.append(dec_enc_attention_prob)

        return outputs, self_attention_probs, dec_enc_attention_probs


class Transformer(nn.Module):
    def __init__(
        self,
        sequence_length,
        num_layers,
        hidden_dim,
        num_heads,
        head_dim,
        feed_forward_dim,
        dropout_prob,
        layer_norm_eps,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            sequence_length=sequence_length,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            feed_forward_dim=feed_forward_dim,
            dropout_prob=dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )

        self.decoder = Decoder(
            sequence_length=sequence_length,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            feed_forward_dim=feed_forward_dim,
            dropout_prob=dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(
        self,
        enc_inputs,
        dec_inputs,
    ):
        enc_outputs, enc_self_attention_probs = self.encoder(
            inputs=enc_inputs,
        )
        dec_outputs, dec_self_attention_probs, dec_enc_attention_probs = self.decoder(
            dec_inputs=dec_inputs,
            enc_inputs=enc_inputs,
            enc_outputs=enc_outputs,
        )

        return (
            dec_outputs,
            enc_self_attention_probs,
            dec_self_attention_probs,
            dec_enc_attention_probs,
        )