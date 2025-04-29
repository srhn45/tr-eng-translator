import torch
import torch.nn as nn
import math


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_token_id=0):
        super(EmbeddingLayer, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pad_token_id = pad_token_id

        self.pos_embedding = nn.Embedding(512, embed_dim) # max length of 100 for positional encoding

    def forward(self, input_ids, key_padding_mask): # input_ids: (batch_size, seq_len)
        batch_size, seq_len = input_ids.size()
        
        token_emb = self.token_embedding(input_ids) * math.sqrt(self.token_embedding.embedding_dim) # (batch_size, seq_len, embed_dim)

        pos_emb = self.pos_embedding(torch.arange(seq_len, device=input_ids.device)).unsqueeze(0).expand(batch_size, seq_len, -1) # (batch_size, seq_len, embed_dim)

        emb = token_emb + pos_emb

        return emb
    

class EncodingLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(EncodingLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.SiLU(), # using SiLU activation for better performance
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):

        x_norm = self.layernorm1(x) # layernorm before attention (pre-norm)
        attn_y = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)[0]
        x = x + self.dropout(attn_y)


        x_norm = self.layernorm2(x) # using pre-norm for stability
        ff_output = self.feedforward(x_norm)
        x = x + self.dropout(ff_output)


        return x


class DecodingLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(DecodingLayer, self).__init__()

        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.SiLU(), # using SiLU instead of ReLU for better performance  
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, tgt_key_padding_mask=None, memory_key_padding_mask=None, causal_mask=None):
        if causal_mask is None:
            seq_len = x.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        x_norm = self.layernorm1(x) # pre-norm on self attention
        masked_attn = self.attn1(x_norm, x_norm, x_norm, attn_mask=causal_mask, key_padding_mask=tgt_key_padding_mask)[0] # masked self attention
        x = x + self.dropout(masked_attn)


        x_norm = self.layernorm2(x) # pre-norm on cross attention
        cross_attn = self.attn2(x_norm, encoder_out, encoder_out, key_padding_mask=memory_key_padding_mask)[0]
        x = x + self.dropout(cross_attn)

        x_norm = self.layernorm3(x)
        ff_output = self.feedforward(x_norm)
        x = x + self.dropout(ff_output)

        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, ff_dim, num_heads,
                 max_seq_len=None, n_encoders=2, n_decoders=2, pad_token_id=0, dropout=0.1):
        super(Transformer, self).__init__()

        self.pad_token_id = pad_token_id

        self.src_embedding = EmbeddingLayer(vocab_size, embed_dim, pad_token_id) # embedding for source
        self.tgt_embedding = EmbeddingLayer(vocab_size, embed_dim, pad_token_id) # embedding for target

        self.encoder_layers = nn.ModuleList([ # stack of encoding layers
            EncodingLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(n_encoders)
        ])

        self.decoder_layers = nn.ModuleList([ # stack of decoding layers
            DecodingLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(n_decoders)
        ])

        self.decoder_final_norm = nn.LayerNorm(embed_dim) # final layernorm for decoder output

        self.output_projection = nn.Linear(embed_dim, vocab_size) # output layer

    def encode(self, src_ids, src_key_padding_mask):
        x = self.src_embedding(src_ids, src_key_padding_mask)
        for layer in self.encoder_layers:
            x = layer(x, key_padding_mask=src_key_padding_mask)
        return x

    def decode(self, tgt_ids, encoder_out, tgt_key_padding_mask, memory_key_padding_mask):
        x = self.tgt_embedding(tgt_ids, tgt_key_padding_mask)

        seq_len = tgt_ids.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=tgt_ids.device), diagonal=1).bool()

        for layer in self.decoder_layers:
            x = layer(x, encoder_out, memory_key_padding_mask=memory_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, causal_mask=causal_mask)

        x = self.decoder_final_norm(x) # final layernorm for decoder output
        
        return x

    def forward(self, src_ids, tgt_ids, src_key_padding_mask=None, tgt_key_padding_mask=None):
        encoder_out = self.encode(src_ids, src_key_padding_mask)
        decoder_out = self.decode(tgt_ids, encoder_out, tgt_key_padding_mask, src_key_padding_mask)
        logits = self.output_projection(decoder_out)
        return logits  # (batch_size, tgt_seq_len, vocab_size)