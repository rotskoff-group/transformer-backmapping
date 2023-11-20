import torch
import math
import torch.nn as nn
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, all_src_cont, all_src_cat, all_tgt):
        self.all_src_cont = all_src_cont
        self.all_src_cat = all_src_cat
        self.all_tgt = all_tgt
        assert self.all_src_cont.shape[0] == self.all_src_cat.shape[0] == self.all_tgt.shape[0]

    def __len__(self):
        return self.all_src_cont.shape[0]

    def __getitem__(self, idx):
        return self.all_src_cont[idx], self.all_src_cat[idx], self.all_tgt[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len, dropout_p):
        """Args:
            dim_model: Dimension of the model
            max_len: Maximum length of the sequence
            dropout_p: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len).unsqueeze(-1).float()

        k = torch.arange(0, dim_model//2, 1).float()
        log_w_k = (-(2 * k)/dim_model) * math.log(10000.0)
        w_k = torch.exp(log_w_k)

        pos_encoding[:, 0::2] = torch.sin(positions_list * w_k)
        pos_encoding[:, 1::2] = torch.cos(positions_list * w_k)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding):
        """Args:
            token_embedding: (batch_size, seq_len, dim_model)
        """
        return self.dropout(token_embedding + self.pos_encoding[:, :token_embedding.shape[1], :])


class Transformer(nn.Module):
    def __init__(self, num_tokens_src,
                 num_tokens_tgt,
                 dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p):
        """Args:
            num_tokens: Number of tokens in the vocabulary
            dim_model: Dimension of the model
            num_heads: Number of heads in the multi-head attention
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dropout_p: Dropout probability
        """
        super().__init__()
        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(dim_model=dim_model,
                                                     dropout_p=dropout_p,
                                                     max_len=5000)
        self.src_embedding_cont = nn.Linear(4, dim_model//2)
        self.src_embedding_cat = nn.Embedding(num_tokens_src, dim_model//2)


        self.tgt_embedding = nn.Embedding(num_tokens_tgt, dim_model)

        self.transformer = nn.Transformer(d_model=dim_model,
                                          nhead=num_heads,
                                          batch_first=True,
                                          dim_feedforward=1024,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dropout=dropout_p)
        
        self.out = nn.Linear(dim_model, num_tokens_tgt)

    def forward(self, src_cont, src_cat, tgt,
                tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        """Args:
            src: Source sequence - (batch_size, sequence length)
            tgt: Target sequence - (batch_size, sequence length)
        """
        src = torch.cat((self.src_embedding_cont(src_cont),
                         self.src_embedding_cat(src_cat)), dim=-1)
        src = src * math.sqrt(self.dim_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        transformer_out = self.transformer(src, tgt,
                                           tgt_mask=tgt_mask,
                                           src_key_padding_mask=src_pad_mask,
                                           tgt_key_padding_mask=tgt_pad_mask,
                                           tgt_is_causal=True)
        out = self.out(transformer_out)
        return out

    def get_tgt_mask(self, size):
        """Args:
            size: Size of the mask
        """
        mask = torch.tril(torch.ones(size, size))
        mask = mask.masked_fill(mask == 0, -torch.inf)
        mask = mask.masked_fill(mask == 1, 0)
        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)
