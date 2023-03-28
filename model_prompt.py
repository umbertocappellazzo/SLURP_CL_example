"""

Author: Stefano Ciapponi

Extension of the baseline model with Prompt Tuning.

"""
from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
import random
from transformers import  Wav2Vec2Model, Wav2Vec2FeatureExtractor
from prompt import Prompt, PromptArgs

# Model DataClass (Should be extended with Prompt Module Dimensions)

@dataclass
class ModelDimensions:
    # Seq2Seq Dimensions
    n_mels: int
    kernel_size: int
    n_hidden_audio: int
    n_head_audio: int
    n_layer_audio: int
    n_vocab: int
    n_hidden_text: int
    n_head_text: int
    n_layer_text: int
    drop: float
    n_feedforward: int
    batch_first: bool = True
    norm_first: bool = True



# Transformer Positional Encoding

class PositionalEncoding(nn.Module):

    def __init__(self,  d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x=x.permute(1,0,2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1,0,2)
        return self.dropout(x)        
    
# Seq2seq transformer module

class Seq2SeqTransformer(nn.Module):
    def __init__(self, 
                 dims: ModelDimensions,
                 prompt_args=PromptArgs,
                 device="cuda",):
        
        super().__init__()
        
        # Prompting Part
        self.prompt = Prompt(prompt_args)
        
        self.device = device
        self.dims = dims
        
        self.token_embedding = nn.Embedding(self.dims.n_vocab, self.dims.n_hidden_text)
        self.positional_encoder = PositionalEncoding(d_model=self.dims.n_hidden_text, dropout=self.dims.drop, max_len=5000)
        self.ln = nn.LayerNorm(self.dims.n_hidden_audio)
        
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.dims.n_hidden_text, nhead=self.dims.n_head_text, dim_feedforward=self.dims.n_feedforward, dropout=self.dims.drop, batch_first=self.dims.batch_first, norm_first = self.dims.norm_first),  self.dims.n_layer_text, self.ln) 
        self.classif_dec = nn.Linear(self.dims.n_hidden_text,self.dims.n_vocab)
        
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h") 
        
        self.normalize_wav = self.feature_extractor.do_normalize
        
        self.wav2vec.feature_extractor.eval()
        for p in self.wav2vec.feature_extractor.parameters():
            p.requires_grad = False
        
        self._init_weights()
        
        
    def _init_weights(self):
        for p in self.decoder.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)
    
        
        
    def create_pad_mask(self, matrix, pad_token):
            # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
            # [False, False, False, True, True, True]
            return (matrix == pad_token)    
    
    def create_tgt_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        return torch.empty(sz, sz).fill_(-np.inf).triu_(1)
        
    def embed_audio(self, src):
        
        if self.normalize_wav:
            src = F.layer_norm(src, src.shape[1:])
        
        
        x_enc = self.wav2vec(src)[0]

        return x_enc
    
    def embed_audio_iCaRL(self, src):
        if self.normalize_wav:
            src = F.layer_norm(src, src.shape[1:])
        x_enc = self.wav2vec(src)[0]
        
        return x_enc.mean(1)
    
    def embed_audio_ctc(self, src):
        
        if self.normalize_wav:
            src = F.layer_norm(src, src.shape[1:])
        
        x_enc = self.wav2vec(src)[0]

        enc_out = self.classif_enc(x_enc)
        enc_out= torch.nn.functional.log_softmax(enc_out,dim=-1)
        
        
        return enc_out
    
    def decod_audio(self, enc_out, target, tgt_mask = None, tgt_key_padding_mask = None):
        x_dec = self.token_embedding(target) * math.sqrt(self.dims.n_hidden_text)
        x_dec = self.positional_encoder(x_dec)
        dec_out = self.decoder(x_dec,enc_out , tgt_mask = tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        
        return self.classif_dec(dec_out)
        
        
    def append_prompt(self, x):
        x = self.prompt(x)
        print(x)
        return x
   
    def forward(self, src, target, tgt_mask = None, tgt_key_padding_mask = None):
        
        if self.normalize_wav:
            src = F.layer_norm(src, src.shape[1:])
        
        x_enc = self.wav2vec(src)[0] # Audio Embedding

        # Now we need to process the Audio embedding Adding Prompts
        x_enc = append_prompt(x_enc)



        
        x_dec = self.token_embedding(target) * math.sqrt(self.dims.n_hidden_text)
        x_dec = self.positional_encoder(x_dec)
        
        
        
        
        dec_out = self.decoder(x_dec, x_enc, tgt_mask = tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        
        dec_out = self.classif_dec(dec_out)
        
        return dec_out, x_enc  #enc_out