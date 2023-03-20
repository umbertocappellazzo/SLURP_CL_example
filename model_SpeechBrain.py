#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 09:25:02 2023

@author: umbertocappellazzo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 22:26:27 2023

@author: umbertocappellazzo
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
#from conformer import Conformer
   
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
   

@dataclass
class ModelDimensions:
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
    


class Seq2SeqTransformer(nn.Module):
    def __init__(self, 
                 dims: ModelDimensions,
                 device="cuda",):
        
        super().__init__()
        
        
        
        self.device = device
        self.dims = dims
        
        self.token_embedding = nn.Embedding(self.dims.n_vocab, self.dims.n_hidden_text)
        self.positional_encoder = PositionalEncoding(d_model=self.dims.n_hidden_text, dropout=self.dims.drop, max_len=5000)
        self.ln = nn.LayerNorm(self.dims.n_hidden_audio)
        #self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.dims.n_hidden_audio, nhead=self.dims.n_head_audio, dim_feedforward=self.dims.n_feedforward, dropout=self.dims.drop, norm_first=self.dims.norm_first, batch_first=self.dims.batch_first),  self.dims.n_layer_audio,self.ln)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.dims.n_hidden_text, nhead=self.dims.n_head_text, dim_feedforward=self.dims.n_feedforward, dropout=self.dims.drop, batch_first=self.dims.batch_first, norm_first = self.dims.norm_first),  self.dims.n_layer_text, self.ln) 
        
        #self.classif_enc = nn.Linear(self.dims.n_hidden_audio,self.dims.n_vocab)
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
        
        
        #x_enc = self.Conv_subsampling(x_enc.permute(0,2,1))#.permute(0,2,1)
        #x_enc = x_enc.permute(0, 2, 1)
        #x_enc = x_enc * math.sqrt(self.dims.n_hidden_audio)
        
        #x_enc = self.positional_encoder(x_enc)
        
        
        
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
        
        #x_enc = self.Conv_subsampling(x_enc.permute(0,2,1))#.permute(0,2,1)
        
        
        #x_enc = x_enc * math.sqrt(self.dims.n_hidden_audio)
        
        #x_enc = self.positional_encoder(x_enc)
        
        
        
        enc_out = self.classif_enc(x_enc)
        enc_out= torch.nn.functional.log_softmax(enc_out,dim=-1)
        
        
        return enc_out
    
    def decod_audio(self, enc_out, target, tgt_mask = None, tgt_key_padding_mask = None):
        x_dec = self.token_embedding(target) * math.sqrt(self.dims.n_hidden_text)
        x_dec = self.positional_encoder(x_dec)
        dec_out = self.decoder(x_dec,enc_out , tgt_mask = tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        
        return self.classif_dec(dec_out)
        
        
        
    
    
    def forward(self, src, target, tgt_mask = None, tgt_key_padding_mask = None):
        
        if self.normalize_wav:
            src = F.layer_norm(src, src.shape[1:])
        
        x_enc = self.wav2vec(src)[0]
        
        
        
        #x_enc = self.Conv_subsampling(x_enc.permute(0,2,1))#.permute(0,2,1)
        #x_enc = x_enc.permute(0, 2, 1)
        
        
        
        x_dec = self.token_embedding(target) * math.sqrt(self.dims.n_hidden_text)
        x_dec = self.positional_encoder(x_dec)
        
        
        
        
        dec_out = self.decoder(x_dec, x_enc, tgt_mask = tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        
        
        #enc_out = self.classif_enc(x_enc)
        #enc_out=torch.nn.functional.log_softmax(enc_out,dim=-1)
        
        dec_out = self.classif_dec(dec_out)
        
        return dec_out, x_enc  #enc_out
    
if __name__ == '__main__':
    
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    dims = ModelDimensions(768, 3, 768, 8, 18, 1000, 768, 8, 6, 0.1, 768*4)
    #dims= ModelDimensions(768,3, 768, 4, 3,10,768,4,3,0.1,2048)
    trans = Seq2SeqTransformer(dims)
    yy = torch.randint(0,10,(16,30),dtype=torch.int64)
    mask_pad = trans.create_pad_mask(yy, 0)
    tgt_mask = trans.create_tgt_mask(yy.shape[1])
    
    x = torch.randn((16,64000))
    
    #y1 = trans.embed_audio(x)
    
    #y2 = trans.embed_audio_iCaRL(x)
    #mask_pad = trans.create_pad_mask(yy, 0)
    #tgt_mask = trans.create_tgt_mask(yy.shape[1])
    #print(tgt_mask.shape)
    
    #y= trans.decod_audio(y, yy,tgt_mask=tgt_mask ,tgt_key_padding_mask=mask_pad)
    #print(y[0,0,0:5])
    y,y2 = trans(x,yy,tgt_mask=tgt_mask ,tgt_key_padding_mask=mask_pad)
    print(y.shape)
        

