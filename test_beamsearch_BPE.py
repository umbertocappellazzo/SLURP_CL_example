#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 12:48:09 2022

@author: umbertocappellazzo
"""
import operator
import copy
from torch.utils.data import DataLoader
from functools import partial
from torch.optim import Adam, AdamW
from Speech_CLscenario.fluentspeech import FluentSpeech
from Speech_CLscenario.slurp_aug import Slurp
from Speech_CLscenario.class_incremental import ClassIncremental
#from continuum import ClassIncremental
#from continuum.datasets import FluentSpeech
import torch
import torch.nn.functional as F
import argparse
from continuum.metrics import Logger
import numpy as np
#from model_slurp import CL_model
#from model_transformer import CL_model, ModelDimensions
#from model_trans_wav2vec import Seq2SeqTransformer, ModelDimensions
from model_SpeechBrain import Seq2SeqTransformer, ModelDimensions
from tools.utils import get_kdloss,get_kdloss_onlyrehe, freeze_parameters
import time
import datetime
import json
import wandb
#from continuum import rehearsal
from Speech_CLscenario.memory import RehearsalMemory
from statistics import mean
import math
#from tools.utils import TextTransform
from torchaudio import transforms as tr
import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import torch.nn as nn
#import pytorch_warmup as warmup
import string
from torchmetrics import WordErrorRate, CharErrorRate
from torch import Tensor
from typing import Optional, Any, Union, Callable
import math
from tokenizers import Tokenizer
from queue import PriorityQueue
#from Levenshtein import distance
import jsonlines

def trunc(x,max_len):
    
    #l = len(x)
    #if l > max_len:
        #x = x[l//2-max_len//2:l//2+max_len//2]
    #    x = x[:max_len]
    #if l < max_len:
    #    x = F.pad(x, (0, max_len-l), value=0.)
        
   
    eps = np.finfo(np.float64).eps
    sample_rate = 16000
    n_mels = 80
    win_len = 20
    hop_len= 10
    win_len = int(sample_rate/1000*win_len)
    hop_len = int(sample_rate/1000*hop_len)
    mel_spectr = tr.MelSpectrogram(sample_rate=16000,
            win_length=win_len, hop_length=hop_len, n_mels=n_mels)
    
                # If I set 8 seconds as cutoff value, the output has 801 as temp size.                             
    
    
    
    #return np.log(mel_spectr(x)[:,:max_len//hop_len]+eps)
    #return np.log(mel_spectr(x)+eps)
    return mel_spectr(x+eps).log10()

    
class TextTransform:
    """Maps characters to integers and vice versa
    
    28 --> SOS (#)
    29 --> EOS (*)
    30 --> PAD (@)
    """
    
    def __init__(self):
        
        self.char_map_str = ["@","-","_",".",",","0","1","2","3","4","5","6","7","8","9",">","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","C","B","A","O"]
        
        self.char_map = {}
        self.index_map = {}
        for index,ch in enumerate(self.char_map_str):
        
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        #text = [char for char in text if char not in self.punctuations]
        
        for c in text:
            
            ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string_ = []
        for i in labels:
            string_.append(self.index_map[i])
        return ''.join(string_).replace('@', '')



def data_processing(data,max_len_audio, tokenizer, SOS_token=2, EOS_token=3, PAD_token=0):
    text_transform = TextTransform()
    
    SOS = torch.tensor([SOS_token])
    EOS = torch.tensor([EOS_token])
    
    # label_lengths are used if CTC loss is exploited in the experiments.
    
    #label_lengths = []
    transcripts_labels = [] 
    x = []
    #y = []
    #t= []
    audio_wavs = []
    #_,y,t, _ = zip(*data)
    #print(x[0].shape)
    
    
    #y = torch.tensor(np.array(y))
    #t = torch.tensor(t)
    
    for i in range(len(data)):
        audio_sig = data[i][0]
        
        if len(audio_sig) > max_len_audio:
            pass
        else:
            #spect = torch.tensor(trunc(audio_sig,max_len_audio))
            #x.append(spect.squeeze(0).transpose(0,1))
            x.append(audio_sig)
            
            
            transcript = data[i][3]
            #label = torch.tensor(text_transform.text_to_int(transcript))
            label = torch.tensor(tokenizer.encode(transcript).ids)
            
            #label = F.pad(label, (0, max_len_text-len(label)), value=PAD_token)
            label = torch.cat((SOS,label,EOS))
            
            audio_wav = data[i][4]
            audio_wavs.append(torch.tensor(text_transform.text_to_int(audio_wav)))
            
            #label_lengths.append(len(label))
            transcripts_labels.append(label)
            #y.append(torch.tensor(data[i][1]))
            #t.append(torch.tensor(data[i][2]))
    
    transcripts_labels = torch.nn.utils.rnn.pad_sequence(transcripts_labels, batch_first=True, padding_value=PAD_token)     #transcripts_labels = torch.stack(transcripts_labels)        #transcripts_labels = torch.nn.utils.rnn.pad_sequence(transcripts_labels, batch_first=True,padding_value=PAD_token)
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value = 0)      #x = torch.stack(x)      #x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    audio_wavs = torch.nn.utils.rnn.pad_sequence(audio_wavs, batch_first=True, padding_value=PAD_token) 
    #y = torch.stack(y)
    #t = torch.stack(t)
    
    #return x,y,t,transcripts_labels#,torch.tensor(label_lengths)
    return x,transcripts_labels, audio_wavs#,torch.tensor(label_lengths)

    
    
# def data_processing(data,max_len_audio, SOS_token=28, EOS_token=29, PAD_token=30):
#     text_transform = TextTransform()
    
#     SOS = torch.tensor([SOS_token])
#     EOS = torch.tensor([EOS_token])
    
#     # label_lengths are used if CTC loss is exploited in the experiments.
    
#     #label_lengths = []
#     transcripts_labels = [] 
#     x = []
#     #y = []
#     #t= []
    
#     #_,y,t, _ = zip(*data)
#     #print(x[0].shape)
    
    
#     #y = torch.tensor(np.array(y))
#     #t = torch.tensor(t)
    
#     for i in range(len(data)):
#         audio_sig = data[i][0]
        
#         if len(audio_sig) > max_len_audio:
#             pass
#         else:
#             #spect = torch.tensor(trunc(audio_sig,max_len_audio))
#             #x.append(spect.squeeze(0).transpose(0,1))
#             x.append(audio_sig)
            
            
#             transcript = data[i][3]
#             label = torch.tensor(text_transform.text_to_int(transcript))
            
            
#             #label = F.pad(label, (0, max_len_text-len(label)), value=PAD_token)
#             label = torch.cat((SOS,label,EOS))
            
#             #label_lengths.append(len(label))
#             transcripts_labels.append(label)
#             #y.append(torch.tensor(data[i][1]))
#             #t.append(torch.tensor(data[i][2]))
    
#     transcripts_labels = torch.nn.utils.rnn.pad_sequence(transcripts_labels, batch_first=True, padding_value=PAD_token)     #transcripts_labels = torch.stack(transcripts_labels)        #transcripts_labels = torch.nn.utils.rnn.pad_sequence(transcripts_labels, batch_first=True,padding_value=PAD_token)
#     x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value = 0)      #x = torch.stack(x)      #x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
#     #y = torch.stack(y)
#     #t = torch.stack(t)
    
#     #return x,y,t,transcripts_labels#,torch.tensor(label_lengths)
#     return x,transcripts_labels#,torch.tensor(label_lengths)    
    


# def data_processing(data,max_len_audio, max_len_text, SOS_token=28, EOS_token=29, PAD_token=30):
#     text_transform = TextTransform()
    
#     SOS = torch.tensor([SOS_token])
#     EOS = torch.tensor([EOS_token])
    
#     # label_lengths are used if CTC loss is exploited in the experiments.
    
#     #label_lengths = []
#     transcripts_labels = [] 
#     x = []
#     #y = []
#     #t= []
    
#     #_,y,t, _ = zip(*data)
#     #print(x[0].shape)
    
    
#     #y = torch.tensor(np.array(y))
#     #t = torch.tensor(t)
    
#     for i in range(len(data)):
#         transcript = data[i][3]
#         label = torch.tensor(text_transform.text_to_int(transcript))
#         if len(label) > max_len_text:
#             pass
#         else:
#             #label = F.pad(label, (0, max_len_text-len(label)), value=PAD_token)
#             label = torch.cat((SOS,label,EOS))
#             label = F.pad(label, (0, max_len_text-len(label)), value=PAD_token)
#             #label_lengths.append(len(label))
#             transcripts_labels.append(label)
#             audio_sig = data[i][0]
#             spect = torch.tensor(trunc(audio_sig,max_len_audio))
#             x.append(spect.squeeze(0).transpose(0,1))
#             #y.append(torch.tensor(data[i][1]))
#             #t.append(torch.tensor(data[i][2]))
    
#     transcripts_labels = torch.stack(transcripts_labels)        #transcripts_labels = torch.nn.utils.rnn.pad_sequence(transcripts_labels, batch_first=True,padding_value=PAD_token)
#     x = torch.stack(x)      #x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
#     #y = torch.stack(y)
#     #t = torch.stack(t)
    
#     #return x,y,t,transcripts_labels#,torch.tensor(label_lengths)
#     return x,transcripts_labels#,torch.tensor(label_lengths)



# Greedy decoder implementation. Decode a trascription by picking the most probable token for each text sequence frame.

def greedy_decode(model, input_sequence, device, max_length=130, SOS_token=28, EOS_token=29):
    model.eval()
    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)
    input_sequence = input_sequence.to(device)
    model = model.to(device)
    
    for _ in range(max_length):
        pred_output,_ = model(input_sequence,y_input)
        
        next_token = torch.argmax(pred_output[:, -1], keepdim=True)
        y_input = torch.cat((y_input, next_token), dim=1)
        
        if next_token.item() == EOS_token:
            break
     
    
    return y_input.tolist()[0]


# class BeamSearchNode(object):
#     def __init__(self, previousNode, wordId, logProb, length):
#         '''
#         :param hiddenstate:
#         :param previousNode:
#         :param wordId:
#         :param logProb:
#         :param length:
#         '''
#         #self.h = hiddenstate
#         self.prevNode = previousNode
#         self.wordid = wordId
#         self.logp = logProb
#         self.leng = length

#     def eval(self, alpha=1):
#         reward = 0
#         # Add here a function for shaping a reward
#         #pen = ((5 + self.leng) / (5 + 1)) ** alpha
#         #return self.logp/pen
#         return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


# def beam_decode(model, device, target_tensor, max_length=130, SOS_token=2, EOS_token=3, PAD_token=0, beam_size=10, pen_alpha=0.6,):
#     '''
#     :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
#     :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
#     :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
#     :return: decoded_batch
#     '''
    
#     model.eval()
#     model = model.to(device)
#     target_tensor.to(device)
    
#     beam_width = beam_size
#     topk = 1  # how many sentence do you want to generate
#     decoded_batch = []
    
    
#     encoder_outputs = model.embed_audio(target_tensor)
    
    

#     # decoding goes sentence by sentence
#     for idx in range(target_tensor.size(0)):
        
#         encoder_output = encoder_outputs[idx, :, :].unsqueeze(0)

#         # Start with the start of the sentence token
#         decoder_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

#         # Number of sentence to generate
#         endnodes = []
#         number_required = min((topk + 1), topk - len(endnodes))

#         # starting node -  hidden vector, previous node, word id, logp, length
#         node = BeamSearchNode(None, decoder_input, 0, 1)
#         nodes = PriorityQueue()

#         # start the queue
#         nodes.put((-node.eval(), node))
#         qsize = 1

#         # start beam search
#         while True:
#             # give up when decoding takes too long
#             if qsize > max_length: break

#             # fetch the best node
#             score, n = nodes.get()
#             decoder_input = n.wordid
            

#             if n.wordid.item() == EOS_token and n.prevNode != None:
#                 endnodes.append((score, n))
#                 # if we reached maximum # of sentences required
#                 if len(endnodes) >= number_required:
#                     break
#                 else:
#                     continue

#             # decode for one step using decoder
            
#             tgt_mask = model.create_tgt_mask(decoder_input.shape[1]).to(device)
#             tgt_key_padding_mask = model.create_pad_mask(decoder_input, PAD_token).to(device)
            
#             decoder_output = model.decod_audio(encoder_output,decoder_input, tgt_mask = tgt_mask, tgt_key_padding_mask = tgt_key_padding_mask)
#             #decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

#             # PUT HERE REAL BEAM SEARCH OF TOP
#             decoder_output = torch.log_softmax(decoder_output.squeeze(0), dim=-1)
            
#             log_prob, indexes = torch.topk(decoder_output, beam_width)
#             nextnodes = []

#             for new_k in range(beam_width):
#                 decoded_t = indexes[0][new_k].view(1, -1)
#                 log_p = log_prob[0][new_k].item()

#                 node = BeamSearchNode(n, decoded_t, n.logp + log_p, n.leng + 1)
#                 score = -node.eval()
#                 nextnodes.append((score, node))

#             # put them into queue
#             for i in range(len(nextnodes)):
#                 score, nn = nextnodes[i]
#                 nodes.put((score, nn))
#                 # increase qsize
#             qsize += len(nextnodes) - 1

#         # choose nbest paths, back trace them
#         if len(endnodes) == 0:
#             endnodes = [nodes.get() for _ in range(topk)]

#         utterances = []
#         for score, n in sorted(endnodes, key=operator.itemgetter(0)):
#             utterance = []
#             utterance.append(int(n.wordid))
#             # back trace
#             while n.prevNode != None:
#                 n = n.prevNode
#                 utterance.append(int(n.wordid))

#             utterance = utterance[::-1]
#             utterances.append(utterance)

#         decoded_batch.append(utterances)

#     return decoded_batch



def sequence_length_penalty(length: int, alpha: float=0.6) -> float:
    return ((5 + length) / (5 + 1)) ** alpha

def beam_search(model, input_sequence, device,vocab_size, max_length=130, SOS_token=2, EOS_token=3, PAD_token=0, beam_size=5, pen_alpha=0.6, return_best_beam = True):
    model.eval()
    model = model.to(device)
    
    beam_size = beam_size
    beam_size_count = beam_size
    pen_alpha = pen_alpha
    vocab_size = vocab_size
    
    decoder_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)
    scores = torch.Tensor([0.]).to(device)
    
    input_sequence = input_sequence.to(device)
    
    
    encoder_output = model.embed_audio(input_sequence)
    encoder_output_afterEOS = encoder_output
    final_scores = []
    final_tokens = []
    
    
    for i in range(max_length):
        
        tgt_mask = model.create_tgt_mask(decoder_input.shape[1]).to(device)
        tgt_key_padding_mask = model.create_pad_mask(decoder_input, PAD_token).to(device)
        
        logits= model.decod_audio(encoder_output,decoder_input, tgt_mask = tgt_mask, tgt_key_padding_mask = tgt_key_padding_mask)
        
        
        log_probs = torch.log_softmax(logits[:, -1], dim=1)
        log_probs = log_probs / sequence_length_penalty(i+1, pen_alpha)
        
    
        scores = scores.unsqueeze(1) + log_probs
        
        
        scores, indices = torch.topk(scores.reshape(-1), beam_size_count)
  
        beam_indices  = torch.divide(indices, vocab_size, rounding_mode='floor')
        token_indices = torch.remainder(indices, vocab_size) 
    
        next_decoder_input = []
        
        EOS_beams_index = []
        for ind, (beam_index, token_index) in enumerate(zip(beam_indices, token_indices)):
            
            
            prev_decoder_input = decoder_input[beam_index]
            
           
            if token_index == EOS_token:
                token_index = torch.LongTensor([token_index]).to(device)
                final_tokens.append(torch.cat([prev_decoder_input, token_index]))
                
                final_scores.append(scores[ind])
                beam_size_count -= 1
                encoder_output = encoder_output_afterEOS.expand(beam_size_count, *encoder_output_afterEOS.shape[1:])
                #scores_list = scores.tolist()
                #del scores_list[ind]
                #scores = torch.tensor(scores_list, device=device)
                EOS_beams_index.append(ind)
                #print(f"Beam #{ind} reached EOS!")
                
            else:
                token_index = torch.LongTensor([token_index]).to(device)
                next_decoder_input.append(torch.cat([prev_decoder_input, token_index]))
        
        if len(EOS_beams_index) > 0:
            scores_list = scores.tolist()
            for tt in EOS_beams_index[::-1]:
                del scores_list[tt]
            scores = torch.tensor(scores_list, device=device)
            
        if len(final_scores) == beam_size:
            break
        
        decoder_input = torch.vstack(next_decoder_input)
        

        if i==0:
            encoder_output = encoder_output.expand(beam_size_count, *encoder_output.shape[1:])
    
    
    if i == (max_length -1): # We have reached max # of allowed iterations.
    
        for beam_unf, score_unf in zip(decoder_input,scores):
            final_tokens.append(beam_unf)
            final_scores.append(score_unf)
        
        assert len(final_tokens) == beam_size and len(final_scores) == beam_size, ('Final_tokens and final_scores lists do not match beam_size size!')
       
            
            
    
    #decoder_output, _ = max(zip(decoder_input, scores), key=lambda x: x[1])
    #decoder_output = decoder_output[1:]
    #torch.set_printoptions(precision=10)
    #print(scores)
    
    #return decoder_input.tolist()  # If I want to return all beams
    #return decoder_output.tolist()
    
    # If we want to return most probable predicted beam.
    if return_best_beam:
        
        max_val = max(final_scores)
        return final_tokens[final_scores.index(max_val)].tolist()
    else:
        return final_tokens, final_scores
    

def rescoring(final_tokens, final_scores, model, audio_seq, loss_ctc, device, ctc_weight = 0.1):
    
    """
    This function takes in input a bunch of attentio-based hypothesis with the corresponding scores 
    (obtained with beam search). Then, each hypo is rescored based on the CTC probability using an interpolation 
    coeff. (ctc_weight). The best hypo will be returned.
    
    """
    
    model.eval()
    model = model.to(device)
    audio_seq = audio_seq.to(device)
    
    rescored_scores = []
    loss_ctc = loss_ctc 
    
    encoder_output_ctc = model.embed_audio_ctc(audio_seq)
    input_len = torch.full(size=(1,),fill_value=encoder_output_ctc.shape[1], dtype=torch.long)
    
    
    for tokens, score in zip(final_tokens, final_scores): 
        token_len = torch.tensor(len(tokens))
        score_ctc = loss_ctc(encoder_output_ctc.permute(1,0,2), tokens, input_len, token_len)
        score_ctc = torch.exp(-score_ctc)
        score = torch.exp(score)
        #print("CTC score: ", score_ctc)
        #print("Att score: ", score)
        
        final_score = score_ctc*ctc_weight + score*(1-ctc_weight)
        rescored_scores.append(final_score.detach().item())
    
    
    max_score = max(rescored_scores)
    return final_tokens[rescored_scores.index(max_score)].tolist()
        
        
        
        



    
# def beam_search(model, input_sequence, device,vocab_size, max_length=130, SOS_token=2, EOS_token=3, PAD_token=0, beam_size=20, pen_alpha=0.6, return_best_beam = True):
#     model.eval()
#     model = model.to(device)
    
#     beam_size = beam_size
#     beam_size_count = beam_size
#     pen_alpha = pen_alpha
#     vocab_size = vocab_size
    
#     decoder_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)
#     scores = torch.Tensor([0.]).to(device)
    
#     input_sequence = input_sequence.to(device)
    
    
#     encoder_output = model.embed_audio(input_sequence)
#     encoder_output_afterEOS = encoder_output
#     final_scores = []
#     final_tokens = []
    
    
#     for i in range(max_length):
        
#         tgt_mask = model.create_tgt_mask(decoder_input.shape[1]).to(device)
#         tgt_key_padding_mask = model.create_pad_mask(decoder_input, PAD_token).to(device)
        
#         logits= model.decod_audio(encoder_output,decoder_input, tgt_mask = tgt_mask, tgt_key_padding_mask = tgt_key_padding_mask)
        
        
#         log_probs = torch.log_softmax(logits[:, -1], dim=1)
#         log_probs = log_probs / sequence_length_penalty(i+1, pen_alpha)
        
    
#         scores = scores.unsqueeze(1) + log_probs
        
        
#         scores, indices = torch.topk(scores.reshape(-1), beam_size_count)
        
        
       
#         beam_indices  = torch.divide(indices, vocab_size, rounding_mode='floor')
#         token_indices = torch.remainder(indices, vocab_size) 
    
#         next_decoder_input = []
        
#         EOS_beams_index = []
#         for ind, (beam_index, token_index) in enumerate(zip(beam_indices, token_indices)):
            
            
#             prev_decoder_input = decoder_input[beam_index]
#             #if prev_decoder_input[-1]==EOS_token:
#             #    token_index = EOS_token
           
#             if token_index == EOS_token:
#                 token_index = torch.LongTensor([token_index]).to(device)
#                 final_tokens.append(torch.cat([prev_decoder_input, token_index]))
#                 #print(torch.cat([prev_decoder_input, token_index]))
#                 final_scores.append(scores[ind])
#                 beam_size_count -= 1
#                 encoder_output = encoder_output_afterEOS.expand(beam_size_count, *encoder_output_afterEOS.shape[1:])
#                 #scores_list = scores.tolist()
#                 #del scores_list[ind]
#                 #scores = torch.tensor(scores_list, device=device)
#                 EOS_beams_index.append(ind)
#                 #print(f"Beam #{ind} reached EOS!")
                
#             else:
#                 token_index = torch.LongTensor([token_index]).to(device)
#                 next_decoder_input.append(torch.cat([prev_decoder_input, token_index]))
#         #print(decoder_input)
#         if len(EOS_beams_index) >0:
#             scores_list = scores.tolist()
#             for tt in EOS_beams_index[::-1]:
#                 del scores_list[tt]
#             scores = torch.tensor(scores_list, device=device)
            
#         if len(final_scores) == beam_size:
#             break
        
#         decoder_input = torch.vstack(next_decoder_input)
        
#         #print(decoder_input)
        
        
        
#         #if (decoder_input[:, -1]==EOS_token).sum() == beam_size:
#         #    break
        
#         if i==0:
#             encoder_output = encoder_output.expand(beam_size, *encoder_output.shape[1:])
    
    
#     if i == (max_length -1): # We have reached max # of allowed iterations.
    
#         for beam_unf, score_unf in zip(decoder_input,scores):
#             final_tokens.append(beam_unf.tolist())
#             final_scores.append(score_unf)
        
#         assert len(final_tokens) == beam_size and len(final_scores) == beam_size, ('Final_tokens and final_scores lists do not match beam_size size!')
       
            
            
    
#     #decoder_output, _ = max(zip(decoder_input, scores), key=lambda x: x[1])
#     #decoder_output = decoder_output[1:]
#     #torch.set_printoptions(precision=10)
#     #print(scores)
    
#     #return decoder_input.tolist()  # If I want to return all beams
#     #return decoder_output.tolist()
    
#     # If we want to return most probable predicted beam.
#     if return_best_beam:
        
#         max_val = max(final_scores)
#         return final_tokens[final_scores.index(max_val)].tolist()
#     else:
#         return final_tokens, final_scores
    
    
  


def get_args_parser():
    parser = argparse.ArgumentParser('CiCL for Spoken Language Understandig (Intent classification) on FSC: train and evaluation',
                                     add_help=False)
    
    # Dataset parameters.
    
    parser.add_argument('--data_path', type=str, default='/data/cappellazzo/CL_SLU',help='path to dataset')  #'/data/cappellazzo/slurp'  /data/cappellazzo/CL_SLU
    parser.add_argument('--path_to_best_model',type=str,default='/models_SLURP_wavx_SpeechBrain_intentsaug/model_SF_ep42.pth')
    parser.add_argument('--max_len_audio', type=int, default=112000, 
                        help='max length for the audio signal --> it will be cut')
    parser.add_argument('--max_len_text', type=int, default=130, 
                        help='max length for the text sequence. This size includes the EOS or SOS token that are appended for the model computation.')
    parser.add_argument('--download_dataset', default=False, action='store_true',
                        help='whether to download the FSC dataset or not')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type= str, default='cuda', 
                        help='device to use for training/testing')
    
    
    # Training/inference parameters.
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4, 
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--eval_every", type=int, default=1, 
                        help="Eval model every X epochs")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--label_smoothing', type=float, default=0., 
                        help='Label smoothing for the CE loss')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_ratio', type=int, default=10)
    
    
    # Encoder hyperparameters.
    
    parser.add_argument('--n_mels', type=int, default=768)
    parser.add_argument('--n_seq_audio', type=int, default=400,
                        help='Sequence length')
    parser.add_argument('--n_hidden_audio', type=int, default=768,
                        help='Transformer encoder hidden dim')
    parser.add_argument('--n_head_audio', type=int, default=8,)
    parser.add_argument('--kernel_size', type=int, default=3,)
    parser.add_argument('--n_layer_audio', type=int, default=18)
    parser.add_argument('--drop', type=float, default=0.1)
    parser.add_argument('--n_feedforward', type=int, default=768*4)
    # Decoder hyperparameters.
    
    parser.add_argument('--n_vocab', type=int, default=1000)
    parser.add_argument('--n_seq_text', type=int, default=16,
                        help='Sequence length')
    parser.add_argument('--n_hidden_text', type=int, default=768,
                        help='Transformer encoder hidden dim')
    parser.add_argument('--n_head_text', type=int, default=8,)
    parser.add_argument('--n_layer_text', type=int, default=6)
    
    # Rehearsal memory.
    
    parser.add_argument('--memory_size', default=0, type=int,
                        help='Total memory size in number of stored samples.')
    parser.add_argument('--fixed_memory', default=True,
                        help='Dont fully use memory when no all classes are seen')
    parser.add_argument('--herding', default="barycenter",
                        choices=[
                            'random',
                            'cluster', 'barycenter',
                        ],
                        help='Method to herd sample for rehearsal.')
    
    # DISTILLATION parameters.
    
    parser.add_argument('--distillation-tau', default=1.0, type=float,
                        help='Temperature for the KD')
    parser.add_argument('--feat_space_kd', default=None, choices=[None,'only_rehe','all'])
    parser.add_argument('--preds_space_kd', default=None, choices=[None,'only_rehe','all'])
    
    # Continual learning parameters.
    
    parser.add_argument('--increment', type=int, default=3, 
                        help='# of classes per task/experience')
    parser.add_argument('--initial_increment', type=int, default=3, 
                        help='# of classes for the 1st task/experience')
    parser.add_argument('--nb_tasks', type=int, default=6, 
                        help='the scenario number of tasks')
    parser.add_argument('--offline_train', default=True, action='store_true',
                        help='whether to train in an offline fashion (i.e., no CL setting)')
    parser.add_argument('--total_classes', type=int, default= 18, 
                        help='The total number of classes when we train in an offline i.i.d. fashion. Set to None otherwise.')
    parser.add_argument('--nb_classes_noCL', type=int, default= 54, 
                        help='The total number of classes when we train in an offline i.i.d. fashion. Set to None otherwise.')
    # WANDB parameters.
    
    parser.add_argument('--use_wandb', default=True, action='store_false',
                        help='whether to track experiments with wandb')
    parser.add_argument('--project_name', type=str, default='SLURP_experiments')
    parser.add_argument('--exp_name', type=str, default='test_BPE_mymod_epoch474849')
    
    
    return parser



def main(args):
    
    if args.use_wandb:
        
        wandb.init(project=args.project_name, name=args.exp_name,entity="umbertocappellazzo",
                   )
   
    print(args)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    
    
    

    
    device = torch.device(args.device)
    torch.set_num_threads(20)
    
    # Fix the seed for reproducibility (if desired).
    #seed = args.seed
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    
    # Create the train and test dataset splits + corresponding CiCL scenarios. 
    
    #dataset_train = FluentSpeech(args.data_path,train=True,download=False)
    #dataset_valid = FluentSpeech(args.data_path,train="valid",download=False)
    #dataset_test = FluentSpeech(args.data_path,train=False,download=False)
    
    dataset_train = Slurp(args.data_path, max_len_text=args.max_len_text,max_len_audio=args.max_len_audio,train="train",download=False)
    dataset_valid = Slurp(args.data_path, max_len_text=args.max_len_text,max_len_audio=args.max_len_audio,train="valid",download=False)
    dataset_test = Slurp(args.data_path, max_len_text=args.max_len_text,max_len_audio=args.max_len_audio,train=False,download=False)
    
    
    
    
    text_transform = TextTransform()
    # Define the order in which the classes will be spread through the CL tasks.
    # In my experiments, I use this config and the [0,1,2,3,...] config. Just remove the 
    # class_order parameter from the scenario definition to get the latter config.
    
    #class_order = [19, 27, 30, 28, 15,  4,  2,  9, 10, 22, 11,  7,  1, 25, 16, 14,  5,
    #         8, 29, 12, 21, 17,  3, 20, 23,  6, 18, 24, 26,  0, 13]
    
    
    
    if args.offline_train:   # Create just 1 task with all classes.
       
        scenario_train = ClassIncremental(dataset_train,nb_tasks=1,splitting_crit=None)#,transformations=[partial(trunc, max_len=args.max_len)])
        scenario_valid = ClassIncremental(dataset_valid,nb_tasks=1,splitting_crit=None)#,transformations=[partial(trunc, max_len=args.max_len)])
        scenario_test = ClassIncremental(dataset_test,nb_tasks=1,splitting_crit=None)
    else:
        
        scenario_train = ClassIncremental(dataset_train,increment=args.increment,initial_increment=args.initial_increment,)
                                          #class_order=class_order)
        scenario_test = ClassIncremental(dataset_valid,increment=args.increment,initial_increment=args.initial_increment,)
                                         #class_order=class_order)
    
    # Losses employed: CE + MSE.
    
    
    
    WER = WordErrorRate()
    CER = CharErrorRate()
    
    #criterion_ctc = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity =True)
    # Use all the classes for offline training.
    
    #initial_classes = args.total_classes if args.offline_train else args.initial_increment
    
    
    start_time = time.time()
    
  
    for task_id, exp_train in enumerate(scenario_train):
        
        print("Shape of exp_train: ",len(exp_train))
        print(f"Starting task id {task_id}/{len(scenario_train) - 1}")
        
       
        if task_id == 0:
            #text_transf = TextTransform()
            #vocab_size = len(text_transf.char_map_str)
            print('Creating the CL model:')
            #For CTC
            
            # model = CL_model(initial_classes,vocab_size,in_chan=args.in_chan, n_blocks=args.n_blocks, n_repeats=args.n_repeats,
            #                  out_chan=args.out_chan, hid_chan=args.hid_chan,kernel_size=args.kernel_size,
            #                  device=device).to(device)   
            
            # W/out CTC
            # model = CL_model(initial_classes,args.nb_classes_noCL,in_chan=args.in_chan, n_blocks=args.n_blocks, n_repeats=args.n_repeats,
            #                   out_chan=args.out_chan, hid_chan=args.hid_chan,kernel_size=args.kernel_size,
            #                   device=device).to(device)  
            
            
            #dims = ModelDimensions(args.n_mels, args.kernel_size, args.n_seq_audio, args.n_hidden_audio, args.n_head_audio, args.n_layer_audio, args.n_vocab, args.n_seq_text-1, args.n_hidden_text, args.n_head_text, args.n_layer_text, args.drop)
            
            dims = ModelDimensions(args.n_mels, args.kernel_size, args.n_hidden_audio, args.n_head_audio, args.n_layer_audio, args.n_vocab, args.n_hidden_text, args.n_head_text, args.n_layer_text, args.drop, args.n_feedforward)
            model = Seq2SeqTransformer(dims, device=device).to(device)
            
            assert (args.n_hidden_audio % args.n_head_audio) == 0, ('Hidden dimension of encoder must be divisible by number of audio heads!')
            assert (args.n_hidden_text % args.n_head_text) == 0, ('Hidden dimension of decoder must be divisible by number of text heads!')
            
            #model = CL_model(initial_classes,args.nb_classes_noCL, dims, device=device).to(device)
            #model = nn.DataParallel(model)
            
            
           
            
            
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('Number of params of the model:', n_parameters)
            
        else:
            
            print(f'Updating the CL model, {args.increment} new classes for the classifier.')
            #model.classif_intent.add_new_outputs(args.increment)  
            model.classif.add_new_outputs(args.increment)  
            
        # IN THE CODE LINE BELOW, specify the saved model with the highest valid accuracy among the various epochs. 
        
        path_model1 = os.getcwd() +   args.path_to_best_model    #'/models_SLURP_wav_rescoring/model_BPE1000_wav_rescoring_ep8.pth'    
        #print(path_model)
        #path_model2 = os.getcwd() + '/models_SLURP_wav_SpeechBrain_intentsaug/model_SF_ep48.pth'
        #path_model3 = os.getcwd() + '/models_SLURP_wav_SpeechBrain_intentsaug/model_SF_ep49.pth'
        #path_model4 = os.getcwd() + '/models_SLURP_wav_rescoring/model_BPE1000_wav_rescoring_ep15.pth'
        #path_model5 = os.getcwd() + '/models_SLURP_wav_rescoring/model_BPE1000_wav_rescoring_ep18.pth'
        
        
        
        
        model.load_state_dict(torch.load(path_model1))
        #m1 = model.state_dict()
        
        #model.load_state_dict(torch.load(path_model2))
        #m2 = model.state_dict()
        
        #model.load_state_dict(torch.load(path_model3))
        #m3 = model.state_dict()
        
        #model.load_state_dict(torch.load(path_model4))
        #m4 = model.state_dict()
        
        #model.load_state_dict(torch.load(path_model5))
        #m5 = model.state_dict()
        
        
        #for key in m2:
        #    m1[key] +=  m2[key] + m3[key] #+ m4[key] + m5[key]
        #for key in m1:
        #    m1[key] /= 3
        
        #model.load_state_dict(m1)
        
        
        path_2_tok = os.getcwd() + '/tokenizer_SLURP_BPE_1000_noblank_intents_SFaug.json'
        tokenizer = Tokenizer.from_file(path_2_tok)
       
        test_taskset = scenario_test[:task_id+1]    # Evaluation on all seen tasks.
        #valid_taskset = scenario_valid[:task_id+1]
        
        #train_loader = DataLoader(exp_train, batch_size=args.batch_size, shuffle=True, 
        #                          num_workers=args.num_workers,pin_memory=True, drop_last=False,collate_fn=lambda x: data_processing(x,args.max_len_audio,args.max_len_text-1)) #,collate_fn=lambda x: data_processing(x)
        #valid_loader = DataLoader(valid_taskset, batch_size=args.batch_size, shuffle=True, 
        #                         num_workers=args.num_workers, pin_memory=True, drop_last=False,collate_fn=lambda x: data_processing(x,args.max_len_audio,args.max_len_text-1))
        test_loader = DataLoader(test_taskset, batch_size=64, shuffle=False, num_workers=args.num_workers,
                                 collate_fn=lambda x: data_processing(x,args.max_len_audio,tokenizer),pin_memory=True, drop_last=False,)
        
        print(len(test_loader))
    ###########################################################################
    #                                                                         #
    # Begin of the train and test loops                                       #
    #                                                                         #
    ###########################################################################
   
       
   
    
   
        # reference = ['alarm_query', 'alarm_remove', 'alarm_set',
        #     'audio_volume_down', 'audio_volume_mute','audio_volume_other','audio_volume_up',
        #     'calendar_query', 'calendar_remove','calendar_set',
        #     'cooking_query', 'cooking_recipe',
        #     'datetime_convert', 'datetime_query',
        #     'email_addcontact', 'email_query','email_querycontact', 'email_sendemail',
        #     'general_affirm','general_commandstop','general_confirm','general_dontcare','general_explain','general_greet','general_joke','general_negate','general_praise','general_quirky','general_repeat',
        #     'iot_cleaning','iot_coffee','iot_hue_lightchange','iot_hue_lightdim','iot_hue_lightoff','iot_hue_lighton','iot_hue_lightup','iot_wemo_off','iot_wemo_on',
        #     'lists_createoradd','lists_query','lists_remove',
        #     'music_dislikeness','music_likeness','music_query','music_settings',
        #     'news_query',
        #     'play_audiobook','play_game','play_music','play_podcasts','play_radio',
        #     'qa_currency','qa_definition','qa_factoid','qa_maths','qa_query','qa_stock',
        #     'recommendation_events', 'recommendation_locations','recommendation_movies',
        #     'social_post','social_query',
        #     'takeaway_order','takeaway_query',
        #     'transport_query','transport_taxi','transport_ticket','transport_traffic',
        #     'weather_query']
        
        # reference_dict = {'alarm_query':0, 'alarm_remove':1, 'alarm_set':2,
        #     'audio_volume_down':3, 'audio_volume_mute':4,'audio_volume_other':5,'audio_volume_up':6,
        #     'calendar_query':7, 'calendar_remove':8,'calendar_set':9,
        #     'cooking_query':10, 'cooking_recipe':11,
        #     'datetime_convert':12, 'datetime_query':13,
        #     'email_addcontact':14, 'email_query':15,'email_querycontact':16, 'email_sendemail':17,
        #     'general_affirm':18,'general_commandstop':19,'general_confirm':20,'general_dontcare':21,'general_explain':22,'general_greet':23,'general_joke':24,'general_negate':25,'general_praise':26,'general_quirky':27,'general_repeat':28,
        #     'iot_cleaning':29,'iot_coffee':30,'iot_hue_lightchange':31,'iot_hue_lightdim':32,'iot_hue_lightoff':33,'iot_hue_lighton':34,'iot_hue_lightup':35,'iot_wemo_off':36,'iot_wemo_on':37,
        #     'lists_createoradd':38,'lists_query':39,'lists_remove':40,
        #     'music_dislikeness':41,'music_likeness':42,'music_query':43,'music_settings':44,
        #     'news_query':45,
        #     'play_audiobook':46,'play_game':47,'play_music':48,'play_podcasts':49,'play_radio':50,
        #     'qa_currency':51,'qa_definition':52,'qa_factoid':53,'qa_maths':54,'qa_query':55,'qa_stock':56,
        #     'recommendation_events':57, 'recommendation_locations':58,'recommendation_movies':59,
        #     'social_post':60,'social_query':61,
        #     'takeaway_order':62,'takeaway_query':63,
        #     'transport_query':64,'transport_taxi':65,'transport_ticket':66,'transport_traffic':67,
        #     'weather_query':68}
   
    
                
        list_preds_test = []
        list_gold_test = []
            
        correct = 0
        total = 0      
        preds_jsonl = []
        with torch.inference_mode():
                  
            for idx_test_batch, (x_test,text_test,ids) in enumerate(test_loader):
                    print("Testing batch #: ", idx_test_batch)
                    x_test = x_test.to(device)
                    #x_test = x_test.transpose(1,2)
                    text_test = text_test.to(device)
                    text_to_loss_test = text_test[:,1:]
                    
                    
                    
                    
                    for x in range(x_test.shape[0]):
                        current_dict = {}
                        wav_id = text_transform.int_to_text(ids[x,:].tolist())
                        current_dict["file"] = wav_id
                        
                        
                        pred_token = beam_search(model, x_test[x,:].unsqueeze(0), device, args.n_vocab, beam_size=20,return_best_beam= True)
                        #pred_token = rescoring(final_tokens, final_scores, model, x_test[x,:].unsqueeze(0), criterion_ctc, device, ctc_weight = 0.3)
                        
                        print(pred_token)
                        
                    
                        list_preds_test.append(tokenizer.decode(pred_token))
                        gold_pred = tokenizer.decode(text_to_loss_test[x,:].tolist())
                        list_gold_test.append(gold_pred)
                        
                        transc_tok = list(map(str.strip,tokenizer.decode(pred_token).split('_SEP')))
                        pred_intent = transc_tok[0]
                        
                        if len(pred_intent.split("_")) == 1:
                            current_dict["scenario"] = pred_intent.split("_")[0]
                            current_dict["action"] = pred_intent.split("_")[0]
                        else:
                        
                            current_dict["scenario"] = pred_intent.split("_")[0]
                            current_dict["action"] = pred_intent.split("_")[1] if len(pred_intent.split("_")) == 2 else pred_intent.split("_")[1] + '_' + pred_intent.split("_")[2]
                        
                        entities = []
                        
                        for entity in transc_tok[1:-1]:
                            if len(entity.split("_FILL")) != 2:
                                continue
                            ent_type = entity.split("_FILL")[0].strip()
                            ent_val = entity.split("_FILL")[1].strip()
                            
                            dict_ = {}
                            dict_["type"] = ent_type
                            dict_["filler"] = ent_val
                            
                            entities.append(dict_)
                        
                        current_dict["entities"] = entities
                        preds_jsonl.append(current_dict)
                        
                        
                        gold_intent = list(map(str.strip,gold_pred.split('_SEP')))[0]
                        #pred_intent =list(map(str.strip,tokenizer.decode(pred_token).split('_SEP')))[0]
                        #lev_dist = [distance(pred_intent,xx) for xx in reference]
                        
                        #gold_intent = list(map(str.strip,gold_pred.split('_SEP')))[0]
                        #pred_index = np.argmin(lev_dist)
                       
                        
                        if pred_intent == gold_intent:
                        #if reference[pred_index] == gold_intent:
                            correct += 1
                        total +=1
                        
                        
                        if x == 0:
                            print('True: ', gold_pred)
                            print("Predicted: ",pred_intent)
        
        
        file_name = "pred_outputs_offline"
        with jsonlines.open(file_name,'w') as writer:
            writer.write_all(preds_jsonl)
        
        
        print(f"TEST WER: {WER(list_preds_test,list_gold_test)}")
        print(f"TEST CER: {CER(list_preds_test,list_gold_test)}")
            
        print(f"Total corrected intent predictions: {correct} out of {total}")
        print("Intent Accuracy: ", correct/total)
        
        #torch.save(list_preds_test,'list_preds_test.pt')
        #torch.save(list_gold_test,'list_gold_test.pt')
            
                    
                  
        
     
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    if args.use_wandb:
        wandb.finish() 
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser('CiCL for Spoken Language Understandig (Intent classification) on FSC: train and evaluation',
                                    parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
   
   





    
    
    
    
    
        
        