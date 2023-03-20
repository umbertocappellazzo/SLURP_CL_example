#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:24:34 2023

@author: umbertocappellazzo
"""

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
#from model_confenc_wav import Seq2SeqTransformer, ModelDimensions
from model_SpeechBrain import Seq2SeqTransformer, ModelDimensions
from tools.utils import freeze_parameters
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
from torch.optim.lr_scheduler import LambdaLR
from tokenizers import Tokenizer
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
    return mel_spectr(x+eps).log()


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
    y = []
    #t= []
    audio_wavs = []
    #_,y,t, _ = zip(*data)
    #print(x[0].shape)
    
    
    #y = torch.tensor(np.array(y))
    #t = torch.tensor(t)
    
    for i in range(len(data)):
        audio_sig = data[i][0]
        #print("Wav file: ",data[i][4])
        
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
            y.append(torch.tensor(data[i][1]))
            #t.append(torch.tensor(data[i][2]))
    
    transcripts_labels = torch.nn.utils.rnn.pad_sequence(transcripts_labels, batch_first=True, padding_value=PAD_token)     #transcripts_labels = torch.stack(transcripts_labels)        #transcripts_labels = torch.nn.utils.rnn.pad_sequence(transcripts_labels, batch_first=True,padding_value=PAD_token)
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value = 0)      #x = torch.stack(x)      #x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    y = torch.stack(y)
    #t = torch.stack(t)
    audio_wavs = torch.nn.utils.rnn.pad_sequence(audio_wavs, batch_first=True, padding_value=PAD_token) 
    #return x,y,t,transcripts_labels#,torch.tensor(label_lengths)
    return x,y,transcripts_labels, audio_wavs#,torch.tensor(label_lengths)


    
def sequence_length_penalty(length: int, alpha: float=0.6) -> float:
    return ((5 + length) / (5 + 1)) ** alpha


def beam_search(model, input_sequence, device,vocab_size, max_length=130, SOS_token=2, EOS_token=3, PAD_token=0, beam_size=20, pen_alpha=0.6, return_best_beam = True):
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
            #if prev_decoder_input[-1]==EOS_token:
            #    token_index = EOS_token
           
            if token_index == EOS_token:
                token_index = torch.LongTensor([token_index]).to(device)
                final_tokens.append(torch.cat([prev_decoder_input, token_index]))
                #print(torch.cat([prev_decoder_input, token_index]))
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
        #print(decoder_input)
        if len(EOS_beams_index) >0:
            scores_list = scores.tolist()
            for tt in EOS_beams_index[::-1]:
                del scores_list[tt]
            scores = torch.tensor(scores_list, device=device)
            
        if len(final_scores) == beam_size:
            break
        
        decoder_input = torch.vstack(next_decoder_input)
        
        #print(decoder_input)
        
        
        
        #if (decoder_input[:, -1]==EOS_token).sum() == beam_size:
        #    break
        
        if i==0:
            encoder_output = encoder_output.expand(beam_size_count, *encoder_output.shape[1:])
    
    
    if i == (max_length -1): # We have reached max # of allowed iterations.
    
        for beam_unf, score_unf in zip(decoder_input,scores):
            final_tokens.append(beam_unf)
            final_scores.append(score_unf)
        
        assert len(final_tokens) == beam_size and len(final_scores) == beam_size, ('Final_tokens and final_scores lists do not match beam_size size!')
       
            
            
    
    
    # If we want to return the most probable predicted beam.
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
        
  


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        #print("Current LR: ", rate)
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_args_parser():
    parser = argparse.ArgumentParser('CiCL for Spoken Language Understandig (Intent classification) on FSC: train and evaluation',
                                     add_help=False)
    
    # Dataset parameters.
    
    parser.add_argument('--data_path', type=str, default='/data/cappellazzo/CL_SLU',help='path to dataset')  #'/data/cappellazzo/slurp'  /data/cappellazzo/CL_SLU
    parser.add_argument('--max_len_audio', type=int, default=112000, 
                        help='max length for the audio signal --> it will be cut')
    parser.add_argument('--max_len_text', type=int, default=130, 
                        help='max length for the text sequence. This size includes the EOS or SOS token that are appended for the model computation.')
    parser.add_argument('--download_dataset', default=False, action='store_true',
                        help='whether to download the FSC dataset or not')
    parser.add_argument('--seed', type=int, default=11)   #11
    parser.add_argument('--device', type= str, default='cuda', 
                        help='device to use for training/testing')
    
    
    # Training/inference parameters.
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5, 
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--eval_every", type=int, default=1, 
                        help="Eval model every X epochs")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--label_smoothing', type=float, default=0., 
                        help='Label smoothing for the CE loss')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_ratio', type=int, default=1)
    
    
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
    
    # Decoder hyperparameters.
    
    parser.add_argument('--n_vocab', type=int, default=1000)
    #parser.add_argument('--n_seq_text', type=int, default=16,
    #                    help='Sequence length')
    parser.add_argument('--n_hidden_text', type=int, default=768,
                        help='Transformer encoder hidden dim')
    parser.add_argument('--n_head_text', type=int, default=8,)
    parser.add_argument('--n_layer_text', type=int, default=6)
    parser.add_argument('--n_feedforward', type=int, default=768*4)
    parser.add_argument('--feed_forward_expansion_factor', type=int, default=4)
    parser.add_argument('--PAD_token', default=0)
    
    # Rehearsal memory.
    
    parser.add_argument('--memory_size', default=1260, type=int,
                        help='Total memory size in number of stored samples.')
    parser.add_argument('--fixed_memory', default=True,
                        help='Dont fully use memory when no all classes are seen')
    parser.add_argument('--herding', default="random",
                        choices=[
                            'random',
                            'cluster', 'barycenter',
                        ],
                        help='Method to herd sample for rehearsal.')
    
    # DISTILLATION parameters.
    
    parser.add_argument('--distillation-tau', default=1.0, type=float,
                        help='Temperature for the KD')
    parser.add_argument('--use_kd_audio', default=False,)
    parser.add_argument('--use_kd_token', default=False)
    parser.add_argument('--use_kd_token_all', default=False)
    parser.add_argument('--use_kd_seq', default=False)
    parser.add_argument('--use_kd_cross', default= False)

    
    
    # Continual learning parameters.
    
    parser.add_argument('--increment', type=int, default=3, 
                        help='# of classes per task/experience')
    parser.add_argument('--initial_increment', type=int, default=3, 
                        help='# of classes for the 1st task/experience')
    parser.add_argument('--nb_tasks', type=int, default=6, 
                        help='the scenario number of tasks')
    parser.add_argument('--offline_train', default=True, action='store_false',
                        help='whether to train in an offline fashion (i.e., no CL setting)')
    parser.add_argument('--total_classes', type=int, default= 18, 
                        help='The total number of classes when we train in an offline i.i.d. fashion. Set to None otherwise.')
    #parser.add_argument('--nb_classes_noCL', type=int, default= 54, 
    #                    help='The total number of classes when we train in an offline i.i.d. fashion. Set to None otherwise.')
    # WANDB parameters.
    
    parser.add_argument('--use_wandb', default=False, action='store_false',
                        help='whether to track experiments with wandb')
    parser.add_argument('--project_name', type=str, default='SLURP_experiments')
    parser.add_argument('--exp_name', type=str, default='SLURP_rehe1%icarl_6tasks_KDtokrehe_decorder_seed11_paper') #KDaudio  rehe2%random   rehe2%random+KDaudio
    
    
    return parser
    




def main(args):
    
    if args.use_wandb:
        
        wandb.init(project=args.project_name, name=args.exp_name,entity="umbertocappellazzo",
                   )
    
    print(args)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    
    WER = WordErrorRate()
    CER = CharErrorRate()
  
    device = torch.device(args.device)
    torch.set_num_threads(10)
    
    # Fix the seed for reproducibility (if desired).
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    
    text_transform = TextTransform()
    
    dataset_train = Slurp(args.data_path, max_len_text=args.max_len_text, max_len_audio=args.max_len_audio, train="train", download=False)
    
    
    dataset_valid = Slurp(args.data_path, max_len_text=args.max_len_text, max_len_audio=args.max_len_audio, train="valid", download=False)
    dataset_test = Slurp(args.data_path, max_len_text=args.max_len_text, max_len_audio=args.max_len_audio, train=False, download=False)
    
    class_order = [2,5,6,11,12,14,0,1,3,4,7,8,9,10,13,15,16,17]  # Most populated scenarios come first.
    #class_order = [11, 13, 3, 7, 16, 12, 2, 9, 10, 1, 15, 8, 6, 14, 5, 4, 17, 0]  # Random order.
   
    if args.offline_train:   # Create just 1 task with all classes.
       
        scenario_train = ClassIncremental(dataset_train,nb_tasks=1)#,transformations=[partial(trunc, max_len=args.max_len)])
        scenario_valid = ClassIncremental(dataset_valid,nb_tasks=1)#,transformations=[partial(trunc, max_len=args.max_len)])
        scenario_test = ClassIncremental(dataset_test,nb_tasks=1)
    else:
        
        scenario_train = ClassIncremental(dataset_train,increment=args.increment,initial_increment=args.initial_increment,
                                          class_order=class_order,splitting_crit=None)
        scenario_valid = ClassIncremental(dataset_valid,increment=args.increment,initial_increment=args.initial_increment,
                                         class_order=class_order,splitting_crit=None)
        scenario_test = ClassIncremental(dataset_test,increment=args.increment,initial_increment=args.initial_increment,
                                         class_order=class_order,splitting_crit=None)
    
    
    criterion_text = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)#,ignore_index=30)
    #criterion_intent = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_mse = torch.nn.MSELoss()
    #criterion_ctc = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity =True)
    
    teacher_model = None
    
    memory = None
    if args.memory_size > 0:
        memory = RehearsalMemory(args.memory_size, herding_method= args.herding, 
                                           fixed_memory=args.fixed_memory, nb_total_classes=scenario_train.nb_classes,splitting_crit=None)
        
    
    # Use all the classes for offline training.
    
    #initial_classes = args.total_classes if args.offline_train else args.initial_increment
    
    
    start_time = time.time()
    
    #epochs = args.epochs
    epochs = [40,25,15,15,15,15]
    
    
   
    ###########################################################################
    #                                                                         #
    # Begin of the task loop                                                  #
    #                                                                         #
    ###########################################################################
    
    IC_ACC_overall = []
    WER_overall = []
    CER_overall = []
    
    for task_id, exp_train in enumerate(scenario_train):
        
        print("Shape of exp_train: ",len(exp_train))
        print(f"Starting task id {task_id}/{len(scenario_train) - 1}")
        
        
        if (args.use_kd_audio or args.use_kd_token or args.use_kd_token_all or args.use_kd_seq or args.use_kd_cross) and task_id > 0:
            teacher_model = copy.deepcopy(model)
            freeze_parameters(teacher_model)
            teacher_model.eval()
        
        if task_id > 0 and memory is not None:
            exp_train.add_samples(*memory.get())
        
        if task_id == 0:
            
            print('Creating the CL model:')
            
            dims = ModelDimensions(args.n_mels, args.kernel_size, args.n_hidden_audio, args.n_head_audio, args.n_layer_audio, args.n_vocab, args.n_hidden_text, args.n_head_text, args.n_layer_text, args.drop, args.n_feedforward)
            
            assert (args.n_hidden_audio % args.n_head_audio) == 0, ('Hidden dimension of encoder must be divisible by number of audio heads!')
            assert (args.n_hidden_text % args.n_head_text) == 0, ('Hidden dimension of decoder must be divisible by number of text heads!')
            
            model = Seq2SeqTransformer(dims, device=device).to(device)
            #model = nn.DataParallel(model)
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('Number of params of the model:', n_parameters)
            
        # We don't have a classifier to update anymore.
        # else:
            
        #     print(f'Updating the CL model, {args.increment} new classes for the classifier.')
        #     #model.classif_intent.add_new_outputs(args.increment)  
        #     model.classif.add_new_outputs(args.increment)  
            
            
        
        
        path_2_tok = os.getcwd() + '/tokenizer_SLURP_BPE_1000_noblank_intents_SFaug.json'
        tokenizer = Tokenizer.from_file(path_2_tok)
        
        
        optimizer = AdamW(model.parameters(),lr=args.lr,betas=(0.9,0.98),eps=1e-6,weight_decay=args.weight_decay)
        #optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9,0.98), eps=1e-8, weight_decay=args.weight_decay)
        
        test_taskset = scenario_test[:task_id+1]    # Evaluation on all seen tasks.
        valid_taskset = scenario_valid[:task_id+1]
        
        
        alpha = task_id/(task_id+1)
        
        # I create a dictionare whose keys are the wav ids and the values are the correpsonding transcript using the teacher model (beam search).
        if task_id > 0 and args.use_kd_seq: # Create transcriptions using teacher model for KD-seq.
            train_loader = DataLoader(exp_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,  #num_workers=args.num_workers
                                      collate_fn=lambda x: data_processing(x,args.max_len_audio,tokenizer),pin_memory=True,drop_last=False,)
            
            dict_kd = {}
            
            with torch.inference_mode():
                for idx_batch, (x_,y_,text_,ids_) in enumerate(train_loader): 
                    x_ = x_.to(device)
                    
                    for temp in range(x_.shape[0]):
                        
                        if y_[temp] not in list(memory.seen_classes): continue
                        
                        wav_id = text_transform.int_to_text(ids_[temp,:].tolist())
                        pred_token_ = beam_search(teacher_model, x_[temp,:].unsqueeze(0), device, args.n_vocab, beam_size=20)
                        dict_kd[wav_id] = pred_token_
                
                
                assert len(dict_kd) == len(memory), ("Dictionary len doesn't match memory len!")
                        
                        
                        
                        
                        
                
            
        
        train_loader = DataLoader(exp_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,  #num_workers=args.num_workers
                                      collate_fn=lambda x: data_processing(x,args.max_len_audio,tokenizer),pin_memory=True,drop_last=False,) #collate_fn=lambda x: data_processing(x,args.max_len_audio)
        valid_loader = DataLoader(valid_taskset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                  collate_fn = lambda x: data_processing(x,args.max_len_audio,tokenizer),pin_memory=True, drop_last=False,)
        test_loader = DataLoader(test_taskset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 collate_fn=lambda x: data_processing(x,args.max_len_audio,tokenizer),pin_memory=True, drop_last=False,)
        
        
        #if args.memory_size > 0 and args.use_kd_audio:
        #    seen_classes = list(memory.seen_classes)
        
        
        #count = 0
        #for (x,_,_) in train_loader:
        #    count += x.shape[0]
        #print("Count: ",count)
            
        
        #num_steps = len(train_loader)*epochs
        
        warmup_period = len(train_loader)*args.warmup_ratio
        
        #warmup_period = int(num_steps*(args.warmup_ratio/100))
        
        print("Warm up: ", warmup_period)
        #lr_scheduler = LambdaLR(
        #    optimizer=optimizer, lr_lambda=lambda step: rate(step, model_size=args.n_hidden_audio,factor=1.0,warmup=warmup_period))
        
        
        
        optim = NoamOpt(args.n_hidden_audio, warmup_period, optimizer)
        
    ###########################################################################
    #                                                                         #
    # Begin of the train and test loops                                       #
    #                                                                         #
    ###########################################################################
        print(f"Start training for {epochs[task_id]} epochs")
        
        n_valid = int(len(valid_loader)*0.1)
    
        best_IC_ACC = 0.0
        best_CER = 2.0
        best_WER = 2.0
        best_model = None
        best_epoch = -1
        #path = os.getcwd() + '/models_SLURP_wav_SpeechBrain_intentsaug/'   #'/models_SLURP_ASR/'
        
        
        
        
        for epoch in range(epochs[task_id]):
            
            
            model.train()
            train_loss = 0.
            KD_audio_loss = 0.
            KD_token_loss = 0.
            KD_seq_loss = 0.
            KD_cross_loss = 0.
            #text_loss_ = 0.
            #ctc_loss_ = 0.
            
            
            
            print(f"Epoch #: {epoch}")
           
            #for x, y, t, token, token_len in train_loader:
            #for idx_batch, (x, y, t, text) in enumerate(train_loader):
            
            
            
            for idx_batch, (x, y,text,ids) in enumerate(train_loader):   
                
                #optimizer.zero_grad()
                optim.optimizer.zero_grad()
                #x = torch.squeeze(x,1).transpose(1,2)
                #x = x.transpose(1,2)
               
                
                text = text.squeeze(1)

                #y = y.squeeze(1)
 
                # Find samples from the current batch that correspond to the past classes (i.e., buffer memory samples).
                
                loss = 0.
                mse_loss = None
                kd_seq_loss = None
                
                
                
                if task_id >0 and args.use_kd_audio:
                    indexes_batch = []
                    for seen_class in memory.seen_classes:
                        indexes_class = np.where(y.numpy()==seen_class)[0]
                        indexes_batch.append(indexes_class)
                    indexes_batch = np.concatenate(indexes_batch)
                    
                    if len(indexes_batch) != 0: 
                        x_memory = x[indexes_batch].to(device)
                        current_features = model.embed_audio_iCaRL(x_memory)
                        past_features = teacher_model.embed_audio_iCaRL(x_memory)
                        alpha_mse = math.sqrt(len(indexes_batch)/len(x))
                        
                        mse_loss = alpha_mse*criterion_mse(current_features,past_features)
                        loss += mse_loss
                        KD_audio_loss += mse_loss.detach().item()
                    
                    
                    
                    # For KD audio all data!!
                    
                    # current_features = model.embed_audio_iCaRL(x.to(device))
                    # past_features = teacher_model.embed_audio_iCaRL(x.to(device))
                    
                    # alpha_mse = alpha
                    # mse_loss = alpha_mse*criterion_mse(current_features,past_features)
                    # loss += mse_loss
                    # KD_audio_loss += mse_loss.detach().item()
                    
                        
                        
                
                
                if task_id >0 and args.use_kd_seq:
                    indexes_batch = []
                    kd_labels = []
                    for seen_class in memory.seen_classes:
                        indexes_class = np.where(y.numpy()==seen_class)[0]
                        indexes_batch.append(indexes_class)
                    indexes_batch = np.concatenate(indexes_batch)
                    
                    if len(indexes_batch) != 0: 
                        x_memory = x[indexes_batch].to(device)
                        ids_memory = ids[indexes_batch]
                        
                        for id_kd in range(ids_memory.shape[0]):
                            label_kd = torch.tensor(dict_kd[text_transform.int_to_text(ids_memory[id_kd,:].tolist())])
                            kd_labels.append(label_kd)
                        
                        kd_labels = torch.nn.utils.rnn.pad_sequence(kd_labels, batch_first=True, padding_value=args.PAD_token)
                        
                        kd_labels = kd_labels.to(device)
                        
                        text_to_model_kd = kd_labels[:,:-1]
                        text_to_loss_kd = kd_labels[:,1:]   
                        
                        tgt_mask_kd = model.create_tgt_mask(text_to_model_kd.shape[1]).to(device)
                        tgt_key_padding_mask_kd = model.create_pad_mask(text_to_model_kd, args.PAD_token).to(device)
                        
                        pred_text_kd, _ = model(x_memory,text_to_model_kd, tgt_mask = tgt_mask_kd, tgt_key_padding_mask = tgt_key_padding_mask_kd)
                        text_loss_kd = criterion_text(pred_text_kd.permute(0,2,1),text_to_loss_kd)
                        
                        alpha_kd_seq = math.sqrt(len(indexes_batch)/len(x))
                        kd_seq_loss = alpha_kd_seq*text_loss_kd
                        loss += kd_seq_loss
                        KD_seq_loss += kd_seq_loss.detach().item()
                        
                        
                        #del x_memory, ids_memory, kd_labels, text_to_model_kd, text_to_loss_kd, tgt_key_padding_mask_kd, tgt_mask_kd, pred_text_kd, text_loss_kd, kd_seq_loss
                        
                        
                x = x.to(device)
                #y = y.to(device)
                text = text.to(device)
                
                
                #predictions_scenario, predictions_action = model.embed_audio_intent(x)
                
                text_to_model = text[:,:-1]
                text_to_loss = text[:,1:]
                
                tgt_mask = model.create_tgt_mask(text_to_model.shape[1]).to(device)
                tgt_key_padding_mask = model.create_pad_mask(text_to_model, args.PAD_token).to(device)
                
                
                
                pred_text, enc_out = model(x,text_to_model, tgt_mask = tgt_mask, tgt_key_padding_mask = tgt_key_padding_mask)
                
                
                # yy = beam_search(model, x[0,:].unsqueeze(0), device, args.n_vocab,max_length=130, SOS_token=28, EOS_token=29, beam_size=5, pen_alpha=0.6,)
                # print("Decoded",yy)
                
               
                # if idx_batch == (len(train_loader) -2) or idx_batch == (len(train_loader)//2):
                #     print("True text (TRAIN): ", text_transf.int_to_text(text_to_loss[0,:].tolist()))
                #     print("Predicted text (TRAIN): ", text_transf.int_to_text(torch.argmax(pred_text,dim=2)[0,:].tolist()))
                #     print("True text (TRAIN): ", text_transf.int_to_text(text_to_loss[4,:].tolist()))
                #     print("Predicted text (TRAIN): ", text_transf.int_to_text(torch.argmax(pred_text,dim=2)[4,:].tolist()))
                #     print("True text (TRAIN): ", text_transf.int_to_text(text_to_loss[8,:].tolist()))
                #     print("Predicted text (TRAIN): ", text_transf.int_to_text(torch.argmax(pred_text,dim=2)[8,:].tolist()))
                
                if idx_batch == (len(train_loader) -2) or idx_batch == (len(train_loader)//2):
                    print("True text (TRAIN): ", tokenizer.decode(text_to_loss[0,:].tolist()))
                    print("Predicted text (TRAIN): ", tokenizer.decode(torch.argmax(pred_text,dim=2)[0,:].tolist()))
                    print("True text (TRAIN): ", tokenizer.decode(text_to_loss[1,:].tolist()))
                    print("Predicted text (TRAIN): ", tokenizer.decode(torch.argmax(pred_text,dim=2)[1,:].tolist()))
                    print("True text (TRAIN): ", tokenizer.decode(text_to_loss[2,:].tolist()))
                    print("Predicted text (TRAIN): ", tokenizer.decode(torch.argmax(pred_text,dim=2)[2,:].tolist()))
                    
                    
                    #for x in range(args.batch_size):
                    #    gold_pred = text_transf.int_to_text(text_to_loss[x,:].tolist())
                    #    list_gold.append(gold_pred.replace('@',''))
                    #    list_preds.append(text_transf.int_to_text(torch.argmax(pred_text,dim=2)[x,:].tolist()))
                        #list_gold.append(text_transf.int_to_text(text_to_loss[x,:].tolist()))
                    #print(f"WER at batch {idx_batch} of epoch {epoch}: {WER(list_preds,list_gold)} ")
                    #wer_train = WER(list_preds,list_gold)
    
                    
                text_loss = criterion_text(pred_text.permute(0,2,1),text_to_loss)
  
                
                #input_lengths = torch.full(size=(enc_out.shape[0],),fill_value=enc_out.shape[1], dtype=torch.long)
                #ctc_loss = criterion_ctc(enc_out.permute(1,0,2),text,input_lengths,token_len)
    
                if task_id > 0 and mse_loss:
                    loss += (1-alpha_mse)*text_loss
                elif task_id > 0 and kd_seq_loss:
                    loss += (1-alpha_kd_seq)*text_loss
                else:
                    loss += text_loss#*0.7 + ctc_loss*0.3       #*0.50 + 0.50*ctc_loss
                
            
                if task_id > 0 and args.use_kd_token:
                    indexes_batch = []
                    for seen_class in memory.seen_classes:
                        indexes_class = np.where(y.numpy()==seen_class)[0]
                        indexes_batch.append(indexes_class)
                    indexes_batch = np.concatenate(indexes_batch)
                    
                    if len(indexes_batch) != 0: 
                        x_memory = x[indexes_batch].to(device)
                        text_memory = text[indexes_batch].to(device)
                        
                        text_to_model_mem = text_memory[:,:-1]
                        
                        tgt_mask_mem = model.create_tgt_mask(text_to_model_mem.shape[1]).to(device)
                        tgt_key_padding_mask_mem = model.create_pad_mask(text_to_model_mem, args.PAD_token).to(device)
                    
                        pred_text_old, _ = teacher_model(x_memory,text_to_model_mem, tgt_mask = tgt_mask_mem, tgt_key_padding_mask = tgt_key_padding_mask_mem)
                        pred_text_new, _ = model(x_memory,text_to_model_mem, tgt_mask = tgt_mask_mem, tgt_key_padding_mask = tgt_key_padding_mask_mem)
                        
                        alpha_token = math.sqrt(len(indexes_batch)/len(x))
                        loss, kd_token_loss = get_kdloss(pred_text_new,pred_text_old,loss,args.distillation_tau,alpha_token)
                        KD_token_loss += kd_token_loss.item()
                
                
                if task_id > 0 and args.use_kd_token_all:
                    with torch.no_grad():
                        pred_text_old, _ = teacher_model(x,text_to_model, tgt_mask = tgt_mask, tgt_key_padding_mask = tgt_key_padding_mask)
                    loss, kd_token_loss = get_kdloss(pred_text,pred_text_old,loss,args.distillation_tau,alpha)
                    KD_token_loss += kd_token_loss.item()        
                
                
                
                if task_id > 0 and args.use_kd_cross:
                    indexes_batch = []
                    for seen_class in memory.seen_classes:
                        indexes_class = np.where(y.numpy()==seen_class)[0]
                        indexes_batch.append(indexes_class)
                    indexes_batch = np.concatenate(indexes_batch)
                    if len(indexes_batch) != 0: 
                        x_memory = x[indexes_batch].to(device)
                        text_memory = text[indexes_batch].to(device)
                        text_to_model_mem = text_memory[:,:-1]
                        
                        tgt_mask_mem = model.create_tgt_mask(text_to_model_mem.shape[1]).to(device)
                        tgt_key_padding_mask_mem = model.create_pad_mask(text_to_model_mem, args.PAD_token).to(device)
                        
                        with torch.no_grad():
                            enc_old = teacher_model.embed_audio(x_memory)
                            enc_new = model.embed_audio(x_memory)
                            pred_text_old = teacher_model.decod_audio(enc_old, text_to_model_mem, tgt_mask = tgt_mask_mem, tgt_key_padding_mask = tgt_key_padding_mask_mem)
                        
                        pred_text_new = model.decod_audio(enc_new, text_to_model_mem, tgt_mask = tgt_mask_mem, tgt_key_padding_mask = tgt_key_padding_mask_mem)
                        
                        alpha_token = math.sqrt(len(indexes_batch)/len(x))
                        loss, kd_cross_loss = get_kdloss(pred_text_new,pred_text_old,loss,args.distillation_tau,alpha_token)
                        KD_cross_loss += kd_cross_loss.item()
                            
                        
                            
                        
            
            
                train_loss += loss.detach().item()
                #text_loss_ += text_loss.item()
                #ctc_loss_ += ctc_loss.detach().item()
                
                loss.backward()
                optimizer.step()
            
                
                
                
            
            # Test phase
            if args.eval_every and (epoch+1) % args.eval_every == 0:
                model.eval()
                valid_loss = 0.
                #test_text_loss_ = 0.
                train_loss /= len(train_loader)
                KD_audio_loss /= len(train_loader)
                KD_token_loss /= len(train_loader)
                KD_seq_loss /= len(train_loader)
                KD_cross_loss /= len(train_loader)
                #text_loss_ /= len(train_loader)
                #ctc_loss_ /= len(train_loader)
                print(f"Trainloss at epoch {epoch}: {train_loss}")
                #list_preds_val = []
                #list_gold_val = []
                
                list_preds_val = []
                list_gold_val = []
                correct = 0
                total = 0  
                
                with torch.inference_mode():
                    #for x_valid, y_valid, t_valid, trans_valid, boh in test_loader:
                    #for idx_valid_batch, (x_valid, y_valid, t_valid, text_valid) in enumerate(valid_loader):
                    for idx_valid_batch, (x_valid,_, text_valid,_) in enumerate(valid_loader):
                        
                        x_valid = x_valid.to(device)
                        #y_valid = y_valid.to(device)
                        text_valid = text_valid.to(device)
                        text_valid = text_valid.squeeze(1)
                        
                        #x_valid = x_valid.transpose(1,2)
                        #predic_valid_sce, predic_valid_act = model.embed_audio_intent(x_valid.cuda())
                        #predic_valid_sce = model(x_valid.cuda())
                        
                        text_to_model_val = text_valid[:,:-1]
                        text_to_loss_val = text_valid[:,1:]
                        
                        tgt_mask = model.create_tgt_mask(text_to_model_val.shape[1]).to(device)
                        tgt_key_padding_mask = model.create_pad_mask(text_to_model_val, args.PAD_token).to(device)
                        
                        pred_text_val, _ = model(x_valid.cuda(),text_to_model_val, tgt_mask = tgt_mask, tgt_key_padding_mask = tgt_key_padding_mask)
                        
                        if idx_valid_batch < n_valid:
                            for x in range(x_valid.shape[0]):
                                pred_token = beam_search(model, x_valid[x,:].unsqueeze(0), device, args.n_vocab, beam_size=10)
                                list_preds_val.append(tokenizer.decode(pred_token))
                                gold_pred = tokenizer.decode(text_to_loss_val[x,:].tolist())
                                list_gold_val.append(gold_pred)
                                
                                pred_intent =list(map(str.strip,tokenizer.decode(pred_token).split('_SEP')))[0]
                                gold_intent = list(map(str.strip,gold_pred.split('_SEP')))[0]
                                if pred_intent == gold_intent:
                                    correct += 1
                                total +=1
                                
                            
                        
                        
                        if idx_valid_batch == (len(valid_loader) -2):

                            # print("True text (VALID): ", text_transf.int_to_text(text_to_loss_val[0,:].tolist()))
                            # print("Predicted text (VALID): ", text_transf.int_to_text(torch.argmax(pred_text_val,dim=2)[0,:].tolist()))
                            # print("True text (VALID): ", text_transf.int_to_text(text_to_loss_val[1,:].tolist()))
                            # print("Predicted text (VALID): ", text_transf.int_to_text(torch.argmax(pred_text_val,dim=2)[1,:].tolist()))
                            # print("True text (VALID): ", text_transf.int_to_text(text_to_loss_val[2,:].tolist()))
                            # print("Predicted text (VALID): ", text_transf.int_to_text(torch.argmax(pred_text_val,dim=2)[2,:].tolist()))
                            
                            print("True text (VALID): ", tokenizer.decode(text_to_loss_val[0,:].tolist()))
                            print("Predicted text (VALID): ", tokenizer.decode(torch.argmax(pred_text_val,dim=2)[0,:].tolist()))
                            print("True text (VALID): ", tokenizer.decode(text_to_loss_val[1,:].tolist()))
                            print("Predicted text (VALID): ", tokenizer.decode(torch.argmax(pred_text_val,dim=2)[1,:].tolist()))
                            print("True text (VALID): ", tokenizer.decode(text_to_loss_val[2,:].tolist()))
                            print("Predicted text (VALID): ", tokenizer.decode(torch.argmax(pred_text_val,dim=2)[2,:].tolist()))
                            
                            
                        val_text_loss = criterion_text(pred_text_val.permute(0,2,1),text_to_loss_val)
                        #y_valid = y_valid.squeeze(1)
                        valid_loss +=  val_text_loss
                        
                        
                        
                        
                        
                    #print(f"VALID WER at epoch {epoch}: {WER(list_preds_val,list_gold_val)}")
                    #wer_valid = WER(list_preds_val,list_gold_val)
                    
                    
                    
                    valid_loss /= len(valid_loader)
                    val_wer = WER(list_preds_val,list_gold_val)
                    val_cer = CER(list_preds_val,list_gold_val)
                    intent_accuracy_val = correct/total
                    print(f"VAL WER at epoch {epoch}: {val_wer}")
                    print(f"VAL CER at epoch {epoch}: {val_cer}")
                        
                    print(f"Total corrected VAL intent predictions at epoch {epoch}: {correct} out of {total}")
                    print("VAL Intent Accuracy at epoch {epoch}: ", intent_accuracy_val)
                
                if task_id > 0 and epoch < 4 :
                    pass
                else:

                    if intent_accuracy_val > best_IC_ACC:
                            best_model = model.state_dict()
                            best_WER = val_wer
                            best_CER = val_cer
                            best_epoch = epoch
                            best_IC_ACC = intent_accuracy_val
                    elif intent_accuracy_val == best_IC_ACC:
                        if val_cer < best_CER and val_wer < best_WER:
                            best_model = model.state_dict()
                            best_WER = val_wer
                            best_CER = val_cer
                            best_epoch = epoch
                            best_IC_ACC = intent_accuracy_val
            
                with torch.inference_mode():  
                    
                    print(f"Valid loss at epoch {epoch} and task {task_id}: {valid_loss}")
                    
                    #for idx_test_batch, (x_test, _, _, text_test) in enumerate(test_loader):
                    for idx_test_batch, (x_test,_, text_test,_) in enumerate(test_loader):
                        #print("Batch TEST #: ",idx_test_batch)
                        x_test = x_test.to(device)
                        #x_test = x_test.transpose(1,2)
                        text_test = text_test.to(device).squeeze(1)
                        text_to_loss_test = text_test[:,1:]
                        
                        
                        # for x in range(x_test.shape[0]):
                        #     pred_token = beam_search(model, x_test[x,:].unsqueeze(0), device, args.n_vocab, beam_size=20)
                            
                        #     #pred_token = rescoring(final_tokens, final_scores, model, x_test[x,:].unsqueeze(0), criterion_ctc, device, ctc_weight = 0.3)
                                
                            
                        
                        #     list_preds_test.append(tokenizer.decode(pred_token))
                        #     gold_pred = tokenizer.decode(text_to_loss_test[x,:].tolist())
                        #     list_gold_test.append(gold_pred)
                            
                        #     pred_intent =list(map(str.strip,tokenizer.decode(pred_token).split('_SEP')))[0]
                        #     #lev_dist = [distance(pred_intent,xx) for xx in reference]
                            
                        #     gold_intent = list(map(str.strip,gold_pred.split('_SEP')))[0]
                        #     #pred_index = np.argmin(lev_dist)
                           
                            
                        #     if pred_intent == gold_intent:
                        #     #if reference[pred_index] == gold_intent:
                        #         correct += 1
                        #     total +=1
                        
                        
                        # if idx_test_batch == 2:
                            
                        #     test_wer = WER(list_preds_test,list_gold_test)
                        #     test_cer = CER(list_preds_test,list_gold_test)
                        #     intent_accuracy = correct/total
                        #     print(f"TEST WER at epoch {epoch}: {test_wer}")
                        #     print(f"TEST CER at epoch {epoch}: {test_cer}")
                                
                        #     print(f"Total corrected intent predictions at epoch {epoch}: {correct} out of {total}")
                        #     print("Intent Accuracy at epoch {epoch}: ", intent_accuracy)
                        #     break
                        
                        # for x in range(x_test.shape[0]):
                        #     pred_token = greedy_decode(model, x_test[x,:].unsqueeze(0), device)
                        #     list_preds_test.append(text_transf.int_to_text(pred_token))
                        #     gold_pred = text_transf.int_to_text(text_to_loss_test[x,:].tolist())
                        #     list_gold_test.append(gold_pred.replace('@',''))
                            #list_gold_test.append(text_transf.int_to_text(text_to_loss_test[x,:].tolist()))
                        if idx_test_batch == (len(test_loader) -2):
                             
                            
                            pred_token = beam_search(model, x_test[0,:].unsqueeze(0), device, args.n_vocab)
                            pred_token = tokenizer.decode(pred_token)
                            #pred_token_20 = beam_search(model, x_test[0,:].unsqueeze(0), device, args.n_vocab, beam_size=20)
                            #pred_token_20 = tokenizer.decode(pred_token_20)
                            
                            #pred_token_list_5 = beam_search_nbest(model, x_test[0,:].unsqueeze(0), device, args.n_vocab)
                            #pred_token_list_5 = rescoring()
                            
                            #pred_token_list_20 = beam_search_nbest(model, x_test[0,:].unsqueeze(0), device, args.n_vocab, beam_size=20)
                            #pred_token_list_20 = rescoring()
                            
                            
                            
                            #pred_token = greedy_decode(model, x_test[0,:].unsqueeze(0), device)
                            gold_pred = tokenizer.decode(text_to_loss_test[0,:].tolist())
                            #gold_pred = gold_pred.replace('@','')[:-1]
                            print("Predicted text (TEST): ", pred_token)
                            #print("Predicted text (TEST) 20 beams: ", pred_token_20)
                            print("True text (TEST): ", gold_pred)
                            
                            #pred_token = greedy_decode(model, x_test[1,:].unsqueeze(0), device)
                            pred_token = beam_search(model, x_test[1,:].unsqueeze(0), device, args.n_vocab)
                            pred_token = tokenizer.decode(pred_token)
                            #pred_token_20 = beam_search(model, x_test[1,:].unsqueeze(0), device, args.n_vocab, beam_size=20)
                            #pred_token_20 = tokenizer.decode(pred_token)
                            gold_pred = tokenizer.decode(text_to_loss_test[1,:].tolist())
                            #gold_pred = gold_pred.replace('@','')[:-1]
                            print("Predicted text (TEST): ", pred_token)
                            #print("Predicted text (TEST) 20 beams: ", pred_token_20)
                            print("True text (TEST): ", gold_pred)
                            #pred_token = greedy_decode(model, x_test[2,:].unsqueeze(0), device)
                            pred_token = beam_search(model, x_test[2,:].unsqueeze(0), device, args.n_vocab)
                            pred_token = tokenizer.decode(pred_token)
                            #pred_token_20 = beam_search(model, x_test[2,:].unsqueeze(0), device, args.n_vocab, beam_size=20)
                            #pred_token_20 = tokenizer.decode(pred_token_20)
                            gold_pred = tokenizer.decode(text_to_loss_test[2,:].tolist())
                            #gold_pred = gold_pred.replace('@','')[:-1]
                            print("Predicted text (TEST): ", pred_token)
                            #print("Predicted text (TEST) 20 beams: ", pred_token_20)
                            print("True text (TEST): ", gold_pred)
                        
                            
                    # if epoch == 10:
                    #     for idx_test_batch, (x_test,text_test) in enumerate(test_loader):
                    #         x_test = x_test.to(device)
                    #         x_test = x_test.transpose(1,2)
                    #         text_test = text_test.to(device)
                    #         text_to_loss_test = text_test[:,1:]
                            
                    #         for x in range(x_test.shape[0]):
                    #             # Greedy decoding doesn't remove the SOS token --> pred_token[1:]
                    #             # Beam search does remove SOS.
                                
                    #             pred_token = greedy_decode(model, x_test[x,:].unsqueeze(0), device)
                    #             list_preds_test.append(text_transf.int_to_text(pred_token))
                    #             gold_pred = text_transf.int_to_text(text_to_loss_test[x,:].tolist())
                    #             list_gold_test.append(gold_pred.replace('@',''))
                    #             #list_gold_test.append(text_transf.int_to_text(text_to_loss_test[x,:].tolist()))
                        
                    #     print(f"TEST WER at epoch {epoch}: {WER(list_preds_test,list_gold_test)}")
                    #     print(f"TEST CER at epoch {epoch}: {CER(list_preds_test,list_gold_test)}")
                    
                    #print(f"TEST WER at epoch {epoch}: {WER(list_preds_test,list_gold_test)}")
                    #wer_test = WER(list_preds_test,list_gold_test)
                            
  
                    if args.use_wandb:
                        wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,"val_wer": val_wer, "val_cer": val_cer, "int_acc":intent_accuracy_val,
                                   "KD_token_loss":KD_token_loss#"KD_audio_loss":KD_audio_loss#"KD_seq_loss": KD_seq_loss#"KD_token_loss":KD_token_loss#,#"KD_cross_loss": KD_cross_loss 
                                   #"ASR_loss": text_loss_, "CTC_loss": ctc_loss_,
                                   #"WER_train": wer_train, "WER_valid": wer_valid, 
                                   #"WER_test": wer_test
                                   }
                                  )
        
        list_preds_test = []
        list_gold_test = []
        correct = 0
        total = 0 
        
        del model
        model = Seq2SeqTransformer(dims, device=device).to(device)
        model.load_state_dict(best_model)
        preds_jsonl = []
        
        with torch.inference_mode():
            model.eval()
            
            for idx_test_batch, (x_test,_, text_test,ids) in enumerate(test_loader):
                #print("Batch TEST #: ",idx_test_batch)
                x_test = x_test.to(device)
                #x_test = x_test.transpose(1,2)
                text_test = text_test.to(device).squeeze(1)
                text_to_loss_test = text_test[:,1:]
                
                
                for x in range(x_test.shape[0]):
                    current_dict = {}
                    wav_id = text_transform.int_to_text(ids[x,:].tolist())
                    current_dict["file"] = wav_id
                    
                    pred_token = beam_search(model, x_test[x,:].unsqueeze(0), device, args.n_vocab, beam_size=20)
                    
                    #pred_token = rescoring(final_tokens, final_scores, model, x_test[x,:].unsqueeze(0), criterion_ctc, device, ctc_weight = 0.3)
                        
                    
                
                    list_preds_test.append(tokenizer.decode(pred_token))
                    gold_pred = tokenizer.decode(text_to_loss_test[x,:].tolist())
                    list_gold_test.append(gold_pred)
                    
                    transc_tok = list(map(str.strip,tokenizer.decode(pred_token).split('_SEP')))
                    pred_intent = transc_tok[0]
                    
                    #pred_intent =list(map(str.strip,tokenizer.decode(pred_token).split('_SEP')))[0]
                    
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
 
                    #lev_dist = [distance(pred_intent,xx) for xx in reference]
                    
                    gold_intent = list(map(str.strip,gold_pred.split('_SEP')))[0]
                    #pred_index = np.argmin(lev_dist)
                   
                    
                    if pred_intent == gold_intent:
                    #if reference[pred_index] == gold_intent:
                        correct += 1
                    total +=1
                
        
        file_name = f"pred_outputs_rehe1%icarl_6tasks_KDtokrehe_orderdec_seed11_task{task_id}" #KDaudio
        with jsonlines.open(file_name,'w') as writer:
            writer.write_all(preds_jsonl)
            
        print(f"For the next task, I'll go forward with the best model from the previous task from epoch {best_epoch}.")
        
        test_wer = WER(list_preds_test,list_gold_test)
        test_cer = CER(list_preds_test,list_gold_test)
        intent_accuracy_test = correct/total
        
        IC_ACC_overall.append(intent_accuracy_test)
        WER_overall.append(float(test_wer))
        CER_overall.append(float(test_cer))
        print(f"TEST WER at TASK {task_id}: {test_wer}")
        print(f"TEST CER at TASK {task_id}: {test_cer}")
            
        print(f"TEST Total corrected intent predictions at TASK {task_id}: {correct} out of {total}")
        print(f"TEST Intent Accuracy at TASK {task_id}: ", intent_accuracy_test)
        
        
        
        if memory is not None:
            
            if args.herding == 'random':
            
                memory.add(*scenario_train[task_id].get_raw_samples(),z=None) 
            else: 
                loader = DataLoader(scenario_train[task_id], batch_size=args.batch_size,shuffle=False,num_workers=2,collate_fn=lambda x: data_processing(x,args.max_len_audio,tokenizer),
                                    pin_memory=True, drop_last=False)
                 
                features= []
                
                with torch.no_grad():
                    for x, _, _,_ in loader:
                        x = x.to(device)
                        feats = model.embed_audio_iCaRL(x)   # Take the encoder's outputs and squeeze the temporal dimension --> size: (batch_size, hidden_size).
                        feats = feats.cpu().numpy()
                       
                        features.append(feats)
                        
                
                features = np.concatenate(features)
                
                
                memory.add(*scenario_train[task_id].get_raw_samples(),z=features) 
            
                
            print("Seen classes by the memory: ",memory.seen_classes)
            print(len(memory))
            assert len(memory) <= args.memory_size  
        
                
            
            
    print("IC_ACC_overall: ", IC_ACC_overall)
    print("Avg IC_ACC: ", mean(IC_ACC_overall))
    print("Last IC_ACC: ", IC_ACC_overall[-1])
    
    print("WER_overall: ", WER_overall)
    print("Avg WER: ", mean(WER_overall))
    print("Last WER: ", WER_overall[-1])
    
    print("CER_overall: ", CER_overall)
    print("Avg CER: ", mean(CER_overall))
    print("Last CER: ", CER_overall[-1])
    
    
    
    
    
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
   
    # data_path ='/Users/umbertocappellazzo/Desktop/PHD'
    # a = FluentSpeech(data_path,train=True,download=False)
    # class_order = [19, 27, 30, 28, 15,  4,  2,  9, 10, 22, 11,  7,  1, 25, 16, 14,  5,
    #          8, 29, 12, 21, 17,  3, 20, 23,  6, 18, 24, 26,  0, 13]
    # scenario_train = ClassIncremental(a,increment=3,initial_increment=4,
    #                                   transformations=[partial(trunc, max_len=16000)],class_order=class_order)


