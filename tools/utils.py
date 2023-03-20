#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 16:14:13 2022

@author: umbertocappellazzo
"""
from torch.nn import functional as F
from torchaudio import transforms as t
import numpy as np
import string
from typing import Optional, List

def trunc(x, max_len):
    l = len(x)
    if l > max_len:
        x = x[l//2-max_len//2:l//2+max_len//2]
    if l < max_len:
        x = F.pad(x, (0, max_len-l), value=0.)
    
    eps = np.finfo(np.float64).eps
    sample_rate = 16000
    n_mels = 40
    win_len = 25
    hop_len= 10
    win_len = int(sample_rate/1000*win_len)
    hop_len = int(sample_rate/1000*hop_len)
    mel_spectr = t.MelSpectrogram(sample_rate=16000,
            win_length=win_len, hop_length=hop_len, n_mels=n_mels)
    
    return np.log(mel_spectr(x)+eps)  
    

def freeze_parameters(m, requires_grad=False):
    for p in m.parameters():
        p.requires_grad = requires_grad


def get_kdloss(predictions,predictions_old,current_loss,tau,is_both_kds=False):
    logits_for_distil = predictions[:, :predictions_old.shape[1]]
    alpha = np.log((predictions_old.shape[1] / predictions.shape[1])+1)
    
    _kd_loss = F.kl_div(
        F.log_softmax(logits_for_distil / tau, dim=1),
        F.log_softmax(predictions_old / tau, dim=1),
        reduction='mean',
        log_target=True) * (tau ** 2)
    
    if is_both_kds: return current_loss + alpha*_kd_loss, _kd_loss
    else: return (1-alpha)*current_loss + alpha*_kd_loss
    
    
def get_kdloss_onlyrehe(predictions,predictions_old,current_loss,tau,alpha,is_both_kds=False):
    logits_for_distil = predictions[:, :predictions_old.shape[1]]
    
    _kd_loss = F.kl_div(
        F.log_softmax(logits_for_distil / tau, dim=1),
        F.log_softmax(predictions_old / tau, dim=1),
        reduction='mean',
        log_target=True) * (tau ** 2)
    if is_both_kds: return current_loss + alpha*_kd_loss
    else: return (1-alpha)*current_loss + alpha*_kd_loss


class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        self.punctuations = list(string.punctuation)
        del self.punctuations[6]
        self.punctuations.append('’')
        
        self.char_map_str = ["<eps>","\'",">","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
        
        self.char_map = {}
        self.index_map = {}
        for index,ch in enumerate(self.char_map_str):
        
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        text = [char for char in text if char not in self.punctuations]
        
        for c in text:
            if c == ' ':
                ch = self.char_map['>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string_ = []
        for i in labels:
            string_.append(self.index_map[i])
        return ''.join(string_).replace('>', ' ')
    

def _slice(
    y: np.ndarray,
    t: Optional[np.ndarray],
    keep_classes: Optional[List[int]] = None,
    discard_classes: Optional[List[int]] = None,
    keep_tasks: Optional[List[int]] = None,
    discard_tasks: Optional[List[int]] = None
):
    """Slice dataset to keep/discard some classes/task-ids.
    Note that keep_* and and discard_* are mutually exclusive.
    Note also that if a selection (keep or discard) is being made on the classes
    and on the task ids, the resulting intersection will be taken.
    :param y: An array of class ids.
    :param t: An array of task ids.
    :param keep_classes: Only keep samples with these classes.
    :param discard_classes: Discard samples with these classes.
    :param keep_tasks: Only keep samples with these task ids.
    :param discard_tasks: Discard samples with these task ids.
    :return: A new Continuum dataset ready to be given to a scenario.
    """
    if keep_classes is not None and discard_classes is not None:
        raise ValueError("Only use `keep_classes` or `discard_classes`, not both.")
    if keep_tasks is not None and discard_tasks is not None:
        raise ValueError("Only use `keep_tasks` or `discard_tasks`, not both.")

    if t is None and (keep_tasks is not None or discard_tasks is not None):
        raise Exception(
            "No task ids information is present by default with this dataset, "
            "thus you cannot slice some task ids."
        )
    y = y.astype(np.int64)
    if t is not None:
        t = t.astype(np.int64)

    indexes = set()
    if keep_classes is not None:
        indexes = set(np.where(np.isin(y, keep_classes))[0])
    elif discard_classes is not None:
        keep_classes = list(set(y) - set(discard_classes))
        indexes = set(np.where(np.isin(y, keep_classes))[0])

    if keep_tasks is not None:
        _indexes = np.where(np.isin(t, keep_tasks))[0]
        if len(indexes) > 0:
            indexes = indexes.intersection(_indexes)
        else:
            indexes = indexes.union(_indexes)
    elif discard_tasks is not None:
        keep_tasks = list(set(t) - set(discard_tasks))
        _indexes = np.where(np.isin(t, keep_tasks))[0]
        if len(indexes) > 0:
            indexes = indexes.intersection(_indexes)
        else:
            indexes = indexes.union(_indexes)

    indexes = np.array(list(indexes), dtype=np.int64)
    return indexes

if __name__ == "__main__":
    obj = TextTransform()
    b = "ciao! Come stai? Io' bene"
    a = obj.text_to_int(b.lower())
    print(obj.int_to_text(a))
