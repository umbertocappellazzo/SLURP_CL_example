#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:53:12 2023

@author: umbertocappellazzo
"""

import os
from Speech_CLscenario.base_dataset import _ContinuumDataset
#from base_dataset import _ContinuumDataset

#from class_incremental import ClassIncremental
import numpy as np
from typing import Union
import string
try:
    import soundfile
except:
    soundfile = None

class Slurp(_ContinuumDataset):
    def __init__(self, data_path, max_len_text, max_len_audio, train: Union[bool, str] = True, download: bool = True):
        if not isinstance(train, bool) and train not in ("train", "valid", "test","train_real","train_synthetic"):
            raise ValueError(f"`train` arg ({train}) must be a bool or train/valid/test/train_synthetic/train_real.")
        if isinstance(train, bool):
            if train:
                train = "train"
            else:
                train = "test"

        data_path = os.path.expanduser(data_path)
        self.max_len_text = max_len_text
        self.max_len_audio = max_len_audio
        super().__init__(data_path, train, download)
        
    
    def get_data(self):
        path_to_wavs = "/cappellazzo/slurp"
        x, y, transcriptions = [], [], []  # For now ENTITIES are not taken into account.
        #digits = ['0','1','2','3','4','5','6','7','8','9'] 
        
        
        punctuations = list(string.punctuation)
        punctuations.append('’')
        
        with open(os.path.join(self.data_path, f"{self.train}_intents.csv")) as f:
            lines = f.readlines()[1:]
        for line in lines:
            items = line[:-1].split(';')
            
            transcription = items[3].lower()
            # Remove first 2 characters 's1' or 's2' containing numbers. 
            #if transcription[:2] == 's1' or transcription[:2] == 's2':
            #    transcription = transcription[2:]
            
            # 47 transcriptions still have some numbers --> remove them.
            
            if 'synth' in items[2]:
                    #x.append(os.path.join(self.data_path,'slurp_synth', items[2]))
                #pathh = os.path.join(self.data_path,'slurp_synth', items[2])
                pathh = os.path.join(path_to_wavs,'slurp_synth', items[2])
                #x.append(os.path.join(path_to_wavs,'slurp_synth', items[2]))
            else:
                    #x.append(os.path.join(self.data_path,'slurp_real', items[2]))
                #pathh = os.path.join(self.data_path,'slurp_real', items[2])
                #x.append(os.path.join(path_to_wavs,'slurp_real', items[2]))
                pathh = os.path.join(path_to_wavs,'slurp_real', items[2])
            
            wav = soundfile.read(pathh)[0]
            
            #skip = False
            #if any((c in digits) for c in transcription):
            #    skip = True
            
            if len(transcription) > self.max_len_text or len(wav) > self.max_len_audio:
            
                pass
            
            else:
                
                transc = [char for char in transcription if char not in punctuations]
                transcription = ''
                for charr in transc:
                    transcription += charr
                
                
                scenario, action = items[4:6]
                intent = scenario + '_' + action
              
                transcriptions.append(transcription)
  
    
                
                if 'synth' in items[2]:
                    #x.append(os.path.join(self.data_path,'slurp_synth', items[2]))
                    x.append(os.path.join(path_to_wavs,'slurp_synth', items[2]))
                else:
                    #x.append(os.path.join(self.data_path,'slurp_real', items[2]))
                    x.append(os.path.join(path_to_wavs,'slurp_real', items[2]))
                #transcription = items[3].lower()
                
                #transcriptions.append(items[3].lower())
                y.append([self.scenarios[scenario],
                        self.intents[intent]
                    ])
        
        return np.array(x), np.array(y), None, transcriptions
        
        
        
    @property
    def transformations(self):
        return None    
    
    @property 
    def scenarios(self):
        return {
            'alarm': 0,
            'audio': 1,
            'calendar': 2,
            'cooking': 3,
            'datetime': 4,
            'email': 5,
            'general': 6,
            'iot': 7,
            'lists': 8,
            'music': 9,
            'news': 10,
            'play': 11,
            'qa': 12,
            'recommendation': 13,
            'social': 14,
            'takeaway': 15,
            'transport': 16,
            'weather': 17,
            }
    
    
    @property 
    def actions(self):
        return {
            'addcontact': 0,
            'affirm': 1, 
            'audiobook': 2,
            'cleaning': 3,
            'coffee': 4, 
            'commandstop': 5, 
            'confirm': 6, 
            'convert': 7, 
            'createoradd': 8, 
            'currency': 9, 
            'definition': 10, 
            'dislikeness': 11, 
            'dontcare': 12, 
            'events': 13, 
            'explain': 14, 
            'factoid': 15, 
            'game': 16, 
            'greet': 17, 
            'hue_lightchange': 18, 
            'hue_lightdim': 19, 
            'hue_lightoff': 20, 
            'hue_lighton': 21, 
            'hue_lightup': 22, 
            'joke': 23, 
            'likeness': 24, 
            'locations': 25, 
            'maths': 26, 
            'movies': 27, 
            'music': 28, 
            'negate': 29, 
            'order': 30, 
            'podcasts': 31, 
            'post': 32, 
            'praise': 33, 
            'query': 34, 
            'querycontact': 35, 
            'quirky': 36, 
            'radio': 37, 
            'recipe': 38, 
            'remove': 39, 
            'repeat': 40, 
            'sendemail': 41, 
            'set': 42, 
            'settings': 43, 
            'stock': 44, 
            'taxi': 45, 
            'ticket': 46, 
            'traffic': 47, 
            'volume_down': 48, 
            'volume_mute': 49, 
            'volume_other': 50, 
            'volume_up': 51, 
            'wemo_off': 52, 
            'wemo_on': 53,
            }

    @property
    def intents(self):
        return {
        'alarm_query': 0,
        'alarm_remove': 1,
        'alarm_set': 2,
        'audio_volume_down': 3,
        'audio_volume_mute': 4,
        'audio_volume_other': 5,
        'audio_volume_up': 6,
        'calendar_query': 7,
        'calendar_remove': 8,
        'calendar_set': 9,
        'cooking_query': 10,
        'cooking_recipe': 11,
        'datetime_convert': 12,
        'datetime_query': 13,
        'email_addcontact': 14,
        'email_query': 15,
        'email_querycontact': 16,
        'email_sendemail': 17,
        'general_affirm': 18,
        'general_commandstop': 19,
        'general_confirm': 20,
        'general_dontcare': 21,
        'general_explain': 22,
        'general_greet': 23,
        'general_joke': 24,
        'general_negate': 25,
        'general_praise': 26, 
        'general_quirky': 27,
        'general_repeat': 28,
        'iot_cleaning': 29,
        'iot_coffee': 30,
        'iot_hue_lightchange': 31,
        'iot_hue_lightdim': 32,
        'iot_hue_lightoff': 33,
        'iot_hue_lighton': 34,
        'iot_hue_lightup': 35,
        'iot_wemo_off': 36,
        'iot_wemo_on': 37, 
        'lists_createoradd': 38,
        'lists_query': 39,
        'lists_remove': 40,
        'music_dislikeness': 41,
        'music_likeness': 42,
        'music_query': 43,
        'music_settings': 44,
        'news_query': 45,
        'play_audiobook': 46,
        'play_game': 47,
        'play_music': 48,
        'play_podcasts': 49,
        'play_radio': 50,
        'qa_currency': 51,
        'qa_definition': 52,
        'qa_factoid': 53,
        'qa_maths': 54,
        'qa_query': 55,
        'qa_stock': 56,
        'recommendation_events': 57,
        'recommendation_locations': 58,
        'recommendation_movies': 59,
        'social_post': 60,
        'social_query': 61,
        'takeaway_order': 62,
        'takeaway_query': 63,
        'transport_query': 64, 
        'transport_taxi': 65,
        'transport_ticket': 66,
        'transport_traffic': 67,
        'weather_query': 68
        }
    
