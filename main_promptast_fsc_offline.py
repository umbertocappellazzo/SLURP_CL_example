import copy
from torch.utils.data import DataLoader
from functools import partial
from torch.optim import Adam, AdamW
from Speech_CLscenario.fluentspeech import FluentSpeech
from Speech_CLscenario.slurp_aug import Slurp
from Speech_CLscenario.class_incremental import ClassIncremental
# from continuum import ClassIncremental
# from continuum.datasets import FluentSpeech
import torch
import torch.nn.functional as F
import argparse
from continuum.metrics import Logger
import numpy as np
#from model_slurp import CL_model
#from model_transformer import CL_model, ModelDimensions
#from model_trans_wav2vec import Seq2SeqTransformer, ModelDimensions
#from model_confenc_wav import Seq2SeqTransformer, ModelDimensions
from model_prompt import Seq2SeqTransformer, ModelDimensions
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
from torch.optim.lr_scheduler import LambdaLR
from tokenizers import Tokenizer
from tqdm import tqdm

# HUGGINGFACE IMPORTS
from transformers import AutoProcessor, ASTModel, AutoFeatureExtractor

# PROMPT CLASS & PROMPTED MODEL IMPORTS
from PromptASTClassifier import PromptASTClassifier
from prompt import Prompt, PromptArgs




prompt_args = PromptArgs(length=5, 
                         embed_dim=768, 
                         embedding_key='mean', 
                         prompt_init='uniform',
                         prompt_pool=True,
                         prompt_key=True,
                         pool_size=10,
                         top_k=3,
                         batchwise_prompt=False,
                         prompt_key_init='uniform')


# DATA PROCESSING FOR FSC
def data_processing(data, processor):
    y = [] 
    x = []

    for i in range(len(data)):
        #get audio signal
        audio_sig = data[i][0]
        spec = processor(audio_sig, sampling_rate=16000, return_tensors='pt')
        x.append(spec['input_values'])
        # intent
        intent = data[i][1]
        y.append(torch.tensor(intent))

    return torch.cat(x), torch.tensor(y)
        
  

# OPTIMIZER
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


# ARGS PARSER
def get_args_parser():
    parser = argparse.ArgumentParser('CiCL for Spoken Language Understandig (Intent classification) on FSC: train and evaluation',
                                     add_help=False)
    
    # Dataset parameters.
    
    parser.add_argument('--data_path', type=str, default='/home/ste/Datasets',help='path to dataset')  #'/data/cappellazzo/slurp'  /data/cappellazzo/CL_SLU
    parser.add_argument('--path_to_save_model',type=str,default='/models_SLURP_wav_SpeechBrain_intentsaug/')
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
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5, 
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--eval_every", type=int, default=1, 
                        help="Eval model every X epochs")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--label_smoothing', type=float, default=0., 
                        help='Label smoothing for the CE loss')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_ratio', type=int, default=2)
    
    
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
    
    parser.add_argument('--memory_size', default=0, type=int,
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
    parser.add_argument('--exp_name', type=str, default='SLURP_SpeechBrain_BPEwav2vec_NOCTC_intents_SFaug_mymod_')
    
    
    return parser
    




def main(args):
    
    if args.use_wandb:
        
        wandb.init(project=args.project_name, name=args.exp_name,entity="umbertocappellazzo",
                   config = {"lr": args.lr, "weight_decay":args.weight_decay, 
                   "epochs":args.epochs, "batch size": args.batch_size})
    
    print(args)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    
    WER = WordErrorRate()
    CER = CharErrorRate()
  
    device = torch.device(args.device)
    torch.set_num_threads(20)
    
    # Fix the seed for reproducibility (if desired).
    #seed = args.seed
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    
   
    # FLUENT SPEECH REDEFINITION
    dataset_train = FluentSpeech(args.data_path, train="train", download=False)
    dataset_valid = FluentSpeech(args.data_path, train="valid", download=False)
    dataset_test = FluentSpeech(args.data_path, train="test", download=False)
    
    if args.offline_train:   # Create just 1 task with all classes.
        
        # Added Splitting Crit = None
        scenario_train = ClassIncremental(dataset_train,nb_tasks=1, splitting_crit=None)
        scenario_valid = ClassIncremental(dataset_valid,nb_tasks=1, splitting_crit=None)
        scenario_test = ClassIncremental(dataset_test,nb_tasks=1, splitting_crit=None)
    else:
        
        scenario_train = ClassIncremental(dataset_train,increment=args.increment,initial_increment=args.initial_increment,)                               #class_order=class_order)
        scenario_test = ClassIncremental(dataset_valid,increment=args.increment,initial_increment=args.initial_increment,)
                                         #class_order=class_order)
    
    # Losses employed: CE
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)#,ignore_index=30)

    
    # Use all the classes for offline training.
    
    initial_classes = args.total_classes if args.offline_train else args.initial_increment
    
    
    start_time = time.time()
    
    epochs = args.epochs
    
   
    ###########################################################################
    #                                                                         #
    # Begin of the task loop                                                  #
    #                                                                         #
    ###########################################################################
    
  
    
    for task_id, exp_train in enumerate(scenario_train):
        
        print("Shape of exp_train: ",len(exp_train))
        print(f"Starting task id {task_id}/{len(scenario_train) - 1}")
        
        if task_id == 0:

            print('Creating the CL model:')

            dims = ModelDimensions(args.n_mels, args.kernel_size, args.n_hidden_audio, args.n_head_audio, args.n_layer_audio, args.n_vocab, args.n_hidden_text, n_head_text=1, n_layer_text=1, drop=args.drop, n_feedforward=768*2)
            
            assert (args.n_hidden_audio % args.n_head_audio) == 0, ('Hidden dimension of encoder must be divisible by number of audio heads!')
            assert (args.n_hidden_text % args.n_head_text) == 0, ('Hidden dimension of decoder must be divisible by number of text heads!')
            

            # FIXED PROMPT ARGS: SHOUD BE ADDED TO ARGPARSE
            prompt_args = PromptArgs(length=5, 
                         embed_dim=768, 
                         embedding_key='mean', 
                         prompt_init='uniform',
                         prompt_pool=True,
                         prompt_key=True,
                         pool_size=10,
                         top_k=3,
                         batchwise_prompt=False,
                         prompt_key_init='uniform')

            # AST MODEL   
            model_ckpt = "MIT/ast-finetuned-audioset-10-10-0.4593"
            # MODEL REDEFINITION FROM PRETRAINED

            # pretrained model
            model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            
            # Prompts added
            model = PromptASTClassifier(emb_layer = model._modules['embeddings'],
                        body_layer = model._modules['encoder'],
                        embedding_size=model.config.hidden_size,
                        num_classes=len(dataset_train.class_ids),
                        prompt_args = prompt_args).to(device)
            print(model)


            #model = nn.DataParallel(model)
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('Number of params of the model:', n_parameters)


        
        
            
        # AUDIO PROCESSOR DEFINITION (Creates spectrograms)
        processor = AutoFeatureExtractor.from_pretrained(model_ckpt)


        # OPTIMIZER DEFINITION
        optimizer = AdamW(model.parameters(),lr=args.lr,betas=(0.9,0.98),eps=1e-6,weight_decay=args.weight_decay)
        #optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9,0.98), eps=1e-8, weight_decay=args.weight_decay)
        

        test_taskset = scenario_test[:task_id+1]    # Evaluation on all seen tasks.
        valid_taskset = scenario_valid[:task_id+1]
        
        
        
        # DATALOADERS DEFINITION
        train_loader = DataLoader(exp_train, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  num_workers=args.num_workers,  
                                  collate_fn=lambda x: data_processing(x, processor = processor),
                                  pin_memory=True,
                                  drop_last=False,)
        
        valid_loader = DataLoader(valid_taskset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  num_workers=args.num_workers,
                                  collate_fn=lambda x: data_processing(x, processor = processor),
                                  pin_memory=True, 
                                  drop_last=False,)
        test_loader = DataLoader(test_taskset, 
                                 batch_size=args.batch_size, 
                                 shuffle=False, 
                                 num_workers=args.num_workers,
                                 collate_fn=lambda x: data_processing(x, processor = processor),
                                 pin_memory=True, 
                                 drop_last=False,)
        
        
        warmup_period = len(train_loader)*args.warmup_ratio
        
        
        print("Warm up: ", warmup_period)
        #lr_scheduler = LambdaLR(
        #    optimizer=optimizer, lr_lambda=lambda step: rate(step, model_size=args.n_hidden_audio,factor=1.0,warmup=warmup_period))
        
        
        
        optim = NoamOpt(args.n_hidden_audio, warmup_period, optimizer)
        
    ###########################################################################
    #                                                                         #
    # Begin of the train and test loops                                       #
    #                                                                         #
    ###########################################################################
        print(f"Start training for {epochs} epochs")
        n_valid = int(len(valid_loader)*0.1)
    
        #best_loss = 1e5
        path = os.getcwd() + args.path_to_save_model  #'/models_SLURP_ASR/'
        
        
        
        
        for epoch in range(epochs):
            
            model.train()
            train_loss = 0.
            
            
            
            print(f"Epoch #: {epoch}")
            running_loss = 0

            ###############
            # TRAIN PHASE #
            ###############
            for idx_batch, (x, y) in tqdm(enumerate(train_loader)):   
                

                optim.optimizer.zero_grad()

                print(x.shape)
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
  
                running_loss += loss.item()#*0.7 + ctc_loss*0.3       #*0.50 + 0.50*ctc_loss
                
            
                train_loss += loss.detach().item()
                
                loss.backward()
                optimizer.step()

                if idx_batch % 50 == 49:
                    print(f'[{epoch + 1}, {idx_batch + 1:5d}] loss: {running_loss / 50:.3f}') 
                    running_loss=0.0
            break
                
            ####################
            # EVALUATION PHASE #
            ####################
            if args.eval_every and (epoch+1) % args.eval_every == 0:
                
                model.eval()
                valid_loss = 0.

                train_loss /= len(train_loader)

                print(f"Trainloss at epoch {epoch}: {train_loss}")

                
                list_preds_val = []
                list_gold_val = []
                correct = 0
                total = 0  
                
                with torch.inference_mode():
                    
                    ##############
                    # VALIDATION #
                    ##############
                    for idx_valid_batch, (x_valid, y_valid) in enumerate(valid_loader):
                        
                        x_valid = x_valid.to(device)
                        y_valid = y_valid.to(device)
                        

                        outputs = model(x)
                        # _, predictions = torch.max(outputs, 1)

                        loss = criterion(outputs, y)

                        valid_loss +=  loss
                    
                        
                    valid_loss /= len(valid_loader)

                    print(f"Valid loss: {valid_loss}")

                    #test_text_loss_ /= len(test_loader)
                    #if valid_loss < best_loss:
                    pathh = path + f'model_SF_ep{epoch}.pth'
                    torch.save(model.state_dict(),pathh)
                    #print(f"Saved new model at epoch {epoch}!")
                        #best_loss = valid_loss

                    
                    ########
                    # TEST #
                    ########
                    for idx_test_batch, (x_test, text_test) in enumerate(test_loader):
                        #print("Batch TEST #: ",idx_test_batch)
                        x_test = x_test.to(device)
                        #x_test = x_test.transpose(1,2)
                        text_test = text_test.to(device).squeeze(1)
                        text_to_loss_test = text_test[:,1:]
                        
                        
                        
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
                        
                                
  
                    if args.use_wandb:
                        wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,"val_wer": val_wer, "val_cer": val_cer, "int_acc":intent_accuracy_val,
                                   #"ASR_loss": text_loss_, "CTC_loss": ctc_loss_,
                                   #"WER_train": wer_train, "WER_valid": wer_valid, 
                                   #"WER_test": wer_test
                                   }
                                  )
  
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




