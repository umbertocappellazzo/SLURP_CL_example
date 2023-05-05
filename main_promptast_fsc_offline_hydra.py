import random
from torch.utils.data import DataLoader
from functools import partial
from torch.optim import Adam, AdamW
from Speech_CLscenario.fluentspeech import FluentSpeech
from Speech_CLscenario.slurp_aug import Slurp
from Speech_CLscenario.class_incremental import ClassIncremental
import torch
import torch.nn.functional as F
from continuum.metrics import Logger
import numpy as np
from model_prompt import Seq2SeqTransformer, ModelDimensions
from tools.utils import get_kdloss,get_kdloss_onlyrehe, freeze_parameters
import time
import datetime
import wandb
from Speech_CLscenario.memory import RehearsalMemory
from statistics import mean
import math
from torchaudio import transforms as tr
import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import torch.nn as nn
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
#Hydra
import hydra


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


@hydra.main(version_base=None, config_path='config', config_name='prompt_ast_fsc')
def main(args) -> None:


    if args.use_wandb: 
        wandb.init( project=args.project_name, 
                    name=args.exp_name,
                    entity="sciapponi",
                    config = {  "lr": args.lr, 
                                "weight_decay":args.weight_decay, 
                                "epochs":args.epochs, 
                                "batch size": args.batch_size,
                                "prompt length": args.prompt.length,
                                "prompt pool size": args.prompt.pool_size,
                                "prompt top k": args.prompt.top_k
                                })
    
    print(f"Prompt Length:{args.prompt.length}    Prompt Pool Size:{args.prompt.pool_size}    Prompt TopK:{args.prompt.top_k}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    
    device = torch.device(args.device)
    torch.cuda.set_device(0)
    torch.set_num_threads(20)
    
    
    # Fix the seed for reproducibility (if desired).
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)  
   
    # FLUENT SPEECH REDEFINITION
    dataset_train = FluentSpeech(args.data_path, max_len_audio=64000, train="train", download=False)
    dataset_valid = FluentSpeech(args.data_path,max_len_audio=64000, train="valid", download=False)
    dataset_test = FluentSpeech(args.data_path,max_len_audio=64000, train="test", download=False)
    
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

            # dims = ModelDimensions(args.n_mels, args.kernel_size, args.n_hidden_audio, args.n_head_audio, args.n_layer_audio, args.n_vocab, args.n_hidden_text, n_head_text=1, n_layer_text=1, drop=args.drop, n_feedforward=768*2)
            
            # assert (args.n_hidden_audio % args.n_head_audio) == 0, ('Hidden dimension of encoder must be divisible by number of audio heads!')
            # assert (args.n_hidden_text % args.n_head_text) == 0, ('Hidden dimension of decoder must be divisible by number of text heads!')
            

            # FIXED PROMPT ARGS: SHOUD BE ADDED TO ARGPARSE
            prompt_args = PromptArgs(length=args.prompt.length, 
                                        embed_dim=args.prompt.embed_dim, 
                                        embedding_key=args.prompt.embedding_key, 
                                        prompt_init=args.prompt.prompt_init,
                                        prompt_pool=args.prompt.prompt_pool,
                                        prompt_key=args.prompt.prompt_key,
                                        pool_size=args.prompt.pool_size,
                                        top_k=args.prompt.top_k,
                                        batchwise_prompt=args.prompt.batchwise_prompt,
                                        prompt_key_init=args.prompt.prompt_key_init)

            # AST MODEL   
            model_ckpt = "MIT/ast-finetuned-audioset-10-10-0.4593"
            # MODEL REDEFINITION FROM PRETRAINED

            # pretrained model
            model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            torch.cuda.empty_cache() 
            # Prompts added
            model = PromptASTClassifier(emb_layer = model._modules['embeddings'],
                        body_layer = model._modules['encoder'],
                        embedding_size=model.config.hidden_size,
                        num_classes=len(dataset_train.class_ids),
                        prompt_args = prompt_args).to(device)
            
            # Freezing model layers for Prompt Tuning

            # model.emb_layer.requires_grad_(False)
            model.body_layer.requires_grad_(False)

            # print(model)


            #model = nn.DataParallel(model)
            n_parameters = sum(p.numel() for p in model.parameters())
            print('Number of params of the model:', n_parameters)
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('Number of trainable params of the model:', n_parameters)
            print(f"Prompt Parameters:{sum(p.numel() for p in model.prompt.parameters() if p.requires_grad)}")
            print(f"Encoder Parameters:{sum(p.numel() for p in model.body_layer.parameters() if p.requires_grad)}")
            print(f"Classification Head Parameters:{sum(p.numel() for p in model.classification_head.parameters() if p.requires_grad)}")

        
        
            
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
            
            print(f"Epoch #: {epoch}")

            model.train()
            train_loss = 0.
            running_loss = 0
            total = 0.
            accuracy = 0.
            ###############
            # TRAIN PHASE #
            ###############
            for idx_batch, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):   
                

                optim.optimizer.zero_grad()
                
                # print(x.shape)
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                _, predictions = torch.max(outputs['classification_head'], 1)

                loss = criterion(outputs['classification_head'], y)
                # print(f"loss1: {loss}")
                # loss = loss - 0.5 * outputs['reduce_sim'] #0.5= standard lambda coefficient used on the L2P Paper. Other Coeff. coulf be tested.
                # print(f"loss2: {loss}")

                # if idx_batch % 8 == 7:
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                train_loss += loss.detach().item()

                total += y.size(0)
                accuracy += (predictions == y).sum().item() 

                

                if idx_batch % 50 == 49:
                    print(f'[{epoch + 1}, {idx_batch + 1:5d}] loss: {running_loss / 50:.3f}') 
                    running_loss=0.0
                    # print(model.prompt.prompt)

            intent_accuracy_train = (100 * accuracy / total)
            print(f"Intent Accuracy Train: {intent_accuracy_train}")
                
                
            ####################
            # EVALUATION PHASE #
            ####################
            if args.eval_every and (epoch+1) % args.eval_every == 0:
                
                model.eval()
                valid_loss = 0.
                test_loss = 0.
                train_loss /= len(train_loader)
                list_preds_val = []
                total = 0.
                accuracy = 0.
                print(f"Trainloss at epoch {epoch}: {train_loss}")

                
                

                
                with torch.inference_mode():
                    
                    ##############
                    # VALIDATION #
                    ##############
                    for idx_valid_batch, (x_valid, y_valid) in enumerate(valid_loader):
                        
                        x_valid = x_valid.to(device)
                        y_valid = y_valid.to(device)
                        

                        outputs = model(x_valid)
                        # _, predictions = torch.max(outputs, 1)

                        loss = criterion(outputs['classification_head'], y_valid)
                        # loss = loss - 0.5 * outputs['reduce_sim']

                        valid_loss +=  loss
                        
                    
                        
                    valid_loss /= len(valid_loader)

                    print(f"Valid loss: {valid_loss}")


                    pathh = path + f'model_SF_ep{epoch}.pth'
                    torch.save(model.state_dict(),pathh)
                    print(f"Saved new model at epoch {epoch}!")
                    # best_loss = valid_loss

                    
                    ########
                    # TEST #
                    ########
                    for idx_test_batch, (x_test, y_test) in enumerate(test_loader):
                        
                        x_test = x_test.to(device)
                        y_test = y_test.to(device)
                        
                        
                        
                        # if idx_test_batch == (len(test_loader) -2):

                        outputs = model(x_test)
                        _, predictions = torch.max(outputs['classification_head'], 1)
                        list_preds_val.append(predictions)
                        loss = criterion(outputs['classification_head'], y_test)
                        # loss = loss - 0.5 * outputs['reduce_sim']
                        test_loss +=  loss
                        total += y_test.size(0)
                        accuracy += (predictions == y_test).sum().item()
                    
                    test_loss /= len(test_loader)
                    print(f"Test Loss:{test_loss}")
                    intent_accuracy_test = (100 * accuracy / total)

                    print(f"Intent Accuracy Test: {intent_accuracy_test}")
                    

  
                    if args.use_wandb:
                        wandb.log({"train_loss": train_loss, "valid_loss": valid_loss, "test_loss": test_loss, "intent_acc_test":intent_accuracy_test, "intent_acc_train":intent_accuracy_train})
            
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    if args.use_wandb:
        wandb.finish()     
            
            
            
if __name__ == "__main__":
    main()

