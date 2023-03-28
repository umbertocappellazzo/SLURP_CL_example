import argparse
import main_offline
from model_prompt import ModelDimensions, Seq2SeqTransformer
from prompt import PromptArgs
parser = argparse.ArgumentParser('CiCL for Spoken Language Understandig (Intent classification) on FSC: train and evaluation',
                                    parents=[main_offline.get_args_parser()])
args = parser.parse_args()

dims = ModelDimensions(args.n_mels, 
                       args.kernel_size, 
                       args.n_hidden_audio, 
                       args.n_head_audio, 
                       args.n_layer_audio, 
                       args.n_vocab, 
                       args.n_hidden_text, 
                       n_head_text=1, 
                       n_layer_text=1, 
                       drop=args.drop, 
                       n_feedforward=768*2)

prompr_args = PromptArgs(length=5, 
                         embed_dim=768, 
                         embedding_key='mean', 
                         prompt_init='uniform',
                         prompt_pool=True,
                         prompt_key=True,
                         pool_size=10,
                         top_k=3,
                         batchwise_prompt=False,
                         prompt_key_init='uniform')

model = Seq2SeqTransformer(dims=dims, prompt_args=prompr_args)
print(model)