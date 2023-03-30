import argparse
import main_offline
from model_prompt import ModelDimensions, Seq2SeqTransformer
from prompt import PromptArgs
import torch

print(torch.__version__)

def l2_normalize(x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

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

prompt_args = PromptArgs(length=5, 
                         embed_dim=200, 
                         embedding_key='mean', 
                         prompt_init='uniform',
                         prompt_pool=True,
                         prompt_key=True,
                         pool_size=10,
                         top_k=3,
                         batchwise_prompt=False,
                         prompt_key_init='uniform')

model = Seq2SeqTransformer(dims=dims, prompt_args=prompt_args)

a = torch.randn(5)
print(a)
print(l2_normalize(a, dim=0))

audio = torch.randn(5, 200)
text = torch.randn(5,25)
print(model(audio,text))

