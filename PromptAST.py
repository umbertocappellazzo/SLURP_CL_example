import torch
from torch import nn
import transformers
from prompt import Prompt, PromptArgs
from transformers import AutoProcessor, ASTModel
from datasets import load_dataset

class PromptAST(nn.Module):

    def __init__(self, 
                 emb_layer, 
                 body_layer, 
                 layer_norm,
                 prompt_args: PromptArgs):
        
        super().__init__()
        self.emb_layer = emb_layer
        self.body_layer = body_layer
        self.layer_norm = layer_norm
        self.prompt = Prompt(prompt_args)

    def forward(self, x):

        x = self.emb_layer(x)

        x_prompted = self.prompt(x)['prompted_embedding']
        body_output = self.body_layer(x_prompted)
        out = self.layer_norm(body_output)

        return out



def main():

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
    
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate

    processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    ast = PromptAST(emb_layer = model._modules['embeddings'],
                    body_layer = model._modules['encoder'],
                    layer_norm = model._modules['layernorm'],
                    prompt_args = prompt_args)
    print(ast)
if __name__=='__main__':
    main()