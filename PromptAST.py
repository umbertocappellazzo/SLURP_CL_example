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
                 leayer_norm,
                 n_prompts, 
                 hidden_size, 
                 prompt_args: PromptArgs):
        
        super().__init__()
        self.emb_layer = emb_layer
        self.body_layer = body_layer
        self.layer_norm = leayer_norm
        self.n_prompts = n_prompts
        self.hidden_size = hidden_size
        self.prompt = Prompt(prompt_args)

    def forward(self, x):

        x = self.emb_layer(x)

        x_prompted = self.prompt(x)['prompted_embedding']
        # torch zeros sarebbe un'ipotetica selezione di prompts
        body_output = self.body_layer(x_prompted)
        out = self.layer_norm(body_output)

        return out



def main():
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate

    processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")