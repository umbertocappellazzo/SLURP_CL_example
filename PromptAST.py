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

    def forward(self, input_values):

        x = self.emb_layer(input_values)
        print("embedding size")
        print(x.size())
        x_prompted = self.prompt(x)['prompted_embedding']
        print("prompted embedding size:")
        print(x_prompted.size())
        body_output = self.body_layer(x_prompted)
        # print(f"body output: {body_output}")
        out = self.layer_norm(body_output.last_hidden_state)

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
    # print(ast)
    inputs = processor(dataset[3]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    print(inputs)
    with torch.no_grad():
        outputs = ast(**inputs)
        last_hidden_states = outputs
    print(list(last_hidden_states.shape))

if __name__=='__main__':
    main()