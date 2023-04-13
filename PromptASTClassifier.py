import torch
from torch import nn
import transformers
from prompt import Prompt, PromptArgs
from transformers import AutoProcessor, ASTModel, ASTFeatureExtractor, ASTConfig, AutoFeatureExtractor
from datasets import load_dataset
import soundfile as sf
import numpy as np
from matplotlib import pyplot as plt

class PromptASTClassifier(nn.Module):

    def __init__(self, 
                 emb_layer, 
                 body_layer, 
                 embedding_size,
                 num_classes,
                 prompt_args: PromptArgs):
        
        super().__init__()
        self.emb_layer = emb_layer
        self.body_layer = body_layer
        self.classification_head = nn.Linear(embedding_size, num_classes)
        self.prompt = Prompt(prompt_args)

    def forward(self, input_values):

        x = self.emb_layer(input_values)
        print("embedding size")
        print(x.size())
        x_prompted = self.prompt(x)['prompted_embedding']
        print("prompted embedding size:")
        print(x_prompted.size())
        body_output = self.body_layer(x_prompted)
        out = self.classification_head(torch.mean(body_output.last_hidden_state, 1))
        # print(f"body output: {body_output}")
        

        return out



def main():

    prompt_args = PromptArgs(length=5, 
                         embed_dim=768, 
                         embedding_key='mean', 
                         prompt_init='zero',
                         prompt_pool=True,
                         prompt_key=True,
                         pool_size=10,
                         top_k=3,
                         batchwise_prompt=False,
                         prompt_key_init='uniform')
    
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate

    # processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", max_length = 4*sampling_rate, ignore_mismatched_sizes=True)
    processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    # configuration = ASTConfig(max_length = 4*sampling_rate)
    # processor = AutoProcessor.from_pretrained(configuration)
    # model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", max_length = 4*sampling_rate, ignore_mismatched_sizes=True)
    model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    # model = ASTModel.from_pretrained(configuration)
    # print(model._modules)
    print(model.config.hidden_size)
    ast = PromptASTClassifier(emb_layer = model._modules['embeddings'],
                    body_layer = model._modules['encoder'],
                    embedding_size = model.config.hidden_size,
                    num_classes = 31,
                    prompt_args = prompt_args)
    # print(ast)
    print(len(dataset[65]["audio"]["array"]))
    # sf.write('amazing_sound.wav', np.array(dataset[65]["audio"]["array"]), sampling_rate)
    inputs = processor(dataset[65]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    print(inputs['input_values'][0].shape)

    with torch.no_grad():
        outputs = ast(**inputs)
        last_hidden_states = outputs
    print(list(last_hidden_states.shape))
    print(last_hidden_states)
if __name__=='__main__':
    main()