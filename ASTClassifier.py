import torch
from torch import nn
from transformers import ASTModel


class ASTClassifier(nn.Module):

    def __init__(self, num_classes):
        
        super().__init__()
        self.encoder = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.classification_head = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_values):
        
        x = self.encoder(input_values)
        out = self.classification_head(torch.mean(x.last_hidden_state, 1))

        return out
    


if __name__=="__main__":
    model_ckpt = "MIT/ast-finetuned-audioset-10-10-0.4593"

    torch.cuda.empty_cache() 
    # Prompts added
    model = ASTClassifier(num_classes=32).to("cpu")
    print(model.encoder.encoder)
