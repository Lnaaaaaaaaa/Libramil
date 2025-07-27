import torch
import torch.nn as nn
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

class TextEncoder(nn.Module):
    def __init__(self, output_dim=512, weights_path="conch/pytorch_model.bin" ):
        super().__init__()
        self.tokenizer = get_tokenizer()
        self.base_model, _ = create_model_from_pretrained(
            'conch_ViT-B-16',
            weights_path
        )

    def forward(self, texts):
        """
        Args:
            texts: List[str] 
        Returns:
            torch.Tensor: shape [N, output_dim]
        """
        tokenized = tokenize(texts=texts, tokenizer=self.tokenizer)  

        with torch.no_grad():
            text_feats = self.base_model.encode_text(tokenized)  

        return text_feats