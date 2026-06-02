import torch
import torch.nn as nn
from src.models.decoder import TransformerDecoderBlock
from src.models.layer_norm import LayerNorm

class GPT(nn.Module):
    """
    Decoder‑only Transformer (GPT‑style) built from scratch.
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff, max_seq_len, dropout=0.1, pad_token_id=None,
                 init_type: str = "gpt2"):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False) #final projection to vocab size, no bias since we have weight tying with the token embedding

        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # Weight tying
        self.token_embedding.weight = self.head.weight

        # Apply chosen initialization
        self._init_weights(init_type)

    def _init_weights(self, init_type: str):
        """Applies the specified weight initialization strategy."""
        if init_type == "gpt2":
            # GPT‑2 style: normal(0, 0.02) for Linear & Embedding, bias=0
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    
        elif init_type == "xavier":
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif init_type == "kaiming":
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif init_type == "normal_small":
            # has smaller variance for deeper models
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
        else:
            raise ValueError(f"Unknown init_type: {init_type}")

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) with 1 for real tokens, 0 for pad
        Returns:
            logits: (batch, seq_len, vocab_size)
            attn_weights_list: list of attention weight tensors from each block
        """
        B, T = input_ids.shape #(batch, seq_len)
        if T > self.max_seq_len: #enure that the input sequence length does not exceed the model's maximum sequence length, which is determined by the size of the position embedding. This is important to prevent indexing errors and ensure that the model can handle the input properly.
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}") 

        # Token + position embeddings
        tok_emb = self.token_embedding(input_ids)   # (B, T, d_model)
        #absolute token positions
        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0).expand(B, T) # (1, T) -> (B, T)

        pos_emb = self.position_embedding(positions) # (B, T, d_model)

        x = self.dropout(tok_emb + pos_emb)

        # create the causal mask: upper triangular = True (masked)
        causal_mask = torch.triu(
            torch.ones(T, T, device=input_ids.device, dtype=torch.bool),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        # Combine with padding mask if provided
        if attention_mask is not None and self.pad_token_id is not None:
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            causal_mask = causal_mask | pad_mask #combine causal and padding masks

        attn_weights_list = [] 
        atten_weights_dict = {}
        for i,block in enumerate(self.blocks):
            x, attn_weights = block(x, causal_mask)
            attn_weights_list.append(attn_weights) #collect attention weights from each block for visualization or analysis
            atten_weights_dict[f'block_{i}'] = attn_weights #store attention weights in a dictionary with block index as key for easier access

        x = self.norm(x)
        logits = self.head(x)   # (B, T, vocab_size)
        return logits, attn_weights_list, atten_weights_dict #no softrmax here since we'll use CrossEntropyLoss which applies it internally

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())