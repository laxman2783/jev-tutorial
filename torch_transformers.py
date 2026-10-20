import math

from torch import nn

model = nn.Transformer(d_model=1536, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
print(model)

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        # Set the model dimensionality and vocabulary size
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Instantiate the embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        # Return the token embeddings multiplied by the square root of d_model
        return self.embedding(x) * math.sqrt(self.d_model)

# Instantiate InputEmbeddings and apply it to token_ids
import torch
text = "Service downtime"
text_array = [ord(char) for char in text]
print("Token IDs:", text_array)
token_ids = torch.tensor([text_array], dtype=torch.long)
# Check the values used for d_model and vocab_size
embedding_layer = InputEmbeddings(vocab_size=10_000, d_model=512)
output = embedding_layer(token_ids)
print(output.shape)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.pe = self._build_position_encoding(max_seq_length, d_model)

    def _build_position_encoding(self, max_seq_length, d_model):
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            self.pe = self._build_position_encoding(seq_len, self.d_model)
        return x + self.pe[:, :seq_len]

pos_encoding_layer = PositionalEncoding(d_model=512, max_seq_length=32)
if 'output' not in locals():
    token_ids = torch.tensor([[ord(char) for char in "Service downtime"]], dtype=torch.long)
    embedding_layer = InputEmbeddings(vocab_size=10_000, d_model=512)
    output = embedding_layer(token_ids)
output = pos_encoding_layer(output)
print(output.shape)
print(output[0][0][:10])