import torch
import torch.nn as nn
from typing import List

class CharEmbeddingModule(nn.Module):
    def __init__(self, vocab_size: int, char_emb_dim: int = 32, padding_idx: int = 0):
        super().__init__()
        # nn.Embedding acts as a lookup table mapping token IDs to vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=char_emb_dim,
            padding_idx=padding_idx
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape:  [batch_size, num_words, max_word_length]
        # Output shape: [batch_size, num_words, max_word_length, char_emb_dim]
        return self.embedding(x)


# ==========================================================
# Phase 2.1 Verification: Hooking your Preprocessor Output up
# ==========================================================
if __name__ == "__main__":
    # 1. Get the matrix from your preprocessor
    output_matrix: List[List[int]] = [
        [2, 12, 9, 21, 3, 0, 0, 0, 0, 0], 
        [2, 17, 5, 7, 12, 13, 18, 9, 3, 0]
    ]
    
    # 2. Package into a PyTorch Tensor and add an explicit batch dimension using unsqueeze(0)
    # Target Shape: [Batch=1, Words=2, Chars=10]
    input_tensor = torch.tensor(output_matrix, dtype=torch.long).unsqueeze(0)
    print("Input Tensor Shape:", input_tensor.shape)
    
    # 3. Initialize the Embedding Module 
    # High bound estimate for character vocabulary size (e.g., 256 characters + offset)
    VOCAB_SIZE = 300 
    CHAR_EMB_DIM = 32 # Every single ID becomes a dense vector of size 32
    
    embedder = CharEmbeddingModule(vocab_size=VOCAB_SIZE, char_emb_dim=CHAR_EMB_DIM, padding_idx=0)
    
    # 4. Generate character embeddings!
    embedded_output = embedder(input_tensor)
    print("Embedded Output Tensor Shape:", embedded_output.shape) 
    # Expected: torch.Size([1, 2, 10, 32])