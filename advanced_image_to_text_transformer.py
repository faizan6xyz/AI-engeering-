"""
===============================================================================
ADVANCED IMAGE-TO-TEXT TRANSFORMER (Vision-Language Model)
===============================================================================
Comprehensive Implementation with Workflow, Explanations, and Code Examples

This module implements an advanced image-to-text transformer that combines:
1. Vision Transformer (ViT) - Encodes images into visual embeddings
2. Transformer Decoder - Generates text captions using visual context
3. Cross-Attention Mechanisms - Aligns visual and text representations
4. Pre-training & Fine-tuning - Complete training pipeline

WORKFLOW:
=========
Input Image → ViT Encoder → Visual Embeddings 
           → Transformer Decoder (with Cross-Attention) 
           → Text Tokens → Output Caption

===============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import vit_b_32
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 1: VISION TRANSFORMER ENCODER (Image Feature Extraction)
# ============================================================================

class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer (ViT) Encoder for extracting visual features from images.
    
    Key Components:
    - Patch Embedding: Divides image into patches and projects them
    - Positional Encoding: Adds spatial information
    - Transformer Blocks: Multi-head self-attention + Feed-forward networks
    
    How it works:
    1. Image (H×W×3) → Patches (N, patch_dim)
    2. Linear projection → Embeddings (N, d_model)
    3. Add positional encodings
    4. Pass through L transformer blocks
    Output: Visual features (N, d_model) + class token for global context
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        
        # Patch embedding: Linear projection of flattened patches
        self.patch_embedding = nn.Linear(patch_dim, d_model)
        
        # Class token (learnable token for global image representation)
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding for each patch + class token
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model)
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image tensor (B, 3, H, W)
        Returns:
            patch_features: Patch embeddings (B, N, d_model)
            class_features: Class token output (B, 1, d_model)
        """
        B = x.shape[0]
        
        # Convert image to patches
        patches = self._image_to_patches(x)  # (B, N, patch_dim)
        
        # Project patches to embedding dimension
        patch_embeddings = self.patch_embedding(patches)  # (B, N, d_model)
        
        # Add class token
        class_tokens = self.class_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([class_tokens, patch_embeddings], dim=1)  # (B, N+1, d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding
        x = self.dropout(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (B, N+1, d_model)
        
        # Split class token and patch features
        class_features = x[:, 0:1, :]  # (B, 1, d_model)
        patch_features = x[:, 1:, :]   # (B, N, d_model)
        
        return patch_features, class_features
    
    def _image_to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patches.
        Args:
            x: Image tensor (B, 3, H, W)
        Returns:
            Flattened patches (B, N, patch_dim) where N = (H/patch_size)^2
        """
        B, C, H, W = x.shape
        P = self.patch_size
        
        # Reshape: (B, 3, H, W) → (B, 3, num_patches_h, P, num_patches_w, P)
        x = x.reshape(
            B, C, H // P, P, W // P, P
        )
        # Transpose: (B, 3, num_patches_h, P, num_patches_w, P) 
        #          → (B, num_patches_h, num_patches_w, 3, P, P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        # Flatten: (B, N, 3*P*P)
        x = x.reshape(B, -1, C * P * P)
        
        return x


# ============================================================================
# SECTION 2: MULTI-HEAD CROSS-ATTENTION (Vision-Text Alignment)
# ============================================================================

class MultiHeadCrossAttention(nn.Module):
    """
    Cross-Attention mechanism that aligns visual and text representations.
    
    Purpose: Allow text decoder to attend to relevant visual features
    
    Mechanism:
    - Query: From text tokens (decoder)
    - Key/Value: From visual features (encoder)
    - Output: Text tokens enriched with visual context
    
    Formula:
    Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: Text tokens (B, T, d_model)
            key: Visual features (B, V, d_model)
            value: Visual features (B, V, d_model)
            mask: Attention mask (optional)
        Returns:
            Context vectors (B, T, d_model)
        """
        B = query.shape[0]
        
        # Project and reshape for multi-head attention
        Q = self.W_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention weights
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(context)
        
        return output


# ============================================================================
# SECTION 3: TRANSFORMER DECODER (Text Generation)
# ============================================================================

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder that generates text captions.
    
    Architecture:
    1. Self-Attention: Text attends to itself (for context)
    2. Cross-Attention: Text attends to visual features
    3. Feed-Forward: Non-linear transformation
    
    This creates a tight coupling between visual and textual information.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_dim: int = 2048,
        max_seq_length: int = 50,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Embedding layer for text tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding for text
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, d_model)
        )
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        visual_features: torch.Tensor,
        visual_class: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            token_ids: Input token IDs (B, T)
            visual_features: Patch features from encoder (B, V, d_model)
            visual_class: Class token from encoder (B, 1, d_model)
            causal_mask: Causal mask for preventing future token attention
        Returns:
            Logits for next token (B, T, vocab_size)
        """
        # Embed and add positional encoding
        x = self.embedding(token_ids)  # (B, T, d_model)
        x = x * np.sqrt(self.d_model)
        x = x + self.positional_encoding[:, :x.shape[1], :]
        x = self.dropout(x)
        
        # Concatenate visual class token with visual features for context
        visual_context = torch.cat([visual_class, visual_features], dim=1)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, visual_context, causal_mask)
        
        # Project to vocabulary
        logits = self.output_projection(x)  # (B, T, vocab_size)
        
        return logits


class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention and cross-attention."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-attention (text attends to itself)
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-attention (text attends to visual features)
        self.cross_attention = MultiHeadCrossAttention(
            d_model, num_heads, dropout
        )
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model)
        )
        
        # Layer normalization and residual connections
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        visual_context: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attention(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross-attention
        cross_attn_out = self.cross_attention(x, visual_context, visual_context)
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        # Feed-forward
        mlp_out = self.mlp(x)
        x = self.norm3(x + self.dropout(mlp_out))
        
        return x


# ============================================================================
# SECTION 4: COMPLETE IMAGE-TO-TEXT MODEL
# ============================================================================

class ImageToTextTransformer(nn.Module):
    """
    Complete Vision-Language Model combining ViT Encoder and Transformer Decoder.
    
    WORKFLOW:
    =========
    1. Input: Image (224×224×3) + Previous tokens
    2. Vision Encoder: Extract visual features using ViT
    3. Decoder: Generate next token using visual + textual context
    4. Output: Predicted token probabilities
    5. Repeat: Generate caption token-by-token until [END] token
    
    Training: Cross-entropy loss between predicted and actual tokens
    Inference: Greedy/Beam search decoding
    """
    
    def __init__(
        self,
        vocab_size: int,
        image_size: int = 224,
        patch_size: int = 16,
        d_model_encoder: int = 768,
        d_model_decoder: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        max_caption_length: int = 50,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_caption_length = max_caption_length
        
        # Vision Transformer Encoder
        self.encoder = VisionTransformerEncoder(
            image_size=image_size,
            patch_size=patch_size,
            d_model=d_model_encoder,
            num_heads=12,
            num_layers=12,
            dropout=dropout
        )
        
        # Projection layer: encoder → decoder dimension
        self.encoder_projection = nn.Linear(d_model_encoder, d_model_decoder)
        
        # Transformer Decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model_decoder,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_length=max_caption_length,
            dropout=dropout
        )
        
        self.d_model_decoder = d_model_decoder
    
    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        caption_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Training forward pass.
        
        Args:
            images: Batch of images (B, 3, 224, 224)
            captions: Target captions (B, max_len)
            caption_lengths: Length of each caption (B,)
        Returns:
            Logits (B, max_len, vocab_size)
        """
        # Encode images
        patch_features, class_features = self.encoder(images)
        
        # Project encoder features to decoder dimension
        patch_features = self.encoder_projection(patch_features)
        class_features = self.encoder_projection(class_features)
        
        # Prepare input for decoder (shift captions by 1 for teacher forcing)
        decoder_input = captions[:, :-1]
        
        # Generate causal mask for decoder (prevent attending to future tokens)
        T = decoder_input.shape[1]
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf')), diagonal=1
        ).to(decoder_input.device)
        
        # Decode
        logits = self.decoder(
            decoder_input,
            patch_features,
            class_features,
            causal_mask=causal_mask
        )
        
        return logits
    
    def generate_caption(
        self,
        image: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> List[int]:
        """
        Generate caption for a single image using greedy decoding.
        
        INFERENCE WORKFLOW:
        ===================
        1. Encode image once
        2. Start with [START] token
        3. Repeatedly:
           a. Decode: predict next token
           b. Sample: select token (greedy or top-k)
           c. Append to sequence
           d. Stop if [END] token or max_length reached
        
        Args:
            image: Single image (3, 224, 224)
            start_token_id: ID of [START] token
            end_token_id: ID of [END] token
            max_length: Maximum caption length
            temperature: Softmax temperature (>1: more diverse, <1: more confident)
            top_k: Use top-k sampling instead of greedy
        
        Returns:
            Generated token IDs
        """
        self.eval()
        with torch.no_grad():
            # Encode image
            image = image.unsqueeze(0)  # Add batch dimension
            patch_features, class_features = self.encoder(image)
            patch_features = self.encoder_projection(patch_features)
            class_features = self.encoder_projection(class_features)
            
            # Initialize caption with start token
            caption = [start_token_id]
            
            for _ in range(max_length):
                # Prepare decoder input
                decoder_input = torch.tensor(
                    [caption], device=image.device
                )
                
                # Causal mask
                T = len(caption)
                causal_mask = torch.triu(
                    torch.full((T, T), float('-inf')), diagonal=1
                ).to(image.device)
                
                # Decode
                logits = self.decoder(
                    decoder_input,
                    patch_features,
                    class_features,
                    causal_mask=causal_mask
                )
                
                # Get logits for last token
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sampling strategy
                if top_k is not None:
                    # Top-k sampling
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Convert to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                caption.append(next_token)
                
                # Stop if end token
                if next_token == end_token_id:
                    break
            
            return caption


# ============================================================================
# SECTION 5: DUMMY DATASET & TRAINING PIPELINE
# ============================================================================

class DummyImageCaptionDataset(Dataset):
    """Dummy dataset for demonstration."""
    
    def __init__(self, num_samples: int = 100, vocab_size: int = 1000):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Dummy image
        image = torch.randn(3, 224, 224)
        
        # Dummy caption (sequence of random token IDs)
        caption_length = np.random.randint(10, 50)
        caption = torch.randint(1, self.vocab_size - 1, (caption_length,))
        
        return image, caption


class ImageCaptioningTrainer:
    """Training loop for image-to-text transformer."""
    
    def __init__(
        self,
        model: ImageToTextTransformer,
        vocab_size: int,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.vocab_size = vocab_size
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, captions) in enumerate(train_loader):
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Pad captions to same length
            max_len = 50
            padded_captions = torch.zeros(
                images.shape[0], max_len, dtype=torch.long, device=self.device
            )
            for i, cap in enumerate(captions):
                cap_len = min(len(cap), max_len)
                padded_captions[i, :cap_len] = cap[:cap_len]
            
            # Forward pass
            logits = self.model(images, padded_captions)
            
            # Compute loss
            loss = self.criterion(
                logits.view(-1, self.vocab_size),
                padded_captions[:, 1:].contiguous().view(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss


# ============================================================================
# SECTION 6: EXAMPLE USAGE & DEMONSTRATION
# ============================================================================

def main():
    """
    Demonstration of the Image-to-Text Transformer.
    """
    print("=" * 80)
    print("ADVANCED IMAGE-TO-TEXT TRANSFORMER - DEMONSTRATION")
    print("=" * 80)
    
    # Configuration
    VOCAB_SIZE = 1000
    BATCH_SIZE = 8
    NUM_EPOCHS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n[INFO] Using device: {DEVICE}")
    
    # Initialize model
    print("\n[1] Initializing Model...")
    model = ImageToTextTransformer(
        vocab_size=VOCAB_SIZE,
        image_size=224,
        patch_size=16,
        d_model_encoder=768,
        d_model_decoder=512,
        num_heads=8,
        num_layers=6,
        max_caption_length=50
    )
    print(f"    Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy dataset
    print("\n[2] Creating Dataset...")
    dataset = DummyImageCaptionDataset(num_samples=100, vocab_size=VOCAB_SIZE)
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    print(f"    Dataset size: {len(dataset)}")
    print(f"    Batches per epoch: {len(train_loader)}")
    
    # Initialize trainer
    print("\n[3] Initializing Trainer...")
    trainer = ImageCaptioningTrainer(
        model, VOCAB_SIZE, learning_rate=1e-4, device=DEVICE
    )
    
    # Training loop
    print("\n[4] Training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        avg_loss = trainer.train_epoch(train_loader)
        print(f"Average Loss: {avg_loss:.4f}")
    
    # Generate caption
    print("\n[5] Generating Caption (Inference)...")
    model.eval()
    dummy_image = torch.randn(3, 224, 224).to(DEVICE)
    
    caption = model.generate_caption(
        dummy_image,
        start_token_id=1,
        end_token_id=2,
        max_length=20,
        temperature=1.0,
        top_k=None
    )
    print(f"    Generated caption (token IDs): {caption}")
    print(f"    Caption length: {len(caption)} tokens")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
