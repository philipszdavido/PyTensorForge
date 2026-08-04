```
Prompt
   │
Tokenizer (BPE)
   │
Token IDs
   │
Position Embeddings
   │
Word Embeddings
   │
┌──────────────────────────────┐
│ Transformer Block            │
│ Self Attention               │
│ Feed Forward                 │
│ LayerNorm                    │
│ Residual                     │
└──────────────────────────────┘
           × N layers
   │
Linear Projection
   │
Vocabulary Logits
   │
Softmax + Sampling
   │
Next Token
   │
Append to prompt
   │
Repeat until <EOS>
```