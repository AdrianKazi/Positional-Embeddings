# Positional Embeddings in Transformers

This notebook explains how transformers encode token order using positional embeddings.

Transformers process tokens in parallel and do not have a built-in notion of sequence order. Positional embeddings provide information about where each token appears in the sequence so the attention mechanism can reason about relationships between words.

## Structure

The notebook starts with the mathematical intuition required to understand positional encodings:
- vector rotation
- the unit circle
- the relationship between sine, cosine, and dot products

These ideas explain why sinusoidal functions are useful for encoding positions.

## Learned Absolute Positional Embeddings (LAPE)

In this approach each position in the sequence corresponds to a learned vector from a positional embedding matrix.

These vectors are trained together with token embeddings during model training.

Example:
- used in **GPT-2**
- positional matrix size: `context_length × embedding_dim`

## Sinusoidal Positional Embeddings

Instead of learning positional vectors, they are generated using sine and cosine functions with different frequencies.

Each pair of embedding dimensions represents a rotation on the unit circle.  
This structure allows the dot product between positional vectors to encode **relative distance between tokens**.

Used in:
- the original **Transformer (Attention Is All You Need)**.

## Rotary Positional Embeddings (RoPE)

RoPE modifies the attention mechanism by **rotating query and key vectors based on token position**.

Instead of adding positional embeddings to token embeddings, RoPE applies a rotation transformation in the embedding space.

This makes attention scores depend directly on **relative token positions**.

Used in:
- **LLaMA**
- **GPT-NeoX**
- many modern LLM architectures.

## Summary

Three main positional encoding approaches:

| Method | Idea | Example Models |
|------|------|------|
| LAPE | learned positional vectors | GPT-2 |
| Sinusoidal | fixed sin/cos positional functions | Transformer (2017) |
| RoPE | rotation applied to attention vectors | LLaMA, GPT-NeoX |

The notebook combines mathematical intuition and visualizations to illustrate how transformers encode sequence order.
