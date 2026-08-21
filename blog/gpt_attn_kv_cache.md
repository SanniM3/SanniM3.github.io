# Building GPT-2 From Scratch: From Causal Self-Attention to KV-Cached Decoding

I recently read Sebastain Raschka’s article on understanding and implementing KV cache in LLMs (https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms). It made a very good resource for understanding the idea, and I decided to implement the same idea in a GPT2 implementation, and provide additional breakdowns that helped me fully understand it. This post is essentially me penning down my understanding with an arguable excessive number of notes until every tensor operation made sense to me.

Suppose an LLM is prompted with:

```text
The cat sat on
```

We pass the entire sequence through the model and predict the next token. Little refresher on how attention works in the model: we compute the key, values and query vectors for each of the tokens in the sequence, then compute the attention score (i.e. how relevant is the key vector to the query), normalize scores to a probability distribution (attention weights), and use these weights to computer a weighted sum of value vectors inorder to get the context vectors (i.e. representation that holds the combined information gathered from all relevant value vectores).

<insert attention equation here>

Now, say the model predicts `the`. So far, nothing unusual. But to predict the token after `the`, the standard generation implementation passes this new entire text `the cat sat on the` through the model again. That means recomputing the query, key and value vectors for each of `The`, `cat`, `sat`, and `on`, even though we computed exactly the same vectors one generation step ago.

Generate another token and we do it again.

And again.

The longer the sequence becomes, the more time we spend recomputing representations for tokens whose representations have not changed. This is the problem the **KV cache** solves.

But KV caching is easier to understand once the underlying model is completely transparent. So in this article, I'll build a GPT-2-style decoder-only Transformer from scratch in PyTorch, starting with embeddings and multi-head causal self-attention, before modifying the attention mechanism to cache its keys and values during autoregressive generation.

More importantly, I'll keep track of the tensor shapes throughout. Most of the apparent complexity in attention disappears once it is clear which dimension represents tokens, which represents attention heads, and exactly which matrices are being multiplied.

By the end, the difference between standard generation and KV-cached generation should reduce to one fairly simple idea:

```text
Without KV cache:
recompute the past + compute the new token

With KV cache:
remember the past + compute only the new token
```

The rest is implementation detail.

## The model we are building

I'll use approximately the configuration of the 124M-parameter GPT-2 model:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

At a high level, the architecture is:

```text
token IDs
   ↓
token embeddings + positional embeddings
   ↓
12 × Transformer blocks
   ↓
final layer normalization
   ↓
linear vocabulary projection
   ↓
Logits (distribution over the vocabulary)
```

With each Transformer block containing two large pieces:

```text
Multi-Head Causal Self-Attention
              +
Feed-Forward Network
```

with layer normalization and residual connections around them.



## Turning token IDs into representations

From the earlier high level architecture, we note that the transformer does not receive text directly. The tokenizer first maps text into integer vocabulary IDs.

For example:

```text
"The cat sat on"

        ↓ tokenizer

[token_0, token_1, token_2, token_3]
```

If the batch contains `b` samples/sequences and every sample or sequence contains `num_tokens` tokens, the input tensor generally has shape:

```text
(b, num_tokens)
```

GPT-2 then uses an embedding table, that essentially works as a lookup table to convert token IDs to their corresponding embedding vectors:

```python
self.tok_emb = nn.Embedding(
    cfg["vocab_size"],
    cfg["emb_dim"]
)
```

With an embedding dimension of 768, an input tensor of shape (b, num_tokens) becomes (b, num_tokens,768). So every integer vocabulary ID has become a 768-dimensional vector.

But there is a problem.

Attention itself has no inherent notion of token order. If we gave the model the same set of tokens in a different order, the token embedding alone would not tell which token appeared first, second and so on. GPT2 incorporated this positional information using a second learnable embedding table. Instead of mapping a token_id to a vector (like we did in the token embedding layer), this table maps each possible position in the sequence to its own vector of size `emb_dim = 768` Since the model can process at most `context_length` tokens, there are context_lenght possible positional embeddings. 



```python
self.pos_emb = nn.Embedding(
    cfg["context_length"],
    cfg["emb_dim"]
)
```

For a four-token sequence, the positions are:

```text
0, 1, 2, 3
```

and each position is mapped to its own 768-dimensional vector.

Adding the token embeddings and the positional embedding together, we have a new representation that is still of dimension (b, num_tokens, 768)

```python
x = tok_embeds + pos_embeds
```

We can think of this as giving every token two pieces of information:

```text
token embedding       → what am I?
positional embedding  → where am I?
```

Both will become important later. In particular, the positional embedding becomes slightly tricky once KV caching means the model no longer receives the entire sequence during every decoding step.

## Attention: Q, K and V

The attention layer begins with three learned linear projections:

```python
self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
```

Given:

```text
x: (b, num_tokens, 768)
```

we compute:

```python
queries = self.W_query(x)
keys    = self.W_key(x)
values  = self.W_value(x)
```

which gives:

```text
Q: (b, num_tokens, 768)
K: (b, num_tokens, 768)
V: (b, num_tokens, 768)
```

The easiest intuition I have for these three representation is:

```text
Query → what information am I looking for?
Key   → what information do I contain?
Value → what information should I contribute if selected?
```

Attention first compares queries with keys. It then uses those comparisons to decide how strongly to combine the values. Note this distinction as it becomes important when we eventually ask why a **KV** cache stores keys and values but does not bother caching queries.

## Splitting one attention operation into twelve

This model uses twelve attention heads. Since `embedding dimension = 768` and `number of heads = 12`, each head receives `768/12 = 64` dimensional vectors. We refer to this smaller dimension as the `head_dimension`. Therefore, we can reshape our (b, num_tokens, emb_dim) dimensional queries, keys and values by ‘splitting’ the last emb_dim into a product of `num_heads` and `head_dim. The resulting dimension would change from (b, num_tokens, 768) to (b, num_tokens, 12, 64).


```python
queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
values = values.view(b, num_tokens, self.num_heads, self.head_dim)
```

Note that we have not thrown information away. We have simply interpreted the 768 features as twelve groups of 64 features.

For the attention calculation, it is more convenient to place the head dimension before the token dimension. Here, we are not **regrouping** the tensor elements, rather, we are **swapping** the token and head axes so the tensor changes from (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim).

```python
queries = queries.transpose(1, 2)
keys    = keys.transpose(1, 2)
values  = values.transpose(1, 2)
```
