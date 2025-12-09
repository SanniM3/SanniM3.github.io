---
layout: default
title: Why Attention Is Expensive - Understanding the Bottleneck at the Heart of Modern LLMs
---
# Why Attention Is Expensive - Understanding the Bottleneck at the Heart of Modern LLMs

There is a moment, when you first really look at the attention mechanism inside a Transformer, when something feels slightly absurd. The model takes a sequence of $n$ tokens, and for every single one of them, it insists on comparing it to every other token, whether or not they have anything interesting to say. A conversation between two words becomes an all-hands meeting of a thousand. That’s the cost of expressivity: self-attention gives every token a global view of the sequence, but it extracts that view by performing $n^2$ comparisons.

When sequences were short—thirty tokens, maybe a hundred—this didn’t matter. But LLMs changed the scale of the game. Suddenly the model isn’t reading a sentence; it’s reading a book chapter, a legal brief, a source file, a patient history, a chat log stretching back 8,000 tokens. The same “compare everything to everything” rule still applies, and the cost now grows quadratically. It’s not subtle. Double the sequence length and attention becomes four times more expensive. Go from 2k tokens to 16k and, in the worst case, the work multiplies by sixty-four.

This quadratic explosion is only the start of the trouble.

What makes attention truly painful is not just the math—it’s the memory traffic behind the math. GPUs are spectacular at dense matrix multiplies, but only when the data lives close to the compute. As soon as you start thrashing high-bandwidth memory (HBM), performance collapses. And the naive implementation of attention thrashes HBM with almost gleeful abandon. Every time it forms the attention matrix $QK^T$, it reads the full query matrix, the full key matrix, writes the full matrix of pairwise scores, reads it again for softmax, writes the normalized weights, reads them again to multiply by $V$, and finally writes the output. Each of these is a large, dense tensor the GPU has to pull in and out of memory, even though only a small slice of it is “hot” at any given time.

At sufficient scale, attention becomes less of a mathematical operation and more of a memory-shuffling ceremony.

This article is about understanding exactly why that happens—not in hand-wavey terms, but at the level where the bottlenecks are obvious and almost inevitable. It’s the foundation for the next two articles, where we’ll explore how techniques like KV-cache, Multi-Query / Grouped-Query Attention, and FlashAttention bend, dodge, or completely rewrite these bottlenecks.

But before we talk about optimizations, we need to understand the problem deeply.

## **The attention mechanism in its purest form**

Let’s start from the familiar equation:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right)V.
$$

The entire procedure is simple enough:

1. Project each token’s embedding into three vectors:  
   query (what I'm looking for), key (what I contain), and value (what I contribute).  
2. Compare each query to every key.  
3. Use those comparisons to weight all the value vectors.

There is a wonderful symmetry in this. Self-attention is just “ask each token who it should listen to,” encoded as dot products. This elegant mechanism is the reason Transformers have replaced RNNs and CNNs across nearly every domain: it naturally captures long-range dependencies, and it does so without a fixed inductive bias about locality or recurrence.

The trouble begins when we stop admiring the idea and start counting operations.

If the sequence has length $n$, and the model has $h$ heads, each of dimension $d$, then forming the matrix $QK^T$ involves computing $h$ matrices of shape $n \times n$, each cell containing a dot product over a $d$-dimensional space. The total work is on the order of:

O(h n^2 d)

and the memory footprint of those matrices is:

O(h n^2).

It is this $n^2$ that causes everything downstream to break. Nothing else in the Transformer scales so aggressively with sequence length—not the feed-forward networks, not the projections, not the embeddings. Attention is the only part of the architecture where doubling the sequence length squares the work.

To make this more concrete, imagine the FLOP count for a single layer of a LLaMA-sized model. If we wanted to calculate it in Python, it might look like this:

```python
h, n, d = 32, 4096, 128
flops = 2 * h * n**2 * d  # QK^T + softmax(V)
print(f"{flops/1e12:.2f} TFLOPs")
```

For typical settings (32 heads, head dimension 128, sequence length 4096), you would see a number around **137 trillion floating point operations**. And that is only the attention computation for one forward pass of one layer.

A large model stacks dozens of these layers.

This is still only half of the bottleneck.

## **Why naive attention is memory-bound, not compute-bound**

If attention required 137 TFLOPs and the GPU had infinite memory bandwidth, then the solution would be simple: buy more FLOPs. But GPUs don’t work like that. The arithmetic is incredibly fast—Tensor Cores can chew through multiplies at astonishing speeds—but getting data to and from those cores is expensive. HBM is fast relative to CPU DRAM, but glacial compared to the registers and shared memory inside the GPU.

A naive attention implementation does something pathological from the GPU’s perspective: it repeatedly materializes and reads large matrices that don’t fit in fast memory at once.

Here’s an approximate sketch of what happens inside the GPU during naive attention:

1. Read Q and K from HBM.  
2. Compute $QK^T$ and write the full matrix to HBM.  
3. Read the full matrix back from HBM for softmax.  
4. Write the normalized attention weights to HBM.  
5. Read those weights again to multiply them by V.  
6. Write the final output.

Each read/write touches gigabytes of data at long sequence lengths. FLOPs aren’t the problem—bandwidth is.

A useful analogy: imagine you’re a chef with excellent knife skills (compute), but every step of the recipe requires you to walk to a pantry 50 meters away and bring back ingredients one spoonful at a time (memory). Your knife skills don’t matter; you’re bottlenecked by walking back and forth.

This is why almost all attention optimizations—FlashAttention most famously—focus not on reducing FLOPs, but on reducing memory traffic. Compute is cheap; memory is expensive.

## **Training attention vs inference attention: two worlds with different physics**

The moment you move from training to inference, the attention mechanism undergoes a profound shift. This shift is so fundamental that efficient inference techniques look almost unrelated to efficient training techniques—because the bottlenecks flip.

During training, the model sees the entire sequence at once. Every token attends to every other token. The cost is fully quadratic.

During inference, the model generates one token at a time. Each new token only attends to the past.

This means:

- Training is dominated by O(n²) operations.  
- Inference is dominated by O(n) operations per token.  
- But inference accumulates state across tokens.

And that state has a name: the KV cache.

## **The KV cache: where inference meets its own bottleneck**

During autoregressive generation, each new token computes its Q, K, and V vectors. The new token’s Q attends to **all** previously computed K and V vectors. This means the model must store all K and V vectors for all previous tokens across all heads in all layers.

For a sequence length $n$, the KV cache size for a single layer is:

Size = 2 * n * h * d

And in half precision:

Memory = 2nhd * 2 bytes.

For a LLaMA-like model (h = 32, d = 128, n = 4096), this comes out to roughly **134 MB per layer**. With 32 layers, that’s more than **4 GB** just for the KV cache.

This is why inference is often memory-bound before it is compute-bound. It’s also why techniques like Multi-Query Attention, Grouped-Query Attention, and PagedAttention exist at all—they dramatically shrink the KV cache footprint and make inference feasible on real hardware.

We will dig into these methods in **Article 2**.

## **When attention fails completely**

At very long context lengths, naive attention simply stops being usable.

Consider a sequence length of 32,000 tokens. The attention matrix alone contains:

$$
(32,000)^2 = 1.024 \times 10^9 \text{ elements}
$$

At FP16, storing this matrix takes roughly **2 gigabytes**. That is for the attention scores alone—not the gradients, not Q/K/V, not the softmax output.

And this must be computed for every head in every layer.

No GPU can sustainably handle this load.

This is where FlashAttention enters the story. FlashAttention avoids ever materializing the full attention matrix. Instead, it splits the computation into small tiles that fit entirely in fast on-chip memory, streams through the sequence in a single pass, and performs softmax in a numerically stable way without needing the full matrix.

It does the same math, but it does it in a way the hardware likes.

Article 3 will explore this in depth.


## **The core message of this article**

If you boil everything down, the lesson is embarrassingly simple:

Naive attention does too much work, and it does that work in the slowest part of the GPU.

Everything else—KV cache, MQA, GQA, FlashAttention, PagedAttention, long-context transformers, efficient kernels—is just the universe’s attempt to undo or mitigate that mistake.

## **Where we go from here**

Now that you understand the bottleneck deeply—not as a formula, but as an intuition—we are ready for the next two parts.

### **Article 2: The KV Cache, Multi-Query Attention, and Grouped-Query Attention**  
Why inference is bottlenecked by memory, not compute.  
How to reduce KV cache size by 8× to 64×.  
Why every modern LLM uses GQA.

### **Article 3: FlashAttention and IO-Aware Attention**  
Why naive attention is memory-bound.  
How tiling in shared memory changes everything.  
The internal mechanics of FlashAttention v1 → v3.  
What “attention without the attention matrix” really means.

