# Greedy Search & Beam Search

When we introduced the encoder-decoder architecture, we only used the
_greedy_ strategy where we select at each time step the token with the
highest predicted probability of coming next, until we find the special
end-of-sequence token.

## Greedy Search

This strategy might look reasonable, and in fact it is not so bad.
However, it might be more reasonable to search for the _most likely sequence_
, not the sequence of (greedily selected) _most likely tokens_.
These two objects is quite different actually.

The most likely sequence is the one that maximizes the expression
$\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$
,which is the sum of all token probabilities knowing the past tokens selected.

## Exhaustive Search

To guarantee finding the most likely sequence,
we could employ Exhaustive Search,
enumerating all possible output sequences and selecting the one with the maximum product of conditional probabilities.

Limitation: The computational cost of exhaustive search is $O(|\mathcal{Y}|^{T'})$,
which is exponential in the sequence length.
Given typical vocabulary sizes (e.g., $|\mathcal{Y}| \approx 10,000$)
and sequence lengths (e.g., $T' \approx 10$),
this quickly becomes computationally prohibitive,
making it impractical for real-world applications.

## Beam Search

**Beam Search** is a heuristic algorithm that provides a practical trade-off
between the speed of greedy search and the optimality of exhaustive search.
It is controlled by a hyperparameter, the _beam size_ $k$.

The process of beam search is as follows:

1. **Time Step 1:** Select the $k$ tokens with the highest conditional
   probability $P(y'_1 \mid \mathbf{c})$.
   These become the initial $k$ candidate sequences.
2. **Subsequent Time Steps $t' > 1$:**
   - For the $k$ candidate sequences from step $t'-1$,
     calculate the conditional probability of generating every possible next token
     $y \in \mathcal{Y}$. This results in $k \cdot |\mathcal{Y}|$ potential new sequences.
   - From these $k \cdot |\mathcal{Y}|$ possibilities,
     select the top $k$ resulting sequences with the highest _cumulative_ predicted probability
     (i.e., the highest $\prod_{i=1}^{t'} P(y'_{i} \mid y'_1, \dots, y'_{i-1}, \mathbf{c})$).
   - These $k$ sequences become the candidates for the next time step.
3. **Termination:** The process continues until all $k$ candidate sequences
   terminate with the `<eos>` token or the maximum length $T'$ is reached. The
   final output sequence is the one among the $k$ completed sequences that has
   the highest overall probability.

Beam search maintains the $k$ most promising sequences at each time step,
pruning the vast majority of other possibilities.

**Relationship to Other Strategies:**

- **Greedy Search** is a special case of beam search where the beam size is $k=1$.
- As $k \to |\mathcal{Y}|^{T'}$, beam search approaches exhaustive search, though in practice, $k$ is kept small (e.g., $k=10$).

Beam search offers significantly better results than greedy search with a much smaller increase in computational cost compared to exhaustive search.
