## Deep and Bidirectional Recurrent Neural Networks

### 1. Deep Recurrent Neural Networks (Deep RNNs)

Deep RNNs are designed to enhance the model's **representational capacity** by stacking multiple recurrent layers vertically. This allows the network to learn hierarchical features in sequential data, much like how deep Feedforward Neural Networks learn progressively abstract features from raw input.

#### **Architecture and Parameters**

A Deep RNN is defined by its number of layers, $L$. Each layer $l$ (where $1 \leq l \leq L$) maintains its own set of parameters and hidden states:

- **Layer-Specific Weights:** Each layer $l$ has unique weight matrices for input-to-hidden ($\mathbf{W}_{xh}^{(l)}$) and hidden-to-hidden ($\mathbf{W}_{hh}^{(l)}$) connections.
- **Hidden State:** Each layer $l$ maintains its own sequence of hidden states, $\mathbf{H}_t^{(l)}$.

![](./imgs/deep-rnn.png)

#### **Information Flow: Vertical and Horizontal**

The computation at any time step $t$ involves two distinct dimensions of flow:

1. **Horizontal Flow (Across Time - Recurrence):** Within any layer $l$, the current hidden state $\mathbf{H}_t^{(l)}$ depends on the preceding hidden state $\mathbf{H}_{t-1}^{(l)}$. This is the fundamental memory mechanism.
2. **Vertical Flow (Across Layers - Depth):** For layers deeper than the first ($l > 1$), the "input" comes from the output of the layer immediately below, $\mathbf{H}_t^{(l-1)}$. This allows the features to be refined and abstracted across the network's depth.

#### **Mathematical Formulation**

Let $\phi(\cdot)$ be the activation function (or the gated logic for LSTM/GRU units).

- **First Layer ($l=1$):** This layer receives the external input $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (batch size $n$, input features $d$).
  $$\mathbf{H}_t^{(1)} = \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(1)} + \mathbf{H}_{t-1}^{(1)} \mathbf{W}_{hh}^{(1)} + \mathbf{b}_h^{(1)})$$

- **Intermediate and Final Layers ($1 < l \leq L$):** These layers treat the lower layer's output as their input.
  $$\mathbf{H}_t^{(l)} = \phi(\mathbf{H}_t^{(l-1)} \mathbf{W}_{xh}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{hh}^{(l)} + \mathbf{b}_h^{(l)})$$

- **Output Layer:** The final output $\mathbf{O}_t$ is typically generated only from the hidden state of the highest layer, $\mathbf{H}_t^{(L)}$:
  $$\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{qh} + \mathbf{b}_q$$

#### **Practical Considerations**

- **Parameter Count:** The total number of parameters scales linearly with the number of layers $L$.
- **Stability:** The computational depth in a Deep RNN is $L \times T$. This exacerbates the vanishing and exploding gradient problems. Therefore, Deep RNNs virtually always require the use of **gated units** (Deep LSTMs or Deep GRUs) in every layer for stable training.

---

### 2. Bidirectional Recurrent Neural Networks (Bi-RNNs)

Bi-RNNs are a powerful architectural enhancement designed to ensure that the representation of a sequence element at time $t$ is informed by **both** the past context and the future context, which is critical for tasks like machine translation or sequence tagging.

#### **The Limitation of Unidirectional RNNs**

A standard (unidirectional) RNN computes $\mathbf{H}_t$ based only on $x_1, x_2, \ldots, x_t$. If the task requires knowledge of $x_{t+1}, x_{t+2}, \ldots$ (e.g., in filling in a blank or determining the part-of-speech of a word), the model is fundamentally limited.

#### **Architecture and Mechanism**

A Bi-RNN solves this by employing two independent sets of recurrent layers running in opposite directions:

1. **Forward Recurrent Layer ($\overrightarrow{\mathbf{H}}$):** Processes the input sequence $\mathbf{X}$ in the normal order ($t=1$ to $T$). The resulting hidden state $\overrightarrow{\mathbf{H}}_t$ encodes the **past context**.
2. **Backward Recurrent Layer ($\overleftarrow{\mathbf{H}}$):** Processes the input sequence in reverse order ($t=T$ down to 1). The resulting hidden state $\overleftarrow{\mathbf{H}}_t$ encodes the **future context**.

![](./imgs/bi-rnn.png)

#### **Contextual Output Generation**

At any time step $t$, the full contextual representation $\mathbf{H}_t$ is obtained by combining the output of the two layers:

$$\mathbf{H}_t = [\overrightarrow{\mathbf{H}}_t ; \overleftarrow{\mathbf{H}}_t]$$

- **Concatenation (Most Common):** The two hidden states are typically concatenated to form a single, richer vector. If each layer has $h$ hidden units, the final contextual vector $\mathbf{H}_t$ has $2h$ units.
- **Final Output:** This combined vector $\mathbf{H}_t$ is then used by the final output layer (e.g., a fully connected layer with Softmax) to make a prediction specific to time step $t$.

#### **Mathematical Recurrence**

The two layers operate independently, each using its own set of parameters ($\mathbf{W}_{xh}, \mathbf{W}_{hh}$).

- **Forward Recurrence (Past Context):**
  $$\overrightarrow{\mathbf{H}}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh}^{\overrightarrow{}} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{hh}^{\overrightarrow{}} + \mathbf{b}_h^{\overrightarrow{}})$$

- **Backward Recurrence (Future Context):**
  $$\overleftarrow{\mathbf{H}}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh}^{\overleftarrow{}} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{hh}^{\overleftarrow{}} + \mathbf{b}_h^{\overleftarrow{}})$$

#### **Applications**

Bi-RNNs are the preferred architecture for tasks where the entire context of the sequence must be considered before making a decision for any single element. This includes:

- **Sequence Tagging:** Part-of-Speech (POS) tagging, Named Entity Recognition (NER).
- **Contextual Embeddings:** Generating context-dependent vectors for polysemous words.
