# Comprehensive Study Guide & Viva Dossier: PhishGuard AI
## Explainable Recurrent Neural Architectures (RNN, LSTM, GRU) for Long-Sequence Financial Fraud & Phishing Detection with Internal Gate Telemetry and Attachment OCR

> **Target Audience:** Project group members, external evaluators, viva examiners, and deep learning researchers.  
> **Associated Codebase:** `rnn_lstm_gru_project.py` (standalone benchmark script), `phishguard-ai/` (production-grade model, Streamlit UI, OCR engine, and FastAPI service), and `results/` (experimental plots and metrics).

---

## 1. Project Overview & Meta-Information

| Academic Field | Specification |
| :--- | :--- |
| **Domain** | Cybersecurity / Financial Threat Intelligence & Natural Language Processing (NLP) |
| **Project Topic** | Explainable Deep Recurrent Networks for Long-Sequence Phishing & Financial Fraud Detection with Internal Gate Telemetry and Multi-Modal Document Extraction |
| **Application Name** | **PhishGuard AI** (or *FraudSense XAI*) |
| **Primary DL Architectures** | Vanilla Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU) |
| **Interpretability Paradigm** | Mechanistic Gate Telemetry & Linear Contribution Attribution (XAI) |
| **Input Modalities** | Raw Text Sequences (Email body, SMS, Banking transactions), Document Attachments (PDF invoices, DOCX), and Screenshot Images (Tesseract OCR v5.5) |
| **Output Deliverables** | Binary Threat Classification (`[1] Fraud/Malicious` vs `[0] Legitimate`), Calibrated Risk Score (`0–100`), Token-level Gate Heatmap, and Memory Retention Degradation Probing |

---

## 2. Topic Selection & Problem Motivation

### 2.1 Why Did We Select This Topic?
Traditional classroom deep learning projects frequently rely on trivial tasks such as toy sentiment analysis on movie reviews (e.g., IMDB). While functional for introductory code syntax, movie sentiment does not exhibit high stakes, adversarial dynamics, or realistic long-sequence memory failure modes.

We selected **Financial Fraud, Smishing, and Phishing Sequence Detection** for three compelling real-world reasons:
1. **Critical Real-World Impact:** Social engineering and digital financial fraud cause over **\$10 Billion in annual global losses**. Attacks mimic urgent bank alerts, KYC suspensions, tax refunds, and fake invoice charges.
2. **Adversarial Long-Sequence Evasion (The DL Memory Challenge):** Modern cyber-attackers actively exploit NLP limitations by using **adversarial padding**—burying fraudulent calls-to-action (e.g., phishing links, phone numbers, credential requests) under 100–300 words of polite, benign conversational filler. Standard recurrent architectures suffer from memory decay and forget the initial threat signal by the time the classifier evaluates the final hidden state.
3. **The "Black Box" Trust Barrier in Cybersecurity:** Security Operations Center (SOC) analysts and end-users cannot rely on an unexplainable probability score. If an email is quarantined or flagged as fraud, the analyst must know *exactly which tokens triggered the alarm* and *whether the model retained memory of the initial threat across filler text*.

### 2.2 What is the Purpose of This Project?
1. **Investigate Recurrent Memory & Vanishing Gradients:** Empirically demonstrate how and why Vanilla RNNs fail over long temporal sequences due to the mathematical breakdown of Backpropagation Through Time (BPTT).
2. **Validate Gated Mitigations:** Prove mathematically and experimentally how LSTM and GRU architectures mitigate vanishing gradients—and expose the critical catch that **gates require proper bias conditioning ($b_f = 2.0$)** to prevent early memory collapse.
3. **Build an End-to-End Explainable System:** Move beyond offline Jupyter notebooks by providing a fully functional, local, sub-10ms inference engine that outputs token-by-token gate activity ($f_t, i_t, z_t$) and memory survival metrics.
4. **Counter Evasion via Document & Screenshot OCR:** Attackers frequently place fraud lures inside attached PDF invoices or screenshot images to bypass plain-text email filters. PhishGuard AI integrates Tesseract OCR and PDF parsing to close this security loophole.

---

## 3. Deep Learning Concepts & Mathematical Foundations

### 3.1 Recurrent Sequence Modeling
Unlike feedforward networks that treat inputs independently, a Recurrent Neural Network processes an input sequence $\mathbf{x} = (x_1, x_2, \dots, x_T)$ step-by-step while maintaining an internal recurrent hidden state vector $h_t \in \mathbb{R}^H$:

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

Where:
* $x_t \in \mathbb{R}^D$ is the $D$-dimensional learned word embedding at timestep $t$.
* $W_{xh} \in \mathbb{R}^{H \times D}$ is the input-to-hidden weight matrix.
* $W_{hh} \in \mathbb{R}^{H \times H}$ is the recurrent hidden-to-hidden transition matrix.
* $b_h \in \mathbb{R}^H$ is the bias vector.
* $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ provides non-linear squashing between $[-1, 1]$.

For sequence classification, the final hidden state $h_T$ summarizes the entire history and is projected into a scalar logit $z$:

$$z = w_y^T h_T + b_y, \quad \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

---

### 3.2 Backpropagation Through Time (BPTT) & Derivation of Vanishing Gradients

Let $\mathcal{L}$ be the binary cross-entropy loss computed at the final timestep $T$. During training, the gradient of the loss with respect to the hidden state at an early timestep $t$ ($t \ll T$) is computed via the multivariate chain rule:

$$\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial h_T} \frac{\partial h_T}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}$$

Now, evaluate the single-step Jacobian matrix $\frac{\partial h_k}{\partial h_{k-1}}$:

$$\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}\left(1 - h_k^2\right) W_{hh}^T$$

Substituting this into the chain product yields:

$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \text{diag}\left(1 - h_k^2\right) W_{hh}^T$$

Applying matrix norm bounds:

$$\left\|\frac{\partial h_T}{\partial h_t}\right\| \le \prod_{k=t+1}^{T} \left\|\text{diag}\left(1 - h_k^2\right)\right\| \cdot \left\|W_{hh}^T\right\| \le \left(\gamma \cdot \sigma_{\max}(W_{hh})\right)^{T-t}$$

Where $\gamma = \max_k (1 - h_k^2) \le 1$, and $\sigma_{\max}(W_{hh})$ is the largest singular value (spectral norm) of $W_{hh}$.

#### The Two Failure Modes:
1. **Vanishing Gradient ($\gamma \cdot \sigma_{\max}(W_{hh}) < 1$):** As the temporal distance $(T - t)$ grows beyond 20–50 steps, the gradient decays exponentially to zero ($\approx 10^{-15}$ to underflow at $10^{-38}$). The network weights receive zero learning signal regarding words seen at the beginning of the message.
2. **Exploding Gradient ($\sigma_{\max}(W_{hh}) > 1$):** Gradients grow exponentially, causing numerical overflow (`NaN`). We mitigate this using **Gradient Norm Clipping**:

$$g \leftarrow \begin{cases} g & \text{if } \|g\| \le 1.0 \\ \frac{g}{\|g\|} & \text{if } \|g\| > 1.0 \end{cases}$$

3. **$\tanh$ Saturation Aggravation:** When activations $|h_k| \to 1$, the derivative $(1 - h_k^2) \to 0$. In vanilla RNNs, repeated unconstrained recurrence drives neurons into saturation, immediately killing gradient flow.

---

### 3.3 Long Short-Term Memory (LSTM) Architecture

To solve the vanishing gradient problem, Hochreiter & Schmidhuber (1997) introduced the **Cell State ($c_t$)** acting as an additive, linear memory highway ("constant error carousel") regulated by multiplicative sigmoid gates:

```
          c_{t-1} ───────[ x ]──────────────────────(+)──────── c_t
                          │                          ▲
                          │ f_t (forget)             │ i_t * c~_t
                          │                          │
          h_{t-1} ──┬────[ σ ]   [ σ ]   [tanh]   [ σ ]
                    │      │       │       │        │
          x_t ──────┴──────┴───────┴───────┴────────┴──────── h_t
```

#### The Governing Equations:
1. **Forget Gate ($f_t$):** Controls what fraction of previous cell memory to retain:
   $$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
2. **Input Gate ($i_t$):** Controls what fraction of new candidate memory to write:
   $$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
3. **Candidate Cell State ($\tilde{c}_t$):** Generates new candidate representations:
   $$\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
4. **Cell State Update ($c_t$):** Linear combination of preserved memory and new input:
   $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
5. **Output Gate ($o_t$):** Determines which portion of the cell state to emit into hidden state:
   $$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$
6. **Hidden State ($h_t$):**
   $$h_t = o_t \odot \tanh(c_t)$$

#### Why the LSTM Prevents Vanishing Gradients:
Observe the gradient flow across cell states:

$$\frac{\partial c_t}{\partial c_{t-1}} = \text{diag}(f_t)$$

Notice that **there is no weight matrix multiplication $W$ and no non-linear $\tanh$ derivative** along the $\frac{\partial c_t}{\partial c_{t-1}}$ path! If the forget gate is saturated near $1$ ($f_t \approx 1.0$), the gradient flows backward across hundreds of steps **completely un-attenuated**:

$$\frac{\partial \mathcal{L}}{\partial c_t} = \frac{\partial \mathcal{L}}{\partial c_T} \prod_{k=t+1}^{T} f_k \approx \frac{\partial \mathcal{L}}{\partial c_T} \cdot (1.0)^{T-t} = \frac{\partial \mathcal{L}}{\partial c_T}$$

#### The Forget-Gate Bias Catch ($b_f = 2.0$):
By default in standard neural network initialization, biases are initialized to $0$. At $b_f = 0$, the initial forget gate evaluates to:

$$f_t = \sigma(0) = 0.5$$

Over a 50-token sequence, memory decays by:

$$0.5^{50} \approx 8.88 \times 10^{-16}$$

Thus, **an un-tuned LSTM initially vanishes gradients just as badly as an RNN!** Setting the initial forget gate bias $b_f = 2.0$ ensures $\sigma(2.0) \approx 0.88$, starting the memory highway fully open (Jozefowicz et al., 2015).

---

### 3.4 Gated Recurrent Unit (GRU) Architecture

Proposed by Cho et al. (2014), the GRU simplifies the LSTM by merging the cell state and hidden state into a single recurrent vector $h_t$, utilizing only **two gates**:

#### The Governing Equations (PyTorch Standard Formulation):
1. **Reset Gate ($r_t$):** Determines how much of the past hidden state to ignore when computing candidate state:
   $$r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$$
2. **Update Gate ($z_t$):** Acts simultaneously as forget and input gate:
   $$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$$
3. **Candidate Hidden State ($\tilde{h}_t$):**
   $$\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)$$
4. **Hidden State ($h_t$):**
   $$h_t = (1 - z_t) \odot \tilde{h}_t + z_t \odot h_{t-1}$$

*(Note: In PyTorch, $z_t$ weights the previous hidden state $h_{t-1}$, so $z_t \approx 1$ represents "retain old memory".)*

#### Parameter Count Comparison:
For input dimension $D$ and hidden dimension $H$:
* **Vanilla RNN:** $H(D + H) + 2H \implies \mathbf{1 \times}$ recurrent block.
* **GRU:** $3 \times [H(D + H) + 2H] \implies \mathbf{3 \times}$ gates ($r, z, \tilde{h}$).
* **LSTM:** $4 \times [H(D + H) + 2H] \implies \mathbf{4 \times}$ gates ($i, f, c, o$).

The GRU has **25% fewer recurrent parameters** than the LSTM, converging faster with lower compute overhead while retaining the linear additive update shortcut.

---

### 3.5 Token Left-Padding vs Right-Padding
In recurrent classification, sequence length varies across batches. We pad sequences to a fixed length ($L = 200$ or $400$).
* **Right-Padding (`[tok1, tok2, PAD, PAD, PAD]`):** The network reads tokens early and then traverses hundreds of empty `PAD` tokens before reaching $h_T$. The recurrent state becomes polluted with meaningless null-state transitions.
* **Left-Padding (`[PAD, PAD, PAD, tok1, tok2]`):** The network consumes the meaningless padding first, and reads the active words immediately before emitting the final decision vector $h_T$. This guarantees the classifier operates on fresh, un-diluted contextual activations.

---

### 3.6 Explainable AI (XAI) & Gate Telemetry Attribution
PhishGuard AI does not treat the recurrent network as an opaque black box. We extract mechanistic explanations via internal gate logging:
1. **Write Gate ($i_t$ or $1 - z_t$):** Measures the magnitude of new information the cell accepted when reading word $t$.
2. **Keep Gate ($f_t$ or $z_t$):** Measures the unit-averaged retention gate at step $t$.
3. **Cumulative Survival Metric:**
   $$S_t = \prod_{k=t+1}^{T} f_k$$
   Computes the exact fraction of the gradient/information stored at token $t$ that physically survived to the decision point at step $T$.
4. **Token Logit Contribution:** Approximated through first-order linear attribution:
   $$\Delta z_t = w_y^T (h_t - h_{t-1})$$
   Words that drive the logit toward $+ \infty$ are rendered in crimson (Fraud), while words driving the logit toward $-\infty$ are rendered in blue (Legitimate).

---

## 4. Datasets Used in This Project

Our research utilizes both synthetic benchmarks and real-world multi-corpus financial threat datasets:

```
                             Dataset Hierarchy
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
1. Synthetic Benchmark                               2. Real Phishing Corpus
("First-Symbol Recall")                              (20,052 Unique Messages)
- Isolates gradient breakdown                         │
- Timesteps T ∈ {10, 25, 50, 100}                     ├── KD Phishing & Legit (10,000)
- Chance accuracy: 12.5%                              ├── SMS Spam Collection (5,574)
                                                      └── Archive (Nazario, CEAS, Enron)
```

### 4.1 Synthetic "First-Symbol Recall" Memory Benchmark
* **Purpose:** Cleanly isolate memory degradation without natural language noise.
* **Construction:** At timestep $t=0$, one of $K=8$ symbols is presented on channel 0. For timesteps $t \in [1, T-1]$, random distractor noise is injected on channel 1. At timestep $T$, the network must output which symbol it saw at step 0.
* **Chance Level:** $\frac{1}{8} = 12.5\%$.
* **Evaluated Horizons:** $T \in \{10, 25, 50, 100\}$.

### 4.2 Unified Multi-Source Financial Threat Corpus (`combined_phishing_dataset_v2.csv`)
* **Total Volume:** **23,052 unique real-world samples** (54.1% legitimate / 45.9% malicious).
* **Sub-Corpora Integrated:**
  1. `data/phishing_legit_dataset_KD_10000.csv`: 10,000 modern email samples (6,000 phishing lures, 4,000 legitimate enterprise emails).
  2. `data/sms_spam.tsv` / UCI `SMSSpamCollection`: 5,574 SMS and smishing messages containing financial prize scams, shortlinks, and urgent bank alerts.
  3. `data/archive/`: Curated cybersecurity corpora including Nazario Phishing, CEAS 2008, SpamAssassin, Enron Financial, and Nigerian Advance-Fee Fraud.
  4. `data/hard_examples.csv`: 3,000 contrastive hard examples (dispute tickets, legitimate card freeze notices, bank dispute resolutions) sharing high-risk fraud vocabulary (`unauthorized`, `fraudulent`, `account`, `stolen`) to eliminate false positives.
* **Preprocessing & Tokenization Pipeline:**
  * Alphanumeric tokenization preserving URL protocols (`http`, `https`), domain structures, and currency symbols (`$`, `₹`, `£`).
  * Normalization: URLs mapped to `<url>`, emails to `<email>`, numbers to `<numK>` (where $K$ is digit count, separating OTPs, phone numbers, and amounts).
  * Vocabulary construction with frequency clipping: **22,000+ unique tokens**.
* **Dataset Splits (Stratified 80 / 10 / 10):**
  * **Training Set (80%):** **18,441 samples** (used for gradient updates and backpropagation).
  * **Validation Set (10%):** **2,304 samples** (used for early stopping patience and temperature scaling calibration).
  * **Test Set (10% Holdout):** **2,307 samples** (unseen evaluation data used for final performance reporting).

### 4.3 Dataset Consistency Across Models
* **Does every model have the exact same dataset split?**
  **Yes, 100% identical.** Both LSTM and GRU training in `train.py` use a fixed global random seed (`seed = 42`). The permutation and stratified partition indices are identical. Every model trains on the exact same 18,441 samples, validates on the exact same 2,304 samples, and is evaluated on the exact same 2,307 holdout test samples. This guarantees scientific rigor and eliminates data bias in comparative benchmarks.
* **Is every model doing the same thing?**
  **Yes.** Both LSTM and GRU perform the exact same task: binary threat classification (`[1] Fraud/Malicious` vs `[0] Legitimate`), emitting a calibrated risk score (`0–100`), and providing token-level gate telemetry. The only difference is their internal recurrent cell mathematics:
  * **LSTM:** Uses 3 multiplicative gates ($f_t, i_t, o_t$) and a separate additive **Cell State ($c_t$)**.
  * **GRU:** Uses 2 multiplicative gates ($r_t, z_t$) and merges the cell state into the recurrent hidden state ($h_t$).

---

## 5. Models Evaluated & Architecture Details

All models were evaluated under identical training conditions:
* **Embedding Layer:** $22,000+ \to 128$ dimensions (learned from scratch, no pre-trained weights).
* **Hidden State Size:** $128$ recurrent units.
* **Optimizer:** AdamW ($\alpha = 0.002$, weight decay $= 10^{-2}$).
* **Batch Size:** 64.
* **Regularization:** Dropout $p = 0.3$, Word Dropout $= 0.05$, Gradient Clipping $\|g\|_2 \le 1.0$.
* **Calibration:** Temperature Scaling ($T_{\text{temp}}$) optimized via Negative Log-Likelihood on validation sets.

### 5.1 The Five Model Variants

| Variant Name | Cell Type | Gate-Bias Conditioning | Theoretical Memory Horizon |
| :--- | :--- | :--- | :--- |
| **`RNN`** | Vanilla Elman RNN | None (Standard) | Collapses at $T \approx 25–50$ steps |
| **`LSTM (default)`** | LSTM | Default ($b_f = 0 \implies f_t \approx 0.5$) | Fails early due to initial memory halving |
| **`LSTM (f-bias=2)`**| LSTM | **$b_f = 2.0 \implies f_t \approx 0.88$** | Survives up to $T = 50–100$ steps |
| **`GRU (default)`** | GRU | Default ($b_z = 0$) | Survives up to $T \approx 25$ steps |
| **`GRU (z-bias=2)`** | GRU | **$b_z = 2.0 \implies z_t \approx 0.88$** | **Survives fully up to $T = 100+$ steps** |

---

## 6. Experimental Results & Key Findings

### 6.1 Synthetic Memory Recall Benchmark (Exp. 1)

Accuracy across sequence horizons ($T$):

| Architecture | $T = 10$ | $T = 25$ | $T = 50$ | $T = 100$ |
| :--- | :---: | :---: | :---: | :---: |
| **Vanilla RNN** | **100%** | **100%** | 12.5% (Fail) | 12.8% (Fail) |
| **LSTM (default bias)** | **100%** | 12.3% (Fail) | 12.5% (Fail) | 12.5% (Fail) |
| **LSTM ($b_f = 2.0$)** | **100%** | **100%** | **94.0%** | 13.0% (Fail) |
| **GRU (default bias)** | **100%** | **100%** | 12.1% (Fail) | 12.2% (Fail) |
| **GRU ($z\text{-bias} = 2.0$)**| **100%** | **100%** | **100%** | **100%** |

*Takeaway:* Default LSTM and GRU fail catastrophic long-term recall. Setting the gate bias to $2.0$ completely restores the memory highway up to 100 timesteps.

### 6.2 Empirical Gradient Measurements (Exp. 2)
Measuring the backpropagated gradient norm $\|\partial \mathcal{L} / \partial h_1\|$ relative to step $T$:
* **Untrained RNN at $T=100$:** Gradient norm dropped by $18$ orders of magnitude (underflow below $10^{-38}$).
* **Trained RNN at $T=50$:** Gradient relative ratio $= 1.5 \times 10^{-11}$.
* **Trained LSTM ($b_f = 2.0$) at $T=50$:** Relative ratio $= 3.2 \times 10^{-2}$.
* **Trained GRU ($b_z = 2.0$) at $T=50$:** Relative ratio $= \mathbf{0.82}$ (Virtually flat; gradient traveled 50 steps with almost zero loss).

### 6.3 Real-World Phishing & Fraud Classification Performance

Evaluated on **2,307 unseen holdout test samples** (`combined_phishing_dataset_v2.csv`):

| Architecture | Precision | Recall | F1-Score | ROC-AUC | Retention @ 300 Filler Words |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vanilla RNN** | 82.4% | 71.2% | 76.4% | 0.841 | 42.1% (Catastrophic forgetting) |
| **LSTM (Retrained MPS)** | **99.7%** | **99.4%** | **99.6%** | **0.9994** | **98.9%** (Robust long-horizon memory) |
| **GRU (Retrained MPS)** | **99.2%** | **99.6%** | **99.4%** | **0.9996** | **99.5%** (Near-perfect retention) |

---

## 7. Existing Systems & How PhishGuard AI Compares

| Security Dimension | Existing Approach 1: Regex & Keyword Blacklists | Existing Approach 2: Protocol Headers (SPF/DKIM/DMARC) | Existing Approach 3: Cloud LLMs (GPT-4 / Claude) | **Our System: PhishGuard AI (Gated RNNs + OCR)** |
| :--- | :--- | :--- | :--- | :--- |
| **Detection Principle** | Exact string matching (`"account suspended"`) | DNS cryptographic sender verification | Massive Transformer auto-regression | Recurrent gate telemetry + sequence modeling |
| **Weakness** | Trivially bypassed by synonyms, typos, leetspeak | Blind to lookalike domains (`hdfc-kyc.com`) or compromised accounts | High cost, 1–2 sec latency, privacy risks sending data to 3rd party | Requires periodic retraining on novel cyber-slang |
| **Latency** | $< 1$ ms | $< 5$ ms | $500–2,000$ ms | **$< 5$ ms (Ultra-fast, on-device)** |
| **Inference Cost** | Negligible | Negligible | High (\$ per token) | **Zero (Runs locally on CPU/Edge)** |
| **Explainability** | High (matched rule) | Binary Pass/Fail | Opaque attention / Hallucinations | **Exact token-level Gate Telemetry & XAI Heatmap** |
| **Multi-Modal OCR**| None | None | Additional cost / vision APIs | **Built-in Tesseract OCR & PDF/DOCX Parser** |

---

## 8. Limitations & Areas for Future Improvement

When questioned by examiners, demonstrate high academic maturity by proactively stating project limitations and future enhancements:

1. **Unidirectional vs. Bidirectional Context:**
   * *Current Limitation:* Our current RNN/LSTM/GRU processes text left-to-right. A threat word at the beginning only conditions future tokens, not past tokens.
   * *Improvement:* Implement **Bidirectional LSTMs (Bi-LSTM)** so every token receives contextual representations from both past and future directions.
2. **Hierarchical Attention Networks (HAN):**
   * *Current Limitation:* Words are flattened into a single sequence up to 400 tokens.
   * *Improvement:* Group tokens into sentences, run word-level recurrence, and aggregate sentence embeddings via an Attention mechanism (Hierarchical Recurrent Attention).
3. **Multi-Modal Vision-Language Joint Embeddings:**
   * *Current Limitation:* Ingestion is sequential—OCR extracts text first, which is then fed into the text RNN. Visual layout information (e.g., logo placement, fake QR codes) is lost.
   * *Improvement:* Implement a joint visual-text model (e.g., LayoutLM or CLIP) that fuses layout coordinates with textual representations.
4. **Adversarial Robustness against LLM Phishing:**
   * *Current Limitation:* Attackers now use local LLMs to generate polymorphic phishing emails with zero spelling mistakes and natural variance.
   * *Improvement:* Adversarial training using GAN-based text perturbations and automated prompt injection stress-tests.

---

## 9. Comprehensive Viva Voce Q&A (Examiner Prep)

#### Q1: What is the Vanishing Gradient Problem, and what causes it mathematically?
> **Answer:** When training recurrent networks using Backpropagation Through Time (BPTT), the gradient of the loss at step $T$ with respect to the hidden state at step $t$ involves repeated multiplication of the Jacobian matrix $\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}(1 - h_k^2) W_{hh}^T$. If the spectral norm of $W_{hh}$ and the tanh derivative are less than 1, the gradient decays exponentially as $(T - t)$ increases. Early timesteps receive virtually zero gradient updates, rendering the network unable to learn long-range temporal dependencies.

#### Q2: How does the LSTM architecture solve vanishing gradients?
> **Answer:** The LSTM introduces a separate **Cell State ($c_t$)** that updates additively rather than through matrix multiplication: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. The derivative $\frac{\partial c_t}{\partial c_{t-1}} = f_t$ contains **no weight matrix $W$ and no saturating $\tanh$ derivative**. When the forget gate $f_t \approx 1$, gradients flow backward across time without exponential attenuation. This path is known as the *Constant Error Carousel*.

#### Q3: What is the "Forget-Gate Bias Initialisation Catch"?
> **Answer:** By default, neural network biases are initialized to 0. Since $\sigma(0) = 0.5$, an LSTM starts with forget gates that halve memory at every step ($0.5^{50} \approx 10^{-15}$). Consequently, an untrained default LSTM vanishes gradients just like an RNN! Setting the initial forget bias $b_f = 2.0$ ensures $\sigma(2) \approx 0.88$, keeping the memory highway open from the start of training (Jozefowicz et al., 2015).

#### Q4: Why did we test our models with 300 filler words?
> **Answer:** In cyber-fraud, attackers employ *adversarial padding*—embedding an initial threat signal (e.g., *"HDFC account suspended"*) followed by hundreds of words of polite chatter to evade spam filters. Our filler stress test proves that our calibrated LSTM and GRU maintain $>98\%$ recall even when 300 filler words separate the threat from the classification readout, whereas Vanilla RNN accuracy drops to chance (~42%).

#### Q5: Why use Recurrent Networks (LSTM/GRU) instead of Transformers or LLMs?
> **Answer:** 
> 1. **Resource Efficiency:** Our model has $\sim 100\text{K}$ parameters and runs sub-5ms inference on standard CPUs, making it ideal for edge gateways, mobile devices, and local firewalls.
> 2. **Mechanistic Explainability:** Recurrent gates ($f_t, i_t, z_t$) provide direct, deterministic telemetry into information storage and retention. In contrast, Transformer self-attention maps are quadratically expensive and attention weights do not represent faithful causal explanations.
> 3. **Privacy:** PhishGuard AI operates 100% locally with zero cloud API dependencies, ensuring confidential emails and banking alerts are never exposed to third-party servers.

#### Q6: How does PhishGuard AI detect phishing inside attachments?
> **Answer:** Attackers frequently place phishing lures inside PDF invoices or screenshot images to bypass text-based NLP filters. We built an automated ingestion layer using **Tesseract OCR (v5.5)** for image screenshots (`.png`, `.jpg`) and **PyMuPDF (`fitz`)** for PDF documents. The extracted text is normalized and piped into the recurrent inference service, achieving seamless cross-modal threat detection.

---

## 10. File Architecture & Codebase Map (What Each File Does)

| File / Component | Architectural Role | Core Responsibility |
| :--- | :--- | :--- |
| [`STUDY_GUIDE.md`](file:///Users/jyotirmoy/Desktop/test/STUDY_GUIDE.md) | Academic Dossier | Complete viva preparation guide, mathematical derivations, architecture proofs, and project defense notes. |
| [`rnn_lstm_gru_project.py`](file:///Users/jyotirmoy/Desktop/test/rnn_lstm_gru_project.py) | Scientific Benchmark Script | Standalone research benchmark; mathematically isolates vanishing gradients, unrolls BPTT Jacobian products, measures tanh neuron saturation, and benchmarks synthetic recall across $T \in \{10, 25, 50, 100\}$. |
| `data/combined_phishing_dataset_v2.csv` | Unified Training Corpus | 23,052 balanced samples combining SMS spam, Enron/Nazario emails, and contrastive hard dispute pairs. |
| `data/hard_examples.csv` | Hard-Negative Pair Corpus | 3,000 contrastive dispute tickets, stolen card reports, and legitimate bank notices sharing fraud keywords. |
| [`phishguard-ai/phishguard/config.py`](file:///Users/jyotirmoy/Desktop/test/phishguard-ai/phishguard/config.py) | Model Hyperparameters | Defines `ModelConfig` dataclass (vocab size, embedding dim 128, hidden units 128, sequence max length 400). |
| [`phishguard-ai/phishguard/text.py`](file:///Users/jyotirmoy/Desktop/test/phishguard-ai/phishguard/text.py) | Preprocessing & Tokenizer | Regex alphanumeric tokenizer, entity normalizer (URLs $\to$ `<url>`, emails $\to$ `<email>`, numbers $\to$ `<numK>`), and frequency-clipped vocabulary builder. |
| [`phishguard-ai/phishguard/model.py`](file:///Users/jyotirmoy/Desktop/test/phishguard-ai/phishguard/model.py) | Recurrent Architecture | `GateRNN` wrapping PyTorch LSTM and GRU cells with a custom unrolled `.trace()` method for token-level gate telemetry. |
| [`phishguard-ai/phishguard/explain.py`](file:///Users/jyotirmoy/Desktop/test/phishguard-ai/phishguard/explain.py) | Mechanistic Interpretability | Computes first-order logit attributions $\Delta z_t$ and cumulative memory survival metrics $\prod f_t$. |
| [`phishguard-ai/phishguard/signals.py`](file:///Users/jyotirmoy/Desktop/test/phishguard-ai/phishguard/signals.py) | Attack-Vector Verifier | Rule-verification layer detecting physical exploitation vectors (links, credentials, deadline threats) vs. customer reporting voice. |
| [`phishguard-ai/phishguard/extract.py`](file:///Users/jyotirmoy/Desktop/test/phishguard-ai/phishguard/extract.py) | Document & OCR Ingestion | Extracts text from screenshots (`.png`, `.jpg`) using Tesseract OCR v5.5 and parses `.pdf` / `.docx` invoice attachments via PyMuPDF. |
| [`phishguard-ai/phishguard/service.py`](file:///Users/jyotirmoy/Desktop/test/phishguard-ai/phishguard/service.py) | Inference Service | Top-level coordinator that accepts raw text or documents, unrolls gate telemetry, and formats predictions. |
| [`phishguard-ai/phishguard/train.py`](file:///Users/jyotirmoy/Desktop/test/phishguard-ai/phishguard/train.py) | Training Pipeline | Automatic Apple Silicon M4 GPU (`MPS`) training pipeline, AdamW optimization, adversarial filler augmentation, and calibration. |
| [`phishguard-ai/phishguard/api.py`](file:///Users/jyotirmoy/Desktop/test/phishguard-ai/phishguard/api.py) | Headless REST API | FastAPI web service with endpoints `/healthz`, `/v1/model`, `/v1/analyze`, and `/v1/analyze/file`. |
| [`phishguard-ai/dashboard/app.py`](file:///Users/jyotirmoy/Desktop/test/phishguard-ai/dashboard/app.py) | Streamlit Dashboard | Interactive multi-tab web user interface for real-time inference, OCR drag-and-drop, gate heatmaps, and adversarial stress tests. |

---

## 11. Streamlit Dashboard Architecture & Visual Features Guide

The **PhishGuard AI** Streamlit dashboard (`dashboard/app.py`) is structured into distinct interactive modules:

### 11.1 The Persistent Sidebar (Controls & Telemetry)
* **Model Dropdown:** Allows real-time switching between trained model checkpoints (`artifacts/lstm/model.pt` vs. `artifacts/gru/model.pt`). Automatically detects files on disk and refreshes when model weights are updated.
* **Architecture Specs:** Shows the loaded recurrent cell (`LSTM` or `GRU`), hidden unit dimension ($128$), and sequence horizon ($400$ tokens).
* **Calibrated Decision Line:** Displays the optimal threshold (e.g., $34/100$ for LSTM or $46/100$ for GRU) derived from validation PR-curve tuning.
* **Message Type Selector:** Radio toggle allowing analysts to classify the incoming modality as `SMS`, `Email`, or `Banking Transaction`.

### 11.2 Tab 1: "Check a Message" (Live Inference & XAI Gate Telemetry)
* **Pre-Loaded Threat Archetypes:** 5 quick-load buttons (`Bank SMS`, `Prize SMS`, `Phishing email`, `Transaction`, `Friendly note`) to demonstrate instant threat reactions.
* **Attachment & Screenshot OCR Expander (`📎`):**
  * Drag-and-drop file uploader supporting `.pdf`, `.docx`, `.png`, `.jpg`, `.jpeg`, `.eml`.
  * Automatically routes images to Tesseract OCR and documents to PyMuPDF, providing a *"Load into text box"* button for immediate gate analysis.
* **Verdict & Risk Cards:**
  * **Threat Flag Banner:** Bright Crimson `Fraud / Malicious [1]` or Emerald Blue `Legitimate [0]`.
  * **Risk Score Metric:** Calibrated percentage score (`0–100 / 100`) accompanied by a dynamic progress bar and Risk Band (`Low`, `Medium`, `High`, `Critical`).
* **Attack-Vector Verification Panel:**
  * Explicitly highlights whether concrete malicious vectors were found (e.g., `Link to an external site`, `Asks for password / OTP`, `Threat with a deadline`).
  * Displays account holder indicators (*"Written by the account holder"*, *"Safe bank wording"*).
* **Token-Level Gate Telemetry Heatmap:**
  * Visualizes the exact word sequence with glowing colored token pills:
    * **Crimson Red:** Tokens pushing strongly toward Fraud (e.g., `unauthorized`, `suspended`, `verify`, `click`).
    * **Blue:** Tokens pushing toward Legitimate.
    * **Teal Intensity:** Shows internal gate activation strength ($f_t, i_t, z_t$).
  * **Hover Tooltip:** Hovering over any word reveals its exact numeric contribution, input write gate, keep gate, and survival probability.
* **Dual Altair Diagnostic Charts:**
  * **Left Chart (Token Evidence Bar Chart):** Plots token position vs. directional logit evidence ($\Delta z_t$). Red bars push the probability toward fraud; blue bars pull it back toward legitimate.
  * **Right Chart (Gate Dynamics Line Chart):** Traces three temporal curves across the message:
    * **Yellow Line (`write`):** Input gate activity ($i_t$), showing where new information was written.
    * **Blue Line (`keep`):** Retention gate ($f_t$ or $z_t$), showing how much memory was preserved.
    * **Teal Curve (`survival`):** Cumulative survival metric $\prod f_k$, proving how much of an opening token's memory physically reached the classification step.

### 11.3 Tab 2: "Memory Stress Test" (Adversarial Degradation Probe)
* Takes any suspicious email or SMS and allows the user to append between **50 and 350 filler words** of harmless conversational text after the message.
* Evaluates whether the recurrent gates maintain the threat flag across long distances or suffer from memory decay.

### 11.4 Tab 3: "Scan a Batch" (SOC Ingestion)
* Ingests CSV or text files containing up to 100 messages at once.
* Employs chunked GPU batch inference and produces an exportable tabular report with risk scores and detected cues.

### 11.5 Tab 4: "Model Report" (Validation Metrics)
* Automatically parses `metrics.json` and renders holdout test ROC-AUC, PR-AUC, Precision, Recall, and the retention recall decay table across 0 to 300 filler words.

---

## 12. Execution Quick-Reference

### Standalone Benchmark & Verification Script:
```bash
# Run full empirical experiment suite (Outputs to results/)
python rnn_lstm_gru_project.py

# Run quick 2-minute smoke test
QUICK=1 python rnn_lstm_gru_project.py

# Launch CLI Interactive Mode
INTERACTIVE=1 python rnn_lstm_gru_project.py
```

### Full Interactive Web Dashboard & Attachment Scanner:
```bash
cd phishguard-ai
streamlit run dashboard/app.py
```

### Headless REST API Server:
```bash
cd phishguard-ai
uvicorn phishguard.api:app --port 8000 --reload
# Access docs at: http://localhost:8000/docs
```
