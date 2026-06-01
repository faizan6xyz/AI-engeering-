"""
================================================================================
        GENERATIVE AI - ALL TYPES OF NETWORKS: WORKING, USE CASES,
                        ADVANTAGES & FLAWS
================================================================================

Author : Educational Reference
Topic  : Generative AI Networks
Coverage:
    1.  Generative Adversarial Networks (GANs)
    2.  Variational Autoencoders (VAEs)
    3.  Transformers
    4.  Diffusion Models
    5.  Recurrent Neural Networks (RNNs) & LSTMs
    6.  Flow-Based Models (Normalizing Flows)
    7.  Autoregressive Models
    8.  Energy-Based Models (EBMs)
    9.  Boltzmann Machines / Restricted Boltzmann Machines (RBMs)
    10. Hybrid / Multimodal Models
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# QUICK-REFERENCE SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
NETWORK_SUMMARY = {
    "GAN":              "Adversarial Generator vs Discriminator training",
    "VAE":              "Encode to latent distribution, sample & decode",
    "Transformer":      "Attention-based sequence/token modelling",
    "Diffusion":        "Iterative denoising from Gaussian noise",
    "RNN/LSTM":         "Sequential hidden-state recurrence",
    "Flow-Based":       "Invertible transformations for exact likelihoods",
    "Autoregressive":   "Token-by-token conditional probability chain",
    "EBM":              "Energy landscape; low energy ≈ real data",
    "RBM":              "Two-layer stochastic units; contrastive divergence",
    "Hybrid/Multimodal":"Multiple modalities / architectures combined",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATIVE ADVERSARIAL NETWORKS (GANs)
# ─────────────────────────────────────────────────────────────────────────────
class GAN:
    """
    GENERATIVE ADVERSARIAL NETWORK (GAN)
    ─────────────────────────────────────
    Introduced by Ian Goodfellow et al. in 2014.

    ARCHITECTURE
    ─────────────
    Two competing neural networks:
      • Generator (G)     – Learns to produce fake data from random noise (z).
      • Discriminator (D) – Learns to distinguish real data from G's fakes.

    WORKING (TRAINING LOOP)
    ────────────────────────
    Step 1 : Sample random noise z ~ N(0, I).
    Step 2 : G(z) → fake sample.
    Step 3 : D receives real samples AND fake samples; outputs probability of
             being real.
    Step 4 : D is updated to maximize log D(x) + log(1 − D(G(z))).
    Step 5 : G is updated to minimize log(1 − D(G(z)))  →  maximize log D(G(z)).
    Step 6 : Repeat until Nash Equilibrium (D cannot distinguish real/fake).

    Objective (minimax game):
        min_G  max_D  E[log D(x)] + E[log(1 − D(G(z)))]
    """

    VARIANTS = {
        "DCGAN":     "Deep Convolutional GAN – uses Conv layers for images",
        "CycleGAN":  "Image-to-image translation without paired data",
        "StyleGAN":  "Style-based generator; controls hair, age, etc.",
        "Pix2Pix":   "Paired image-to-image translation",
        "WGAN":      "Wasserstein distance; more stable training",
        "BigGAN":    "Large-scale class-conditional image generation",
        "ProgressiveGAN": "Grows resolution progressively during training",
    }

    USE_CASES = [
        "Photo-realistic face / image synthesis (e.g., thispersondoesnotexist.com)",
        "Data augmentation for medical imaging (MRI, X-rays)",
        "Video game asset / texture generation",
        "Super-resolution (upscaling low-res images)",
        "Drug molecule design in computational chemistry",
        "Deepfake generation (and detection research)",
        "Fashion / product image synthesis for e-commerce",
        "Art & style transfer",
    ]

    ADVANTAGES = [
        "Produces extremely high-quality, sharp outputs",
        "No explicit likelihood computation needed",
        "Highly flexible — can be conditioned on labels, images, text",
        "Efficient sampling (single forward pass after training)",
    ]

    FLAWS = [
        "Training instability — discriminator or generator may dominate",
        "Mode collapse — generator produces limited variety",
        "Vanishing gradients in the original GAN formulation",
        "Evaluation metrics (FID, IS) are imperfect proxies",
        "Requires careful hyperparameter tuning",
        "No explicit latent space structure (unlike VAEs)",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. VARIATIONAL AUTOENCODERS (VAEs)
# ─────────────────────────────────────────────────────────────────────────────
class VAE:
    """
    VARIATIONAL AUTOENCODER (VAE)
    ──────────────────────────────
    Introduced by Kingma & Welling in 2013.

    ARCHITECTURE
    ─────────────
      • Encoder  q_φ(z|x) – Maps input x → latent distribution (μ, σ²).
      • Latent Space        – Samples z ~ N(μ, σ²) via reparameterisation trick.
      • Decoder  p_θ(x|z) – Maps z back to reconstructed x̂.

    WORKING
    ────────
    Step 1 : Encoder maps x to parameters (μ, log σ²) of a Gaussian.
    Step 2 : Reparameterisation: z = μ + σ · ε,  ε ~ N(0,I)
             (allows gradient flow through sampling).
    Step 3 : Decoder reconstructs x̂ from z.
    Step 4 : Loss = Reconstruction Loss + KL Divergence
             ELBO: E[log p(x|z)] − KL(q(z|x) || p(z))
             • Reconstruction loss penalizes pixel/token differences.
             • KL term regularises latent space to stay near N(0,I).

    To GENERATE: sample z ~ N(0,I), pass through decoder → new sample.
    """

    VARIANTS = {
        "β-VAE":         "Stronger KL weight for disentangled representations",
        "VQ-VAE":        "Discrete latent codes; used in DALL-E v1, MusicLM",
        "Conditional VAE":"Conditioned on class labels for controlled generation",
        "Hierarchical VAE":"Multiple layers of latent variables",
    }

    USE_CASES = [
        "Image generation and manipulation",
        "Anomaly detection (high reconstruction loss ≈ anomaly)",
        "Drug / molecule discovery (latent space interpolation)",
        "Representation learning for downstream tasks",
        "Text generation (sentence interpolation in latent space)",
        "Music generation",
        "Handwriting synthesis",
    ]

    ADVANTAGES = [
        "Principled probabilistic framework with tractable likelihood bound",
        "Smooth, structured latent space (easy interpolation & editing)",
        "Stable training compared to GANs",
        "Disentangled representations (β-VAE variant)",
        "Explicit density estimation via ELBO",
    ]

    FLAWS = [
        "Generated images tend to be blurry (due to pixel-level MSE loss)",
        "ELBO is only a lower bound — true likelihood may be higher",
        "Posterior collapse: decoder ignores z if it's too powerful",
        "Weaker sample quality than GANs or Diffusion models",
        "KL vs reconstruction trade-off is hard to balance",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRANSFORMERS
# ─────────────────────────────────────────────────────────────────────────────
class Transformer:
    """
    TRANSFORMER
    ────────────
    Introduced by Vaswani et al. ("Attention Is All You Need") in 2017.
    Foundation of GPT, BERT, T5, DALL-E, Stable Diffusion UNet, etc.

    ARCHITECTURE
    ─────────────
      • Tokenizer             – Converts input (text/image) to tokens/patches.
      • Token Embeddings + Positional Encoding
      • N × Transformer Blocks:
          ◦ Multi-Head Self-Attention (MHSA)
          ◦ Feed-Forward Network (FFN)
          ◦ Layer Normalisation & Residual Connections
      • Output Head (LM head, classifier, etc.)

    MULTI-HEAD SELF-ATTENTION (core mechanism)
    ───────────────────────────────────────────
    For each head h:
        Q = X·W_Q,   K = X·W_K,   V = X·W_V
        Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V
    Heads are concatenated and projected.

    GENERATION (autoregressive decoding):
        At each step, predict the next token using all previous tokens.
        Sampling strategies: Greedy, Beam Search, Top-k, Top-p (nucleus).

    KEY VARIANTS
    ─────────────
      Encoder-only   : BERT, RoBERTa  (classification, embeddings)
      Decoder-only   : GPT series, LLaMA, Mistral  (text generation)
      Encoder-Decoder: T5, BART, mT5  (translation, summarisation)
      Vision         : ViT (Vision Transformer), Swin Transformer
      Cross-modal    : CLIP, DALL-E, Flamingo
    """

    NOTABLE_MODELS = {
        "GPT-4":         "Decoder-only LLM; instruction-following, coding, reasoning",
        "Claude 3 / 4":  "Anthropic's constitutional AI assistant",
        "Gemini":        "Google DeepMind multimodal model",
        "LLaMA 3":       "Meta's open-source LLM",
        "T5":            "Text-to-text encoder-decoder by Google",
        "DALL-E 3":      "Text-to-image via Transformer + Diffusion",
        "Sora":          "OpenAI video generation using spatial-temporal Transformer",
        "Whisper":       "Speech-to-text encoder-decoder Transformer",
    }

    USE_CASES = [
        "Large Language Models (ChatGPT, Claude, Gemini)",
        "Code generation (GitHub Copilot, Claude Code)",
        "Machine translation (DeepL, Google Translate)",
        "Text summarisation and question answering",
        "Text-to-image generation (DALL-E, Midjourney backbone)",
        "Speech recognition and synthesis",
        "Protein structure prediction (AlphaFold 2)",
        "Document understanding and information extraction",
    ]

    ADVANTAGES = [
        "Parallelisable training (unlike RNNs) — scales to billions of params",
        "Long-range dependency capture via attention",
        "Transfer learning: pretrain once, fine-tune for many tasks",
        "Handles text, image, audio, video uniformly via tokenisation",
        "SOTA performance across virtually all NLP / vision benchmarks",
    ]

    FLAWS = [
        "Quadratic attention complexity O(n²) for long sequences",
        "Requires massive data and compute for pretraining",
        "Hallucinations — confidently generates false information",
        "No explicit memory — context window is a hard limit",
        "Black-box reasoning; limited interpretability",
        "Positional encoding struggles with very long documents",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 4. DIFFUSION MODELS
# ─────────────────────────────────────────────────────────────────────────────
class DiffusionModel:
    """
    DIFFUSION MODELS (DDPMs / Score-Based Models)
    ───────────────────────────────────────────────
    Key papers: Ho et al. 2020 (DDPM), Song et al. 2020 (Score Matching),
                Rombach et al. 2022 (Latent Diffusion / Stable Diffusion).

    WORKING
    ────────
    Two Markov chain processes:

    FORWARD (diffusion) process  q:
        Gradually adds Gaussian noise over T steps.
        x_t = √(ᾱ_t)·x_0 + √(1−ᾱ_t)·ε,   ε ~ N(0,I)
        At t=T: x_T ≈ pure Gaussian noise.

    REVERSE (denoising) process  p_θ:
        A neural network ε_θ(x_t, t) is trained to predict
        the noise ε added at each step.
        During generation, start from x_T ~ N(0,I) and
        iteratively denoise using the learned reverse process.

    TRAINING OBJECTIVE (simplified):
        L = E[||ε − ε_θ(x_t, t)||²]

    LATENT DIFFUSION (Stable Diffusion trick):
        Run diffusion in a compressed VAE latent space (not pixel space)
        → ×4-8× cheaper; enables high-resolution synthesis.

    CONDITIONING:
        Text → CLIP / T5 text encoder → cross-attention in UNet/Transformer
        Enables text-to-image, image-to-image, inpainting, etc.
    """

    VARIANTS = {
        "DDPM":             "Denoising Diffusion Probabilistic Model (original)",
        "DDIM":             "Deterministic sampling; ×10-50 fewer steps",
        "Stable Diffusion": "Latent Diffusion; open-source text-to-image",
        "DALL-E 2/3":       "OpenAI's text-to-image diffusion",
        "Imagen":           "Google's cascaded diffusion model",
        "AudioDiffusion":   "Waveform / spectrogram audio generation",
        "Video Diffusion":  "Temporal extension for video frames",
        "DreamBooth":       "Fine-tune on 3-5 personal images; personalisation",
    }

    USE_CASES = [
        "Text-to-image generation (Midjourney, Adobe Firefly, Stable Diffusion)",
        "Image inpainting and outpainting",
        "Image super-resolution",
        "3D shape and scene generation",
        "Audio / music synthesis (AudioLDM, MusicGen via diffusion)",
        "Video generation (Sora, Gen-2, Stable Video Diffusion)",
        "Drug / molecule design",
        "Medical image synthesis (CT, MRI augmentation)",
    ]

    ADVANTAGES = [
        "State-of-the-art image quality (rivals/surpasses GANs)",
        "Stable training — no adversarial dynamics",
        "Flexible conditioning: text, image, depth map, segmentation mask",
        "Exact likelihood computation (continuous-time variants)",
        "Naturally supports inpainting, editing, interpolation",
    ]

    FLAWS = [
        "Slow sampling — hundreds of NFE (network function evaluations)",
        "High memory and compute requirements",
        "DDIM / distillation tricks help but quality/speed trade-off remains",
        "Difficult fine-grained compositional control",
        "Risk of generating harmful / deepfake content",
        "Prompt engineering needed for consistent results",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 5. RECURRENT NEURAL NETWORKS (RNNs) & LSTMs / GRUs
# ─────────────────────────────────────────────────────────────────────────────
class RNN_LSTM:
    """
    RECURRENT NEURAL NETWORKS (RNNs), LSTMs, GRUs
    ────────────────────────────────────────────────
    Pre-Transformer sequential architectures.

    VANILLA RNN
    ────────────
    h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
    y_t = W_y · h_t + b_y
    Hidden state h_t carries memory of past inputs.

    LONG SHORT-TERM MEMORY (LSTM)  [Hochreiter & Schmidhuber, 1997]
    ─────────────────────────────
    Adds a cell state C_t and three gates to solve vanishing gradient:
      • Forget gate  f_t  = σ(W_f·[h_{t-1}, x_t] + b_f)
      • Input gate   i_t  = σ(W_i·[h_{t-1}, x_t] + b_i)
      • Output gate  o_t  = σ(W_o·[h_{t-1}, x_t] + b_o)
      Cell update: C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(W_C·[h_{t-1},x_t]+b_C)
      Hidden:      h_t = o_t ⊙ tanh(C_t)

    GRU (Gated Recurrent Unit):
    Simplified LSTM with 2 gates (reset, update); fewer parameters.

    GENERATION:
    Seed the network with a start token; feed each output as the next input
    (autoregressive loop) until EOS or max length.
    """

    USE_CASES = [
        "Early text generation (char-RNN, word-RNN)",
        "Time-series forecasting (stock prices, sensor data)",
        "Speech recognition (pre-Transformer era)",
        "Machine translation (seq2seq with attention, pre-Transformer)",
        "Music generation (e.g., Magenta project)",
        "Handwriting generation",
        "Sentiment analysis on sequential data",
    ]

    ADVANTAGES = [
        "Natural fit for sequential / temporal data",
        "Parameter-efficient for short sequences",
        "Can operate on variable-length inputs and outputs",
        "LSTM/GRU mitigate vanishing gradient vs vanilla RNN",
    ]

    FLAWS = [
        "Sequential computation — cannot be parallelised over time steps",
        "Struggles with very long dependencies despite LSTM gates",
        "Largely superseded by Transformers for NLP tasks",
        "Vanishing / exploding gradients in very deep unrolled graphs",
        "Hidden state is a fixed-size bottleneck",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 6. FLOW-BASED MODELS (NORMALIZING FLOWS)
# ─────────────────────────────────────────────────────────────────────────────
class NormalizingFlow:
    """
    NORMALIZING FLOWS / FLOW-BASED GENERATIVE MODELS
    ──────────────────────────────────────────────────
    Examples: RealNVP, Glow, MAF, IAF.

    CORE IDEA
    ──────────
    Learn an invertible mapping f_θ : data space X → latent space Z.
    f is a composition of invertible transforms:
        z = f_K ∘ f_{K-1} ∘ … ∘ f_1(x)

    TRAINING (exact log-likelihood via Change of Variables):
        log p_X(x) = log p_Z(f(x)) + log|det(∂f/∂x)|
    Jacobian determinant must be efficiently computable (triangular structure).

    GENERATION:
        Sample z ~ p_Z (simple, e.g., Gaussian) → x = f⁻¹(z).

    VARIANTS
    ─────────
      NICE / RealNVP  : Coupling layers; efficient Jacobian
      Glow            : 1×1 invertible convolutions for images
      MAF / IAF       : Autoregressive; slow sampling or slow density eval
      Neural ODE Flow : Continuous-time flow via ODE solver
    """

    USE_CASES = [
        "Density estimation (anomaly detection, out-of-distribution detection)",
        "High-quality image generation (Glow by OpenAI)",
        "Exact likelihood for scientific data (physics, finance)",
        "Latent variable inference",
        "Molecular / drug generation with exact likelihoods",
        "Speech synthesis (WaveGlow by NVIDIA)",
    ]

    ADVANTAGES = [
        "Exact likelihood computation (not a lower bound like VAE)",
        "Invertible: both generation AND exact inference are possible",
        "No adversarial training instability",
        "Latent space has direct probabilistic meaning",
    ]

    FLAWS = [
        "Architecture must be invertible — limits design choices",
        "Memory-intensive: must store all intermediate activations",
        "Slower than GANs; less sharp than diffusion models",
        "Jacobian computation still costly for complex transforms",
        "Less commonly used since Diffusion models surpassed them",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 7. AUTOREGRESSIVE MODELS
# ─────────────────────────────────────────────────────────────────────────────
class AutoregressiveModel:
    """
    AUTOREGRESSIVE MODELS
    ──────────────────────
    Examples: PixelRNN, PixelCNN, WaveNet, GPT family.

    CORE IDEA
    ──────────
    Factorize the joint distribution using the chain rule of probability:
        p(x) = ∏_i  p(x_i | x_1, …, x_{i-1})

    Each variable (pixel, token, audio sample) is predicted given all
    previous ones — sequential conditional generation.

    TEXT  : GPT predicts next token conditioned on all previous tokens.
    IMAGE : PixelCNN predicts each pixel given all pixels above/left.
    AUDIO : WaveNet predicts each audio sample at 16 kHz resolution.

    MASKED SELF-ATTENTION (GPT mechanism):
        Future tokens are masked so each position attends only to past tokens.
        Enables parallel training (teacher-forcing) while being
        autoregressive at inference.
    """

    VARIANTS = {
        "PixelRNN":  "LSTM over image pixels (very slow)",
        "PixelCNN":  "Masked convolutions for images",
        "WaveNet":   "Dilated causal convolutions for raw audio",
        "GPT-N":     "Transformer decoder for text",
        "ImageGPT":  "GPT applied to pixel sequences",
        "VQVAE-2":   "Hierarchical VQ codes decoded autoregressively",
    }

    USE_CASES = [
        "Text generation (ALL modern LLMs)",
        "Code completion (GitHub Copilot, Claude Code)",
        "Text-to-speech / audio synthesis (WaveNet → Google TTS)",
        "Image generation (PixelCNN, ImageGPT)",
        "Music generation (Jukebox by OpenAI)",
        "Time-series generation",
    ]

    ADVANTAGES = [
        "Tractable exact likelihood",
        "Simple, principled training objective (cross-entropy / NLL)",
        "Scales excellently — GPT scaling laws hold across orders of magnitude",
        "Flexible: any discrete or continuous data can be tokenised",
    ]

    FLAWS = [
        "Inference is sequential — O(n) forward passes to generate n tokens",
        "Slow generation for long sequences, images, audio",
        "Exposure bias: model sees ground-truth at train but own output at test",
        "No explicit structure of latent space",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENERGY-BASED MODELS (EBMs)
# ─────────────────────────────────────────────────────────────────────────────
class EnergyBasedModel:
    """
    ENERGY-BASED MODELS (EBMs)
    ───────────────────────────
    Examples: Deep EBMs, JEM (Joint Energy-Based Model).

    CORE IDEA
    ──────────
    Assign a scalar energy E_θ(x) to each data point x:
        p_θ(x) = exp(−E_θ(x)) / Z_θ
    Z_θ = ∫ exp(−E_θ(x)) dx  is the intractable partition function.

    Low energy → high probability → likely / real data.
    High energy → low probability → unlikely / fake data.

    TRAINING:
    Contrastive divergence or MCMC-based methods (Langevin dynamics):
        x_{k+1} = x_k − (α/2)·∇_x E_θ(x_k) + √α · ε
    Push energy down on real data, push energy up on generated samples.

    GENERATION:
    Start from noise, run Langevin MCMC until convergence to low-energy region.
    """

    USE_CASES = [
        "Anomaly / out-of-distribution detection",
        "Robustness in classification (JEM model)",
        "Physics-inspired generative models",
        "Molecular conformation generation",
        "Image generation (less common than diffusion/GANs)",
        "Conditional generation and planning in RL",
    ]

    ADVANTAGES = [
        "Very flexible: any neural network can be an energy function",
        "Unified framework for generation, inference, and classification",
        "Does not require an explicit latent variable",
        "Can capture complex multi-modal distributions",
    ]

    FLAWS = [
        "Intractable normalising constant Z makes training hard",
        "MCMC sampling is slow and can get stuck",
        "Training instability; requires careful MCMC mixing",
        "Less mature than GANs, VAEs, Diffusion in practice",
        "Hard to evaluate (no closed-form likelihood)",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 9. BOLTZMANN MACHINES / RESTRICTED BOLTZMANN MACHINES (RBMs)
# ─────────────────────────────────────────────────────────────────────────────
class BoltzmannMachine:
    """
    BOLTZMANN MACHINES & RESTRICTED BOLTZMANN MACHINES (RBMs)
    ────────────────────────────────────────────────────────────
    Introduced by Hinton & Sejnowski (1985); RBMs by Smolensky (1986).

    ARCHITECTURE
    ─────────────
    Boltzmann Machine: fully connected stochastic binary units (visible + hidden).
    RBM: bipartite graph — visible layer v, hidden layer h; NO intra-layer connections.

    ENERGY FUNCTION (RBM):
        E(v,h) = −bᵀv − cᵀh − vᵀWh
    Joint distribution:  p(v,h) ∝ exp(−E(v,h))

    TRAINING (Contrastive Divergence — CD-k):
        Positive phase : clamp real data v⁺, infer h⁺
        Negative phase : run k Gibbs sampling steps → (v⁻, h⁻)
        ΔW ∝ E[v⁺ h⁺ᵀ] − E[v⁻ h⁻ᵀ]

    GENERATION:
        Start from random v, alternate Gibbs sampling between v and h.

    DEEP BELIEF NETWORKS (DBNs): Stack of RBMs; each RBM learns features
    of the layer below. Pre-training enabled the "deep learning revolution".
    """

    USE_CASES = [
        "Collaborative filtering (Netflix Prize era recommender systems)",
        "Feature learning and pre-training for deep networks",
        "Dimensionality reduction",
        "Handwritten digit generation (MNIST benchmarks)",
        "Drug discovery feature extraction",
        "Early speech feature learning",
    ]

    ADVANTAGES = [
        "Probabilistic generative model with clear theoretical grounding",
        "Unsupervised learning of distributed representations",
        "Efficient training via Contrastive Divergence",
        "Historically foundational — enabled deep learning resurgence",
    ]

    FLAWS = [
        "Largely superseded by VAEs, GANs, Diffusion, and Transformers",
        "CD-k is a biased approximation of the true gradient",
        "Mixing time issues in Gibbs sampling",
        "Scales poorly to high-dimensional data (images, text)",
        "Intractable partition function",
        "Limited capacity compared to modern architectures",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 10. HYBRID / MULTIMODAL MODELS
# ─────────────────────────────────────────────────────────────────────────────
class HybridMultimodalModel:
    """
    HYBRID / MULTIMODAL GENERATIVE MODELS
    ───────────────────────────────────────
    Modern state-of-the-art systems often combine multiple architectures.

    EXAMPLES
    ─────────
    DALL-E 3 (OpenAI):
        CLIP text encoder (Transformer) → Diffusion UNet (latent space)
        Combines Transformer embeddings with Diffusion generation.

    Stable Diffusion XL:
        VAE encoder/decoder + UNet with Transformer cross-attention blocks.

    Sora (OpenAI):
        Spatial-temporal Transformer (DiT — Diffusion Transformer)
        operating on video patches.

    AudioLM / MusicLM (Google):
        Autoregressive Transformer over discrete audio tokens from
        a SoundStream codec (VQ-VAE-style).

    Gemini / GPT-4o:
        Unified Transformer backbone handling text, images, audio, video.
        Vision encoder (ViT) + LLM decoder in end-to-end system.

    Flamingo (DeepMind):
        Frozen LLM + visual encoder bridged by cross-attention.

    LLAVA / InstructBLIP:
        CLIP vision encoder → projection layer → LLM (LLaMA / Vicuna).

    COMMON PATTERNS
    ────────────────
    1. Modality-specific encoder  →  shared Transformer backbone
    2. VAE / VQ-VAE latent codes  →  Autoregressive or Diffusion decoder
    3. CLIP-style contrastive pretraining  →  downstream generation
    4. Retrieval-Augmented Generation (RAG): LLM + external knowledge store
    """

    USE_CASES = [
        "Visual question answering (GPT-4V, Gemini, Claude 3+)",
        "Text-to-image and text-to-video (DALL-E 3, Sora, Runway)",
        "Document understanding (OCR + LLM)",
        "Multimodal chatbots (Claude, ChatGPT, Gemini)",
        "Code generation from screenshots / wireframes",
        "Embodied AI / robotics (vision + language + action)",
        "Audio-visual content creation",
        "Scientific multimodal analysis (radiology report + image)",
    ]

    ADVANTAGES = [
        "Leverages strengths of each architecture for different modalities",
        "Single model handles diverse tasks — reduced deployment complexity",
        "Emergent cross-modal reasoning abilities",
        "Transfer learning across modalities (CLIP zero-shot)",
    ]

    FLAWS = [
        "Extremely resource-intensive to train and serve",
        "Complex debugging — failures may stem from any component",
        "Alignment across modalities is non-trivial",
        "Hallucinations amplified across modalities",
        "Proprietary nature of frontier models limits research access",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE (printed utility function)
# ─────────────────────────────────────────────────────────────────────────────
def print_comparison_table():
    """
    Print a side-by-side comparison of all Generative AI network types
    across key dimensions.
    """
    import textwrap

    NETWORKS = [
        # (Name, Training Stability, Sample Quality, Speed, Likelihood, Best For)
        ("GAN",           "★★☆ Unstable",   "★★★ Excellent", "★★★ Fast",   "✗ None",         "Photo-realistic images"),
        ("VAE",           "★★★ Stable",     "★★☆ Blurry",    "★★★ Fast",   "≈  ELBO bound",  "Latent space editing"),
        ("Transformer",   "★★★ Stable",     "★★★ Excellent", "★★☆ Medium", "★★★ Exact NLL",  "Text & code generation"),
        ("Diffusion",     "★★★ Stable",     "★★★ Excellent", "★☆☆ Slow",   "★★☆ Approx",     "Text-to-image/video"),
        ("RNN / LSTM",    "★★☆ Medium",     "★★☆ OK",        "★☆☆ Slow",   "★★★ Exact NLL",  "Sequential data"),
        ("Flow-Based",    "★★★ Stable",     "★★☆ Good",      "★★☆ Medium", "★★★ Exact",      "Density estimation"),
        ("Autoregressive","★★★ Stable",     "★★★ Excellent", "★☆☆ Slow",   "★★★ Exact NLL",  "Language models"),
        ("EBM",           "★☆☆ Unstable",   "★★☆ OK",        "★☆☆ Slow",   "✗ Intractable",  "Anomaly detection"),
        ("RBM",           "★★☆ Medium",     "★☆☆ Limited",   "★★☆ Medium", "✗ Intractable",  "Feature pre-training"),
        ("Hybrid",        "★★★ Varies",     "★★★ Best",      "★★☆ Varies", "★★☆ Varies",     "Multimodal tasks"),
    ]

    header = f"{'Network':<16} {'Stability':<20} {'Quality':<20} {'Speed':<16} {'Likelihood':<20} {'Best For'}"
    sep = "─" * 110
    print("\n" + sep)
    print("  GENERATIVE AI NETWORKS — COMPARISON TABLE")
    print(sep)
    print(header)
    print(sep)
    for row in NETWORKS:
        print(f"{row[0]:<16} {row[1]:<20} {row[2]:<20} {row[3]:<16} {row[4]:<20} {row[5]}")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# TIMELINE OF KEY MILESTONES
# ─────────────────────────────────────────────────────────────────────────────
TIMELINE = {
    1985: "Boltzmann Machines (Hinton & Sejnowski)",
    1997: "Long Short-Term Memory — LSTM (Hochreiter & Schmidhuber)",
    2013: "Variational Autoencoder — VAE (Kingma & Welling)",
    2014: "Generative Adversarial Network — GAN (Goodfellow et al.)",
    2016: "WaveNet autoregressive audio model (DeepMind)",
    2016: "Real-NVP Normalizing Flow (Dinh et al.)",
    2017: "'Attention Is All You Need' — Transformer (Vaswani et al.)",
    2018: "GPT-1; BERT (OpenAI / Google)",
    2019: "GPT-2; StyleGAN (OpenAI / NVIDIA)",
    2020: "GPT-3; DDPM Diffusion Model; VQ-VAE-2",
    2021: "DALL-E v1; CLIP; Codex (OpenAI)",
    2022: "Stable Diffusion; ChatGPT; Whisper; DALL-E 2",
    2023: "GPT-4; Llama; Mistral; DALL-E 3; Stable Diffusion XL",
    2024: "Sora; Gemini 1.5; Claude 3; Llama 3",
    2025: "GPT-4o native multimodal; Claude 4; Gemini 2; Llama 4 (multimodal)",
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run a quick demo print
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Print all network summaries
    networks = [GAN, VAE, Transformer, DiffusionModel, RNN_LSTM,
                NormalizingFlow, AutoregressiveModel, EnergyBasedModel,
                BoltzmannMachine, HybridMultimodalModel]

    for net in networks:
        print("\n" + "═" * 80)
        print(net.__doc__)

        # Print use cases
        if hasattr(net, "USE_CASES"):
            print("  USE CASES:")
            for uc in net.USE_CASES:
                print(f"    ✔  {uc}")

        # Print advantages
        if hasattr(net, "ADVANTAGES"):
            print("\n  ADVANTAGES:")
            for adv in net.ADVANTAGES:
                print(f"    ✅  {adv}")

        # Print flaws
        if hasattr(net, "FLAWS"):
            print("\n  FLAWS / LIMITATIONS:")
            for flaw in net.FLAWS:
                print(f"    ⚠️   {flaw}")

    # Comparison table
    print_comparison_table()

    # Timeline
    print("═" * 80)
    print("  KEY MILESTONES IN GENERATIVE AI")
    print("═" * 80)
    for year, event in sorted(TIMELINE.items()):
        print(f"  {year}  →  {event}")
    print()