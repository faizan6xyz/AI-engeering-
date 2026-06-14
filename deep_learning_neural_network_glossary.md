# Deep Learning & Neural Network — Complete Glossary

> A complete reference guide covering every major term in Deep Learning and Neural Networks,
> organized by category with explanations and use cases.

---

## 📚 TABLE OF CONTENTS

1. [Foundational Concepts](#1-foundational-concepts)
2. [Neural Network Basics](#2-neural-network-basics)
3. [Activation Functions](#3-activation-functions)
4. [Network Architectures](#4-network-architectures)
5. [Training Concepts](#5-training-concepts)
6. [Loss Functions](#6-loss-functions)
7. [Optimizers](#7-optimizers)
8. [Regularization Techniques](#8-regularization-techniques)
9. [Normalization Techniques](#9-normalization-techniques)
10. [Attention & Transformer Terms](#10-attention--transformer-terms)
11. [Generative Models](#11-generative-models)
12. [Convolutional Network Terms](#12-convolutional-network-terms)
13. [Recurrent Network Terms](#13-recurrent-network-terms)
14. [Evaluation Metrics](#14-evaluation-metrics)
15. [Advanced Techniques](#15-advanced-techniques)
16. [Hardware & Infrastructure](#16-hardware--infrastructure)
17. [Data Terms](#17-data-terms)

---

## 1. FOUNDATIONAL CONCEPTS

---

### 🔷 Artificial Intelligence (AI)
**What it is:** The broad field of building machines that can perform tasks that normally require human intelligence.
**Use Case:** Self-driving cars, voice assistants, recommendation systems.

---

### 🔷 Machine Learning (ML)
**What it is:** A subset of AI where systems learn patterns from data rather than being explicitly programmed.
**Use Case:** Spam detection, price prediction, fraud detection.

---

### 🔷 Deep Learning (DL)
**What it is:** A subset of ML that uses neural networks with many layers (deep architectures) to learn hierarchical representations from data.
**Use Case:** Image recognition, speech synthesis, language translation.

```
AI ⊃ Machine Learning ⊃ Deep Learning ⊃ Neural Networks
```

---

### 🔷 Neural Network (NN)
**What it is:** A computational system loosely inspired by the human brain, consisting of interconnected nodes (neurons) organized in layers.
**Use Case:** Foundation for all deep learning models.

---

### 🔷 Artificial Neural Network (ANN)
**What it is:** The general term for any neural network built computationally. Consists of an input layer, hidden layers, and an output layer.
**Use Case:** Classification, regression, pattern recognition.

---

### 🔷 Deep Neural Network (DNN)
**What it is:** An ANN with multiple hidden layers (usually 3+), allowing it to learn complex hierarchical features.
**Use Case:** Complex tasks like image classification, NLP, speech recognition.

---

### 🔷 Parameter
**What it is:** A learnable value inside a model — includes weights and biases. These get updated during training.
**Use Case:** Defines the model's learned knowledge. GPT-4 has ~1 trillion parameters.

---

### 🔷 Hyperparameter
**What it is:** A setting defined BEFORE training that controls the learning process — not learned from data.
**Examples:** Learning rate, batch size, number of layers, dropout rate.
**Use Case:** Tuning hyperparameters is critical for model performance.

---

### 🔷 Inference
**What it is:** Using a trained model to make predictions on new, unseen data.
**Use Case:** Running ChatGPT to answer your question — that's inference.

---

### 🔷 Training
**What it is:** The process of feeding data to a model and adjusting its parameters to minimize error.
**Use Case:** Teaching the model to recognize cats, translate text, or generate code.

---

### 🔷 Model
**What it is:** The mathematical structure (architecture + learned parameters) that maps inputs to outputs.
**Use Case:** A trained ResNet model that classifies images into 1000 categories.

---

## 2. NEURAL NETWORK BASICS

---

### 🔶 Neuron (Node)
**What it is:** The basic unit of a neural network. Takes inputs, applies a weight, adds a bias, then passes through an activation function.
**Formula:** `output = activation(Σ(weight × input) + bias)`
**Use Case:** Building block of every neural network.

---

### 🔶 Weight
**What it is:** A learnable number that determines the strength of connection between two neurons. Adjusted during training.
**Use Case:** Encodes what the model has "learned" about the data.

---

### 🔶 Bias
**What it is:** A learnable constant added to the weighted sum in a neuron, allowing the model to shift the activation function.
**Use Case:** Helps the model fit data that doesn't pass through the origin.

---

### 🔶 Layer
**What it is:** A group of neurons that process information at the same level. Data flows layer by layer.

| Layer Type | Role |
|---|---|
| Input Layer | Receives raw data |
| Hidden Layer | Learns intermediate features |
| Output Layer | Produces the final prediction |

---

### 🔶 Input Layer
**What it is:** The first layer that receives raw data (pixels, text tokens, numbers).
**Use Case:** For an image model: each pixel = one input neuron.

---

### 🔶 Hidden Layer
**What it is:** Layers between input and output that learn internal representations of data.
**Use Case:** Early layers learn edges; deeper layers learn faces or objects in image recognition.

---

### 🔶 Output Layer
**What it is:** The final layer that produces the model's prediction or decision.
**Use Case:** In classification: outputs probability for each class.

---

### 🔶 Fully Connected Layer (Dense Layer)
**What it is:** Every neuron in one layer connects to every neuron in the next layer.
**Use Case:** Final layers in classifiers, regression outputs.

---

### 🔶 Forward Propagation
**What it is:** The process of passing input data through the network layer by layer to produce an output/prediction.
**Use Case:** How the model makes a prediction.

---

### 🔶 Backpropagation (Backprop)
**What it is:** Algorithm that calculates how much each weight contributed to the error, then adjusts weights in reverse (output → input) using the chain rule of calculus.
**Use Case:** Core training algorithm for all neural networks.

```
Loss → Output Layer → Hidden Layers → Input Layer
       (gradients flow backwards to update weights)
```

---

### 🔶 Gradient
**What it is:** A vector that tells us how much the loss changes with respect to each weight. Points in the direction of steepest increase in loss.
**Use Case:** Used by optimizers to update weights in the opposite direction.

---

### 🔶 Chain Rule
**What it is:** A calculus rule used in backpropagation to compute gradients through multiple layers by multiplying local gradients together.
**Use Case:** Makes backpropagation mathematically possible.

---

### 🔶 Perceptron
**What it is:** The simplest neural network — a single neuron with a binary step activation. The earliest neural network model (1958).
**Use Case:** Binary classification (historical); foundation of modern NNs.

---

### 🔶 Multi-Layer Perceptron (MLP)
**What it is:** A feedforward neural network with one or more hidden layers. The most basic form of deep learning.
**Use Case:** Tabular data classification, simple regression.

---

## 3. ACTIVATION FUNCTIONS

> Activation functions introduce **non-linearity** into the network, allowing it to learn complex patterns.

---

### 🟢 Sigmoid
**Formula:** `σ(x) = 1 / (1 + e^(-x))`
**Output Range:** 0 to 1
**Use Case:** Binary classification output layer. Converts scores to probabilities.
**Problem:** Vanishing gradient for very large or small inputs.

---

### 🟢 Tanh (Hyperbolic Tangent)
**Formula:** `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
**Output Range:** -1 to 1
**Use Case:** Hidden layers in RNNs; zero-centered output.
**Problem:** Still has vanishing gradient issue.

---

### 🟢 ReLU (Rectified Linear Unit)
**Formula:** `f(x) = max(0, x)`
**Output Range:** 0 to ∞
**Use Case:** Most common hidden layer activation. Fast, simple, effective for CNNs and DNNs.
**Problem:** Dying ReLU (neurons stuck at 0).

---

### 🟢 Leaky ReLU
**Formula:** `f(x) = max(0.01x, x)`
**Output Range:** -∞ to ∞ (small negative slope)
**Use Case:** Fixes dying ReLU problem by allowing small negative values.

---

### 🟢 ELU (Exponential Linear Unit)
**Formula:** `f(x) = x if x > 0, else α(e^x - 1)`
**Use Case:** Smoother version of Leaky ReLU; helps with faster learning.

---

### 🟢 GELU (Gaussian Error Linear Unit)
**Formula:** `GELU(x) = x × Φ(x)` (Φ = Gaussian CDF)
**Use Case:** Used in transformers (BERT, GPT). Smoother than ReLU, better for NLP.

---

### 🟢 Swish
**Formula:** `f(x) = x × sigmoid(x)`
**Use Case:** Used in EfficientNet; slightly outperforms ReLU in many deep models.

---

### 🟢 Softmax
**Formula:** `softmax(xᵢ) = e^xᵢ / Σe^xⱼ`
**Output Range:** 0 to 1 (sums to 1)
**Use Case:** Multi-class classification output layer. Converts raw scores to probabilities.

---

### 🟢 Linear (Identity)
**Formula:** `f(x) = x`
**Use Case:** Regression output layer. No transformation — raw values passed through.

---

### 🟢 PReLU (Parametric ReLU)
**What it is:** Like Leaky ReLU but the slope is a learned parameter.
**Use Case:** When you want the model to learn the best negative slope.

---

## 4. NETWORK ARCHITECTURES

---

### 🔵 Feedforward Neural Network (FNN)
**What it is:** Data flows in one direction only — from input to output. No cycles or loops.
**Use Case:** Basic classification, regression tasks.

---

### 🔵 Convolutional Neural Network (CNN)
**What it is:** Specialized for grid-like data (images). Uses convolutional layers to detect spatial features like edges, textures, and shapes.
**Use Case:** Image classification, object detection, facial recognition.

---

### 🔵 Recurrent Neural Network (RNN)
**What it is:** Has loops/cycles — output from one step is fed back as input to the next. Designed for sequential data.
**Use Case:** Time series, speech, early NLP.
**Problem:** Vanishing gradients over long sequences.

---

### 🔵 Long Short-Term Memory (LSTM)
**What it is:** An improved RNN with special "gates" (input, forget, output) that allow it to remember information over long sequences.
**Use Case:** Machine translation, speech recognition, text generation.

---

### 🔵 Gated Recurrent Unit (GRU)
**What it is:** A simplified version of LSTM with fewer gates (reset and update). Faster to train.
**Use Case:** Similar to LSTM but computationally cheaper.

---

### 🔵 Transformer
**What it is:** Architecture based entirely on the attention mechanism. Processes all tokens in parallel (unlike RNNs). Foundation of all modern LLMs.
**Use Case:** GPT, BERT, Claude, T5 — NLP, code generation, image generation.

---

### 🔵 Encoder
**What it is:** Part of a model that compresses input into a compact internal representation (latent space).
**Use Case:** BERT uses encoder-only architecture for text understanding.

---

### 🔵 Decoder
**What it is:** Part of a model that generates output from the latent representation.
**Use Case:** GPT uses decoder-only architecture for text generation.

---

### 🔵 Encoder-Decoder (Seq2Seq)
**What it is:** Encoder compresses input, decoder generates output from it.
**Use Case:** Machine translation, summarization, T5 model.

---

### 🔵 Autoencoder
**What it is:** Neural network trained to compress data into a smaller representation and then reconstruct it.
```
Input → [Encoder] → Latent Space → [Decoder] → Reconstructed Input
```
**Use Case:** Dimensionality reduction, anomaly detection, denoising.

---

### 🔵 Variational Autoencoder (VAE)
**What it is:** An autoencoder that learns a probabilistic distribution over the latent space, allowing it to generate new samples by sampling from that distribution.
**Use Case:** Image generation, data augmentation, generative modeling.

---

### 🔵 Generative Adversarial Network (GAN)
**What it is:** Two networks (Generator vs Discriminator) compete against each other. Generator creates fake data; Discriminator tries to detect fakes.
**Use Case:** Realistic image generation, deepfakes, style transfer, data augmentation.

---

### 🔵 Diffusion Model
**What it is:** Learns to reverse a gradual noising process. Adds noise to data step by step, then learns to denoise.
**Use Case:** Stable Diffusion, DALL-E, Midjourney — state-of-the-art image generation.

---

### 🔵 Vision Transformer (ViT)
**What it is:** Applies the transformer architecture to images by splitting them into patches and treating patches as tokens.
**Use Case:** Image classification, object detection.

---

### 🔵 Residual Network (ResNet)
**What it is:** Uses "skip connections" or "residual connections" that bypass one or more layers, allowing very deep networks to train without vanishing gradients.
**Use Case:** Deep image recognition (ResNet-50, ResNet-152).

---

### 🔵 U-Net
**What it is:** Encoder-decoder architecture with skip connections between matching encoder and decoder layers.
**Use Case:** Medical image segmentation, image-to-image translation.

---

### 🔵 Siamese Network
**What it is:** Two identical networks sharing weights, comparing two inputs to measure similarity.
**Use Case:** Face verification, signature matching, one-shot learning.

---

### 🔵 Graph Neural Network (GNN)
**What it is:** Neural network that operates on graph-structured data (nodes + edges).
**Use Case:** Social networks, molecule property prediction, knowledge graphs.

---

### 🔵 Mixture of Experts (MoE)
**What it is:** Architecture where different "expert" sub-networks specialize in different types of inputs. A router selects which experts to use for each input.
**Use Case:** Efficient scaling — GPT-4, Mixtral use MoE.

---

## 5. TRAINING CONCEPTS

---

### 🟠 Epoch
**What it is:** One complete pass through the entire training dataset.
**Use Case:** Training for 10 epochs means the model sees all data 10 times.

---

### 🟠 Batch
**What it is:** A subset of the training data used in one forward/backward pass.
**Use Case:** Instead of updating weights after every example, update after every batch.

---

### 🟠 Batch Size
**What it is:** The number of training examples in one batch.
**Examples:**
- Batch Size = 1 → Stochastic Gradient Descent (SGD)
- Batch Size = 32/64 → Mini-batch (most common)
- Batch Size = All data → Full Batch Gradient Descent

---

### 🟠 Iteration
**What it is:** One forward + backward pass using one batch.
`Iterations per Epoch = Total Samples / Batch Size`
**Use Case:** Tracking training progress.

---

### 🟠 Learning Rate (LR)
**What it is:** Controls how much weights are updated per step. Too high = diverges. Too low = too slow.
**Use Case:** Most important hyperparameter. Typical values: 0.001–0.0001.

---

### 🟠 Learning Rate Scheduler
**What it is:** Automatically adjusts the learning rate during training (e.g., decay it over time).
**Types:** Step Decay, Cosine Annealing, Warm Restarts, ReduceLROnPlateau.
**Use Case:** Better convergence — start with high LR, reduce as training progresses.

---

### 🟠 Gradient Descent (GD)
**What it is:** Optimization algorithm that updates weights by moving in the direction of the negative gradient of the loss.
**Types:**
- Full Batch GD: Uses all data per update
- Stochastic GD (SGD): Uses 1 sample per update
- Mini-Batch GD: Uses a small batch per update

---

### 🟠 Momentum
**What it is:** Adds a fraction of the previous weight update to the current update, helping the optimizer move faster in consistent directions.
**Use Case:** Helps escape local minima and oscillations.

---

### 🟠 Overfitting
**What it is:** Model performs well on training data but poorly on new/test data. Memorizes instead of generalizing.
**Solutions:** Dropout, L2 regularization, more data, early stopping.

---

### 🟠 Underfitting
**What it is:** Model performs poorly even on training data. Too simple to capture patterns.
**Solutions:** More layers, more epochs, reduce regularization, better features.

---

### 🟠 Generalization
**What it is:** The ability of a trained model to perform well on new, unseen data.
**Use Case:** The ultimate goal of all ML training.

---

### 🟠 Vanishing Gradient
**What it is:** Gradients become extremely small as they propagate backward through deep networks, making early layers learn very slowly or not at all.
**Solutions:** ReLU, LSTM, residual connections, batch normalization.

---

### 🟠 Exploding Gradient
**What it is:** Gradients become extremely large, causing unstable training and NaN values.
**Solutions:** Gradient clipping, weight initialization techniques.

---

### 🟠 Gradient Clipping
**What it is:** Caps the gradient magnitude to a maximum value to prevent exploding gradients.
**Use Case:** Especially important in RNN/LSTM training.

---

### 🟠 Early Stopping
**What it is:** Stop training when validation loss stops improving, to prevent overfitting.
**Use Case:** Saves compute and gives the best generalizing model.

---

### 🟠 Model Checkpoint
**What it is:** Saving the model weights at specific points during training (e.g., when validation loss is lowest).
**Use Case:** Resume training, use best performing version.

---

### 🟠 Transfer Learning
**What it is:** Taking a model pre-trained on a large dataset and fine-tuning it for a different but related task.
**Use Case:** Use ImageNet-trained ResNet for medical imaging. Use GPT for custom chatbots.

---

### 🟠 Fine-Tuning
**What it is:** Continuing to train a pre-trained model on a new dataset, usually with a small learning rate.
**Use Case:** Adapting LLMs for specific domains (legal, medical, coding).

---

### 🟠 Pre-Training
**What it is:** Training a model on a large general dataset before fine-tuning on a specific task.
**Use Case:** GPT is pre-trained on internet text, then fine-tuned for chat.

---

### 🟠 Few-Shot Learning
**What it is:** Training or adapting a model with very few labeled examples (1–10 samples).
**Use Case:** LLMs can learn new tasks from just a few examples in the prompt.

---

### 🟠 Zero-Shot Learning
**What it is:** Model performs a task it was never explicitly trained on, using only the description.
**Use Case:** "Classify this email as spam or not" — without having trained on spam classification.

---

### 🟠 Knowledge Distillation
**What it is:** Training a smaller "student" model to mimic the outputs of a larger "teacher" model.
**Use Case:** Compressing large models for deployment on edge devices.

---

### 🟠 Data Augmentation
**What it is:** Artificially increasing the size of a dataset by applying transformations (rotation, flipping, cropping, noise) to existing data.
**Use Case:** Images: rotate/flip. Text: synonym replacement. Audio: pitch shift.

---

### 🟠 Class Imbalance
**What it is:** When one class has far more samples than another (e.g., 99% normal, 1% fraud).
**Solutions:** Oversampling (SMOTE), undersampling, class weights, focal loss.

---

## 6. LOSS FUNCTIONS

> Loss functions measure how wrong the model's predictions are. The goal of training is to minimize them.

---

### 🔴 Mean Squared Error (MSE)
**Formula:** `MSE = (1/n) Σ(y_pred - y_true)²`
**Use Case:** Regression tasks. Penalizes large errors heavily.

---

### 🔴 Mean Absolute Error (MAE)
**Formula:** `MAE = (1/n) Σ|y_pred - y_true|`
**Use Case:** Regression. Less sensitive to outliers than MSE.

---

### 🔴 Binary Cross-Entropy (BCE)
**Formula:** `BCE = -[y·log(p) + (1-y)·log(1-p)]`
**Use Case:** Binary classification (spam/not spam, cat/dog).

---

### 🔴 Categorical Cross-Entropy
**Formula:** `CE = -Σ y_true · log(y_pred)`
**Use Case:** Multi-class classification with one-hot labels.

---

### 🔴 Sparse Categorical Cross-Entropy
**What it is:** Same as categorical cross-entropy but uses integer labels instead of one-hot.
**Use Case:** Multi-class classification with integer class indices.

---

### 🔴 KL Divergence (Kullback-Leibler)
**What it is:** Measures how one probability distribution differs from another.
**Use Case:** VAEs, language model training, knowledge distillation.

---

### 🔴 Hinge Loss
**What it is:** Penalizes predictions that are correct but not confident enough.
**Use Case:** Support Vector Machines (SVM), binary classification.

---

### 🔴 Huber Loss
**What it is:** Combines MSE and MAE — behaves like MSE for small errors, MAE for large ones. Robust to outliers.
**Use Case:** Regression with outliers, reinforcement learning.

---

### 🔴 Focal Loss
**What it is:** Modified cross-entropy that down-weights easy examples and focuses on hard ones.
**Use Case:** Object detection with class imbalance (RetinaNet).

---

### 🔴 Contrastive Loss
**What it is:** Minimizes distance between similar pairs, maximizes distance between dissimilar pairs.
**Use Case:** Siamese networks, similarity learning.

---

### 🔴 Triplet Loss
**What it is:** Uses an anchor, positive (similar), and negative (dissimilar) sample. Trains model to embed similar items closer.
**Use Case:** Face recognition, embedding learning.

---

## 7. OPTIMIZERS

> Optimizers use the gradient to update model weights during training.

---

### 🟣 SGD (Stochastic Gradient Descent)
**What it is:** Updates weights using gradient from a single or mini-batch of samples.
**Use Case:** Basic workhorse. Still used in CNNs with momentum.

---

### 🟣 SGD with Momentum
**What it is:** SGD + momentum term that accumulates velocity in consistent directions.
**Use Case:** Faster convergence than pure SGD.

---

### 🟣 Adagrad
**What it is:** Adapts the learning rate per parameter — larger updates for infrequent features.
**Problem:** Learning rate shrinks too aggressively over time.
**Use Case:** Sparse data, NLP.

---

### 🟣 RMSprop
**What it is:** Fixes Adagrad's aggressive decay by using a moving average of squared gradients.
**Use Case:** RNNs, non-stationary problems.

---

### 🟣 Adam (Adaptive Moment Estimation)
**What it is:** Combines momentum + RMSprop. Maintains moving averages of gradients AND squared gradients.
**Use Case:** Most popular optimizer. Works well for most deep learning tasks.

---

### 🟣 AdamW
**What it is:** Adam with decoupled weight decay regularization.
**Use Case:** Training transformers and LLMs (BERT, GPT use AdamW).

---

### 🟣 Adadelta
**What it is:** Improved Adagrad that uses a limited window of past gradients instead of accumulating all of them.
**Use Case:** When you don't want to manually set learning rate.

---

### 🟣 Nadam
**What it is:** Adam + Nesterov momentum for even better convergence.
**Use Case:** When Adam is good but need slightly better performance.

---

### 🟣 Lion Optimizer
**What it is:** A newer optimizer (2023) that uses sign of gradient instead of magnitude — more memory efficient than Adam.
**Use Case:** Large model training where memory is critical.

---

## 8. REGULARIZATION TECHNIQUES

> Regularization prevents overfitting by constraining the model.

---

### 🟤 L1 Regularization (Lasso)
**What it is:** Adds the absolute sum of weights as a penalty to the loss.
**Effect:** Pushes some weights to exactly zero — sparse models.
**Use Case:** Feature selection, sparse neural networks.

---

### 🟤 L2 Regularization (Ridge / Weight Decay)
**What it is:** Adds the squared sum of weights as a penalty to the loss.
**Effect:** Keeps weights small but not zero.
**Use Case:** Most common regularization. Used in AdamW.

---

### 🟤 Dropout
**What it is:** Randomly sets a fraction of neurons to zero during each training step. Forces the network to learn redundant representations.
**Use Case:** Prevents co-adaptation of neurons. Used in almost all deep networks.

---

### 🟤 DropConnect
**What it is:** Like Dropout but randomly zeros out weights instead of neurons.
**Use Case:** Alternative to Dropout for regularizing connections.

---

### 🟤 Early Stopping
**What it is:** Stop training when validation performance stops improving.
**Use Case:** Simple and effective regularization.

---

### 🟤 Data Augmentation (as Regularizer)
**What it is:** Artificially adds variety to training data, reducing overfitting.
**Use Case:** Standard in computer vision training.

---

### 🟤 Label Smoothing
**What it is:** Instead of hard 0/1 labels, uses soft labels (e.g., 0.9/0.1). Prevents overconfident predictions.
**Use Case:** Image classification, NLP — often improves generalization.

---

### 🟤 Stochastic Depth
**What it is:** Randomly drops entire layers during training.
**Use Case:** Very deep networks like ResNets during training.

---

### 🟤 DropBlock
**What it is:** Drops contiguous regions of feature maps (blocks) instead of random individual neurons.
**Use Case:** CNNs — better than standard dropout for spatial data.

---

## 9. NORMALIZATION TECHNIQUES

> Normalization stabilizes and accelerates training.

---

### 🟡 Batch Normalization (BatchNorm)
**What it is:** Normalizes the output of a layer across the mini-batch (zero mean, unit variance). Adds learnable scale (γ) and shift (β) parameters.
**Use Case:** CNNs, deep networks. Allows higher learning rates, faster training.

---

### 🟡 Layer Normalization (LayerNorm)
**What it is:** Normalizes across the features of a single sample (not across the batch).
**Use Case:** Transformers, RNNs — works well with variable-length sequences.

---

### 🟡 Instance Normalization
**What it is:** Normalizes each sample individually, per channel.
**Use Case:** Style transfer, image generation.

---

### 🟡 Group Normalization
**What it is:** Divides channels into groups and normalizes within each group.
**Use Case:** Object detection with small batch sizes.

---

### 🟡 Weight Normalization
**What it is:** Reparameterizes weight vectors to decouple their magnitude from direction.
**Use Case:** Reinforcement learning, generative models.

---

### 🟡 RMS Norm (Root Mean Square Normalization)
**What it is:** Simplified layer norm without mean centering — only scales by RMS.
**Use Case:** LLaMA, modern LLMs — faster than LayerNorm.

---

## 10. ATTENTION & TRANSFORMER TERMS

---

### 🔷 Attention Mechanism
**What it is:** Allows the model to focus on the most relevant parts of the input when generating each output token. Computes a weighted sum of values based on query-key similarity.
**Use Case:** Core of all transformer models.

---

### 🔷 Self-Attention
**What it is:** Each token attends to all other tokens in the same sequence — including itself.
**Use Case:** Understanding context — "bank" in "river bank" vs "savings bank".

---

### 🔷 Multi-Head Attention
**What it is:** Runs multiple attention mechanisms in parallel (different heads), each learning different aspects of relationships.
**Use Case:** Allows the model to attend to different positions for different reasons simultaneously.

---

### 🔷 Cross-Attention
**What it is:** One sequence (query) attends to a different sequence (key/value). Bridges encoder and decoder.
**Use Case:** Machine translation — decoder attends to encoder's output.

---

### 🔷 Query (Q), Key (K), Value (V)
**What it is:** Three learned projections used in attention:
- **Query:** What we're looking for
- **Key:** What we're comparing against
- **Value:** The actual information retrieved
**Formula:** `Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V`

---

### 🔷 Positional Encoding
**What it is:** Since transformers process all tokens simultaneously, positional encoding adds position information to each token embedding.
**Types:** Sinusoidal (original), Learned, RoPE (Rotary), ALiBi.
**Use Case:** Transformers need this to know word order.

---

### 🔷 Token
**What it is:** The basic unit of text input to a language model. Can be a word, subword, or character depending on the tokenizer.
**Use Case:** "Hello world" → ["Hello", " world"] = 2 tokens.

---

### 🔷 Tokenizer
**What it is:** Converts raw text into tokens (numbers) the model can process.
**Types:** BPE, WordPiece, SentencePiece, Unigram.
**Use Case:** Every LLM needs a tokenizer before processing text.

---

### 🔷 Context Window
**What it is:** The maximum number of tokens a model can process in a single forward pass.
**Use Case:** GPT-4 Turbo: 128k tokens. Determines how much text the model "sees" at once.

---

### 🔷 Embedding
**What it is:** A dense vector representation of a token that captures its meaning in a continuous space.
**Use Case:** "King" → [0.2, 0.8, -0.3, ...] — similar words have similar vectors.

---

### 🔷 Positional Embedding
**What it is:** Learned or computed vectors added to token embeddings to encode position.
**Use Case:** Helps transformer know that "dog bites man" ≠ "man bites dog".

---

### 🔷 Feed-Forward Network (FFN) in Transformer
**What it is:** A two-layer MLP applied independently to each position after the attention layer. Expands and then contracts the dimension.
**Use Case:** Adds capacity for learning complex transformations.

---

### 🔷 Encoder Block
**What it is:** Self-Attention + LayerNorm + FFN + LayerNorm (with residual connections).
**Use Case:** Understanding and encoding input (BERT).

---

### 🔷 Decoder Block
**What it is:** Self-Attention + Cross-Attention + LayerNorm + FFN + LayerNorm.
**Use Case:** Generating output tokens (GPT, T5 decoder).

---

### 🔷 Residual Connection (Skip Connection)
**What it is:** Adds the input of a layer directly to its output: `output = F(x) + x`.
**Use Case:** Prevents vanishing gradients in very deep networks (ResNet, Transformers).

---

### 🔷 KV Cache
**What it is:** Stores the Key and Value matrices from past tokens during inference so they don't need to be recomputed at every step.
**Use Case:** Critical for efficient LLM inference — speeds up generation significantly.

---

### 🔷 RoPE (Rotary Position Embedding)
**What it is:** Encodes positional information by rotating the query and key vectors. Allows better length generalization.
**Use Case:** LLaMA, GPT-NeoX, modern LLMs.

---

### 🔷 Flash Attention
**What it is:** An efficient attention algorithm that reduces memory usage by computing attention in tiles without materializing the full attention matrix.
**Use Case:** Training large transformers on long sequences efficiently.

---

## 11. GENERATIVE MODELS

---

### 🟠 Generative Model
**What it is:** A model that learns the underlying distribution of data and can generate new samples from it.
**Use Case:** Image generation, text generation, music generation.

---

### 🟠 Discriminative Model
**What it is:** A model that learns the boundary between classes — just predicts labels, doesn't generate data.
**Use Case:** Classification, regression.

---

### 🟠 GAN — Generator
**What it is:** The part of a GAN that creates fake data from random noise, trying to fool the discriminator.
**Use Case:** Generating realistic images, faces, video.

---

### 🟠 GAN — Discriminator
**What it is:** The part of a GAN that tries to distinguish real data from generated (fake) data.
**Use Case:** Acts as a judge that improves the generator through adversarial training.

---

### 🟠 Latent Space
**What it is:** A compressed, abstract vector space where the model encodes data. Points in latent space decode to realistic outputs.
**Use Case:** VAEs and GANs use latent space for generation. Interpolating in latent space morphs between outputs.

---

### 🟠 Latent Vector (z)
**What it is:** A point in latent space — a compressed representation of data.
**Use Case:** Sampling a random z from a normal distribution → decoding → new image.

---

### 🟠 Diffusion Process
**What it is:** Gradually adds Gaussian noise to data over many steps. The model learns to reverse this process (denoise).
**Use Case:** Stable Diffusion, DALL-E 2, Imagen.

---

### 🟠 Score Function
**What it is:** The gradient of the log probability density — used in diffusion models to guide the denoising direction.
**Use Case:** Score-based generative models.

---

### 🟠 RLHF (Reinforcement Learning from Human Feedback)
**What it is:** Training technique where human raters rank model outputs, and a reward model is trained on these preferences to fine-tune the main model.
**Use Case:** ChatGPT, Claude, InstructGPT — aligns LLMs with human preferences.

---

## 12. CONVOLUTIONAL NETWORK TERMS

---

### 🔵 Convolution
**What it is:** Applies a small filter/kernel that slides over the input, computing dot products to detect local features.
**Use Case:** Detecting edges, textures, shapes in images.

---

### 🔵 Kernel / Filter
**What it is:** A small learnable matrix (e.g., 3×3 or 5×5) that slides over the input to detect specific features.
**Use Case:** Early kernels detect edges; deeper kernels detect complex shapes.

---

### 🔵 Feature Map
**What it is:** The output of applying a convolutional filter to an input — a spatial map of where a feature was detected.
**Use Case:** Each filter produces one feature map.

---

### 🔵 Stride
**What it is:** How many pixels the filter moves at each step. Larger stride = smaller output.
**Use Case:** Controls output spatial size. Stride=2 halves the spatial dimensions.

---

### 🔵 Padding
**What it is:** Adding zeros around the border of the input to control output size.
- **Same Padding:** Output = Input size
- **Valid Padding:** Output shrinks after convolution
**Use Case:** Preserving spatial dimensions.

---

### 🔵 Pooling
**What it is:** Downsamples the feature map by summarizing regions.
**Types:**
- Max Pooling: Takes maximum value in each region
- Average Pooling: Takes average value in each region
- Global Average Pooling: Reduces entire feature map to one value per channel
**Use Case:** Reduces spatial size, introduces translation invariance.

---

### 🔵 Depthwise Separable Convolution
**What it is:** Splits convolution into depthwise (per channel) and pointwise (1×1 across channels). Much more efficient.
**Use Case:** MobileNet, EfficientNet — mobile/edge deployment.

---

### 🔵 Dilated / Atrous Convolution
**What it is:** Convolution with gaps (dilation rate) between kernel elements, increasing the receptive field without more parameters.
**Use Case:** Semantic segmentation, audio processing (WaveNet).

---

### 🔵 Receptive Field
**What it is:** The area of the input that a particular neuron in a deep layer "sees" or is influenced by.
**Use Case:** Deeper layers have larger receptive fields — see more of the image.

---

### 🔵 1×1 Convolution (Pointwise)
**What it is:** A convolution with a 1×1 kernel — changes the number of channels without affecting spatial dimensions.
**Use Case:** Dimensionality reduction, adding non-linearity across channels.

---

## 13. RECURRENT NETWORK TERMS

---

### 🟡 Hidden State
**What it is:** The internal memory of an RNN that carries information from previous time steps.
**Use Case:** Remembering earlier words in a sentence.

---

### 🟡 Cell State (LSTM)
**What it is:** Long-term memory in LSTM that flows through the network with minimal modification, controlled by gates.
**Use Case:** Stores long-range dependencies that the network decides to keep.

---

### 🟡 Forget Gate
**What it is:** LSTM gate that decides which information to throw away from the cell state.
**Use Case:** Forgetting irrelevant information (e.g., subject changed in conversation).

---

### 🟡 Input Gate
**What it is:** LSTM gate that decides which new information to store in the cell state.
**Use Case:** Adding new relevant information to memory.

---

### 🟡 Output Gate
**What it is:** LSTM gate that decides what part of the cell state to output as the hidden state.
**Use Case:** Producing the relevant output at each time step.

---

### 🟡 Bidirectional RNN
**What it is:** Two RNNs — one processes sequence left-to-right, one right-to-left. Outputs are concatenated.
**Use Case:** BERT (bidirectional transformers), NER, sentiment analysis.

---

### 🟡 Sequence-to-Sequence (Seq2Seq)
**What it is:** Encoder reads an input sequence, decoder generates an output sequence.
**Use Case:** Machine translation, text summarization, chatbots.

---

### 🟡 Teacher Forcing
**What it is:** During training, feeding the correct output from the previous step as input to the next step (instead of the model's own prediction).
**Use Case:** Stabilizes and speeds up sequence model training.

---

### 🟡 Truncated BPTT (Backpropagation Through Time)
**What it is:** Backpropagation applied to unrolled RNNs across time steps, but truncated to a fixed number of steps to manage memory.
**Use Case:** Training RNNs and LSTMs on long sequences.

---

## 14. EVALUATION METRICS

---

### 📊 Accuracy
**Formula:** `Correct Predictions / Total Predictions`
**Use Case:** Classification tasks with balanced classes.

---

### 📊 Precision
**Formula:** `True Positives / (True Positives + False Positives)`
**Use Case:** When false positives are costly (spam filter, cancer screening).

---

### 📊 Recall (Sensitivity)
**Formula:** `True Positives / (True Positives + False Negatives)`
**Use Case:** When false negatives are costly (disease detection).

---

### 📊 F1 Score
**Formula:** `2 × (Precision × Recall) / (Precision + Recall)`
**Use Case:** Balances precision and recall — used for imbalanced datasets.

---

### 📊 AUC-ROC
**What it is:** Area Under the ROC Curve. Measures model's ability to distinguish between classes across all thresholds.
**Use Case:** Binary classification evaluation, especially with imbalanced data.

---

### 📊 Confusion Matrix
**What it is:** Table showing True Positives, True Negatives, False Positives, False Negatives.
**Use Case:** Detailed analysis of classification errors.

---

### 📊 Perplexity
**What it is:** Measures how well a language model predicts a sequence. Lower = better.
**Use Case:** Evaluating language model quality.

---

### 📊 BLEU Score
**What it is:** Measures overlap between generated text and reference text using n-gram matching.
**Use Case:** Machine translation evaluation.

---

### 📊 ROUGE Score
**What it is:** Measures recall-based overlap between generated and reference text.
**Use Case:** Summarization evaluation.

---

### 📊 Mean Average Precision (mAP)
**What it is:** Average of precision scores across all classes and IoU thresholds.
**Use Case:** Object detection evaluation (YOLO, Faster RCNN).

---

### 📊 IoU (Intersection over Union)
**What it is:** Measures overlap between predicted bounding box and ground truth box.
**Use Case:** Object detection, image segmentation.

---

## 15. ADVANCED TECHNIQUES

---

### 🔷 Attention Pooling
**What it is:** Uses attention weights to pool/aggregate features instead of simple averaging or max.
**Use Case:** Document classification, aggregating variable-length sequences.

---

### 🔷 Gradient Accumulation
**What it is:** Accumulates gradients over multiple batches before updating weights — simulates a larger batch size.
**Use Case:** Training large models on limited GPU memory.

---

### 🔷 Mixed Precision Training
**What it is:** Using FP16 (half precision) for most computations and FP32 for weight updates. Faster and less memory.
**Use Case:** Training LLMs and large CNNs efficiently.

---

### 🔷 Quantization
**What it is:** Reducing the precision of model weights (e.g., from 32-bit to 8-bit or 4-bit).
**Use Case:** Deploying models on mobile/edge devices (GGUF, AWQ, GPTQ formats).

---

### 🔷 Pruning
**What it is:** Removing unimportant weights or neurons from a trained network to make it smaller and faster.
**Use Case:** Model compression for deployment.

---

### 🔷 Neural Architecture Search (NAS)
**What it is:** Automated method to discover optimal neural network architectures.
**Use Case:** EfficientNet was found using NAS.

---

### 🔷 Meta-Learning (Learning to Learn)
**What it is:** Training a model to quickly adapt to new tasks with minimal data.
**Use Case:** Few-shot learning, MAML algorithm.

---

### 🔷 Curriculum Learning
**What it is:** Training on easy examples first, gradually increasing difficulty.
**Use Case:** Improves convergence and final performance.

---

### 🔷 Contrastive Learning
**What it is:** Learning representations by pulling similar samples together and pushing dissimilar ones apart in embedding space.
**Use Case:** Self-supervised learning — SimCLR, CLIP, MoCo.

---

### 🔷 Self-Supervised Learning
**What it is:** Learning representations from unlabeled data using automatically generated supervisory signals.
**Use Case:** BERT (masked language modeling), MAE (masked autoencoders).

---

### 🔷 LoRA (Low-Rank Adaptation)
**What it is:** Fine-tuning technique that only updates small low-rank matrices instead of all model weights. Very parameter-efficient.
**Use Case:** Fine-tuning LLMs cheaply — used widely for custom LLM adapters.

---

### 🔷 Prompt Tuning
**What it is:** Adding learnable soft tokens to the input instead of modifying model weights.
**Use Case:** Efficient adaptation of frozen LLMs.

---

### 🔷 Reinforcement Learning (RL)
**What it is:** Agent learns to make decisions by receiving rewards/penalties from an environment.
**Use Case:** Game playing (AlphaGo), robotics, RLHF for LLM alignment.

---

### 🔷 Ensemble Learning
**What it is:** Combining multiple models' predictions to get better accuracy.
**Types:** Bagging, Boosting, Stacking.
**Use Case:** Competition-winning solutions, reducing variance.

---

## 16. HARDWARE & INFRASTRUCTURE

---

### 💻 GPU (Graphics Processing Unit)
**What it is:** Parallel processing hardware with thousands of cores, ideal for matrix operations in deep learning.
**Use Case:** Training all deep learning models. NVIDIA A100, H100 most common.

---

### 💻 TPU (Tensor Processing Unit)
**What it is:** Google's custom chip specifically designed for tensor/matrix operations in neural networks.
**Use Case:** Training Google models (BERT, T5, PaLM) at scale.

---

### 💻 VRAM (Video RAM)
**What it is:** Memory on the GPU that stores model weights, activations, and gradients during training.
**Use Case:** Larger VRAM = can train bigger models or larger batches.

---

### 💻 CUDA
**What it is:** NVIDIA's parallel computing platform for GPU programming.
**Use Case:** PyTorch, TensorFlow run on CUDA under the hood.

---

### 💻 Tensor
**What it is:** A multi-dimensional array — the core data structure of deep learning frameworks.
**Examples:** Scalar (0D), Vector (1D), Matrix (2D), 3D/4D tensors.
**Use Case:** Storing images (4D: batch×channel×height×width), text (2D: batch×sequence).

---

### 💻 PyTorch
**What it is:** Open-source deep learning framework by Meta. Dynamic computation graph.
**Use Case:** Research, most LLM training, flexible model building.

---

### 💻 TensorFlow / Keras
**What it is:** Open-source deep learning framework by Google. Static/dynamic computation graphs.
**Use Case:** Production deployment, mobile ML, enterprise.

---

### 💻 ONNX (Open Neural Network Exchange)
**What it is:** Open format for representing deep learning models across different frameworks.
**Use Case:** Convert PyTorch model → ONNX → deploy anywhere.

---

## 17. DATA TERMS

---

### 📁 Training Set
**What it is:** The data used to train (fit) the model.
**Typical Split:** 70–80% of data.

---

### 📁 Validation Set
**What it is:** Data used to tune hyperparameters and monitor training — not used in weight updates.
**Typical Split:** 10–15% of data.

---

### 📁 Test Set
**What it is:** Data used only ONCE at the end to evaluate final model performance.
**Typical Split:** 10–20% of data.

---

### 📁 Cross-Validation
**What it is:** Splitting data into K folds, training K times each using a different fold as validation.
**Use Case:** Better performance estimate when data is limited.

---

### 📁 Feature
**What it is:** An individual measurable property or attribute of the data.
**Use Case:** For a house price model: square footage, bedrooms, location = features.

---

### 📁 Label / Target
**What it is:** The correct output the model should predict.
**Use Case:** In image classification: the class label ("cat", "dog").

---

### 📁 One-Hot Encoding
**What it is:** Converts categorical labels into binary vectors. Only one position is "1".
**Example:** [Cat, Dog, Bird] → Cat=[1,0,0], Dog=[0,1,0], Bird=[0,0,1]
**Use Case:** Input to neural networks, classification targets.

---

### 📁 Normalization / Standardization
**What it is:** Scaling input features to a consistent range before feeding to the model.
- **Min-Max:** Scale to [0, 1]
- **Z-Score:** Mean=0, Std=1
**Use Case:** Improves training stability and speed.

---

### 📁 Dimensionality Reduction
**What it is:** Reducing the number of features while preserving important information.
**Methods:** PCA, autoencoders, t-SNE, UMAP.
**Use Case:** Visualization, removing noise, faster training.

---

## 🌟 QUICK REFERENCE SUMMARY

| Category | Key Terms |
|---|---|
| **Fundamentals** | AI, ML, DL, ANN, DNN, Parameter, Hyperparameter |
| **Network Parts** | Neuron, Weight, Bias, Layer, Forward/Backprop |
| **Activations** | ReLU, Sigmoid, Tanh, GELU, Softmax |
| **Architectures** | CNN, RNN, LSTM, Transformer, GAN, VAE, ResNet |
| **Training** | Epoch, Batch, Learning Rate, Overfitting, Transfer Learning |
| **Loss Functions** | MSE, Cross-Entropy, Focal Loss, KL Divergence |
| **Optimizers** | SGD, Adam, AdamW, RMSprop |
| **Regularization** | Dropout, L1/L2, Early Stopping, Label Smoothing |
| **Normalization** | BatchNorm, LayerNorm, RMSNorm |
| **Transformers** | Attention, Q/K/V, Positional Encoding, KV Cache |
| **Metrics** | Accuracy, F1, AUC-ROC, BLEU, Perplexity |
| **Advanced** | LoRA, Quantization, Pruning, RLHF, Contrastive Learning |
| **Hardware** | GPU, TPU, CUDA, PyTorch, TensorFlow, Tensor |
