'''
Here's a deep dive into every major flaw and the real-world solutions being used to address them.

1. Hallucination
    This is the most famous flaw. The model confidently generates text that sounds correct but is factually wrong — fabricated citations, wrong dates, made-up names. It happens because the model is trained to produce plausible next tokens, not true ones. There is no internal "fact checker" — only statistical patterns.

    Fixes being deployed today:

        Retrieval-Augmented Generation (RAG) — before answering, the model fetches real documents and is instructed to only use those as its source. This grounds answers in verifiable content.

        Citation forcing — the model is trained to cite a source for every claim, making errors auditable.

        Self-consistency sampling — run the model multiple times on the same question and take the answer that appears most often. Hallucinated outputs tend to be inconsistent across runs.


2. Context window limit
    An LLM can only "see" a fixed window of text at a time — typically tens of thousands to a few hundred thousand tokens. Beyond that, it simply doesn't have access to earlier content. It cannot natively remember your conversation from last week or process an entire codebase at once.
    
    Fixes:

        Extended context windows — models like Claude and Gemini have pushed windows to 1M+ tokens, fitting entire books in a single prompt.
        
        External memory systems — key facts from long interactions are stored in a vector database and retrieved on demand.
        
        Summarization chains — long documents are compressed into rolling summaries that fit within the window.


3. Stale knowledge (knowledge cutoff)
    Training data has a cutoff date. The model has no awareness of anything that happened after it was trained — no news, no price changes, no elections, no new releases. It also can't browse the web or query databases on its own.
    
    Fixes:

            Tool use / function calling — the model is given tools (web search, calculator, database lookup) and learns to call them when it needs fresh information. This is how Claude's web search works.
            
            Continual fine-tuning — some deployments periodically retrain on new data, though this is expensive and incomplete.


4. Bias and toxicity
    Because LLMs are trained on internet-scale text, they absorb every bias present in that text — gender stereotypes, racial associations, political slants, harmful content. Without intervention, models reflect and can amplify these patterns.
    
    Fixes:

        RLHF (Reinforcement Learning from Human Feedback) — human raters evaluate model outputs and reward safer, more neutral responses. This is the main technique used by OpenAI, Anthropic, and Google to align models.
        
        Constitutional AI (Anthropic's approach) — the model is given a set of principles and trained to critique and revise its own outputs against them.
        
        Data curation — toxic, biased, or low-quality text is filtered from training data before training begins.


5. Shallow reasoning (no true logic)
    LLMs are extraordinarily good at pattern matching and surface-level language, but they don't reason from first principles the way humans do. They can fail badly on simple arithmetic, multi-step logic puzzles, or problems that require holding a formal model of the world. They are interpolating between training examples, not executing a proof.
    
    Fixes:

        Chain-of-thought (CoT) prompting — instructing the model to "think step by step" dramatically improves reasoning by forcing it to externalise its working. This unlocks latent capability that exists in the weights but doesn't surface by default.
        
        Scratchpad / extended thinking — newer models are trained to spend tokens on internal reasoning before outputting an answer (like Claude's extended thinking mode).
        
        Tool-augmented agents — offload formal reasoning to real tools (Python interpreter, symbolic solver, calculator). The model orchestrates; the tool computes.
        
        Process reward models — instead of only rewarding a correct final answer, these train the model to produce correct reasoning steps, penalising wrong intermediate logic.


6. High compute cost
    Training a frontier LLM costs tens to hundreds of millions of dollars. Running inference on one is also expensive — each token requires a massive matrix multiplication across billions of parameters. This makes LLMs inaccessible without significant cloud infrastructure.
    
    Fixes:

    Quantization — reduce the numerical precision of model weights from 32-bit floats to 8-bit or even 4-bit integers. This shrinks the model significantly with modest quality loss.
    
    Model distillation — train a small "student" model to mimic the outputs of a large "teacher" model. The student is much cheaper to run but captures much of the capability.
    
    Mixture of Experts (MoE) — instead of activating all parameters for every token, only a small subset of "expert" sub-networks fires per token. GPT-4 and Gemini use this architecture. You get a huge model's capacity at a fraction of the inference cost.
    
    Speculative decoding — a tiny draft model generates candidate tokens cheaply; the large model verifies them in parallel. Speeds up inference without any quality loss.


The honest summary is that most of these fixes are workarounds rather than fundamental solutions — they make LLMs dramatically more useful while the deeper architectural research continues. The hallucination problem in particular has no clean solution yet; RAG helps but doesn't eliminate it. True reasoning remains the hardest open problem in the field.'''