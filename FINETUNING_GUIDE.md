# Fine-Tuning, Explained Without the Math

A beginner-friendly guide to how fine-tuning works, the different approaches, and how to decide whether it will help your project.

**Scope:** Mostly large language models (LLMs), with sections on search, images, and speech. This covers the main ideas and practical decisions rather than every research variant. No calculus or linear algebra required. Analogies explain the intuition; they are not literal descriptions of a model's mind.

**Reference check:** September 6, 2026. Tool APIs and hardware support change; the underlying concepts below are more stable. Dataset sizes and experimental settings are illustrations, not universal recipes.

## Contents

1. [What fine-tuning means](#1-what-fine-tuning-means)
2. [What is inside a model?](#2-what-is-inside-a-model)
3. [How learning happens](#3-how-learning-happens)
4. [Pretraining, post-training, and inference](#4-pretraining-post-training-and-inference)
5. [Fine-tuning versus prompting, RAG, and tools](#5-fine-tuning-versus-prompting-rag-and-tools)
6. [A map of fine-tuning types](#6-a-map-of-fine-tuning-types)
7. [Learning from examples: SFT](#7-learning-from-examples-sft)
8. [Learning a domain through continued pretraining](#8-learning-a-domain-through-continued-pretraining)
9. [Learning preferences and rewards](#9-learning-preferences-and-rewards)
10. [Full, partial, and parameter-efficient tuning](#10-full-partial-and-parameter-efficient-tuning)
11. [LoRA and QLoRA](#11-lora-and-qlora)
12. [Other methods and related concepts](#12-other-methods-and-related-concepts)
13. [Preparing good training data](#13-preparing-good-training-data)
14. [Training settings in plain English](#14-training-settings-in-plain-english)
15. [Hardware, memory, and cost](#15-hardware-memory-and-cost)
16. [How to tell whether it worked](#16-how-to-tell-whether-it-worked)
17. [Common failures and how to investigate them](#17-common-failures-and-how-to-investigate-them)
18. [A worked example from start to finish](#18-a-worked-example-from-start-to-finish)
19. [What I would do with 1,000 PDFs](#19-what-i-would-do-with-1000-pdfs)
20. [Fine-tuning a RAG system](#20-fine-tuning-a-rag-system)
21. [Fine-tuning beyond language models](#21-fine-tuning-beyond-language-models)
22. [Saving, deploying, and maintaining the result](#22-saving-deploying-and-maintaining-the-result)
23. [Frequently asked questions](#23-frequently-asked-questions)
24. [A practical learning path](#24-a-practical-learning-path)
25. [Glossary](#25-glossary)
26. [Sources and further reading](#26-sources-and-further-reading)

## 1. What fine-tuning means

**Fine-tuning means taking an already trained model and training it further for a particular purpose.**

Imagine hiring someone who already writes well, then showing them examples of your company's best support replies. They already know the language. You want them to learn your format, terminology, and approach to answering customers.

A model is not a person, but this captures why fine-tuning is useful: you start with existing capabilities instead of building everything from scratch.

For example, a general model might answer:

> Your request sounds like a billing problem. You should contact the billing team for assistance.

After suitable training, it might instead produce:

```json
{"category":"billing","priority":"normal","needs_human":true}
```

Training makes the desired behavior more likely across similar inputs. It does not guarantee that every output will be correct.

**The lasting change is in learned numerical parameters.** Depending on the method, these are the original model's parameters or additional parameters attached to it. Merely uploading files to a chat or putting examples in a prompt does not necessarily train anything. [Hugging Face fine-tuning overview](https://huggingface.co/docs/transformers/en/training)

## 2. What is inside a model?

You only need a few building blocks to follow the rest of this guide.

### Tokens: the pieces of input and output

A language model processes **tokens**. A token might be a word, a piece of a word, punctuation, or another encoded text fragment. Tokens and words are not interchangeable.

A **tokenizer** converts text into token IDs and converts generated IDs back into text. Languages, code, and unusual text can use very different numbers of tokens for the same apparent length.

### Parameters or weights: learned numerical settings

Think of a model as a machine with a huge number of adjustable settings. Its **weights** help determine how input turns into output.

A “7B” model has approximately seven billion parameters. This does not mean it stores seven billion facts. Knowledge and behavior are distributed across many interacting parameters.

The knobs analogy has a limit: there is generally no single “politeness knob” or “Paris fact knob.” Training adjusts many interacting numbers.

### Layers: successive stages of processing

A neural network processes information through layers. Each stage transforms the current representation before passing it along.

In a transformer, **attention** helps combine information from relevant positions in the available context. Other components also transform that information. Fine-tuning changes how these computations behave; it usually does not redesign the model's architecture.

### Embeddings: numerical representations

An embedding represents something using a list of numbers. Inside an LLM, token embeddings are an early representation of tokens.

A separate search embedding model can represent whole passages or queries so software can compare them. These two uses are related, but a search embedding system is not the same thing as the token embedding layer inside a chatbot.

### Context: what the model can see right now

The **context window** is the amount of input and generated content the model can handle within a request, subject to its implementation and limits.

Context is temporary working input. Weights are persistent learned settings. A model can use a fact from its current context without learning it permanently.

## 3. How learning happens

Here is the basic supervised training loop:

```text
Training example
      |
      v
Model predicts the desired output
      |
      v
A loss measures how poorly the prediction matched the target
      |
      v
Backpropagation calculates how trainable settings affect that loss
      |
      v
An optimizer makes a small adjustment
      |
      v
Repeat with many examples; check performance on unseen examples
```

### A small example

Suppose the task is classifying messages:

```text
Input:  "I was charged twice."
Target: "billing"
```

The model initially gives too little probability to the correct target. Training adjusts its settings so that the target becomes more likely in this context.

With varied examples, the model may also learn to classify:

```text
"My card shows two payments for the same order."
```

That is **generalization**: applying a learned pattern to something new.

If it only succeeds on the exact sentences it saw during training, it has mostly memorized them. That is much less useful.

### The math words translated

| Term | Plain-language meaning |
|---|---|
| Forward pass | Run the model to make predictions. |
| Loss | A numerical measure of how badly predictions match the training objective. |
| Gradient | Information about which small parameter changes would increase or decrease loss locally. |
| Backpropagation | An efficient way to calculate those gradients through the network. |
| Optimizer | The procedure that uses gradients and often past updates to adjust parameters. |
| Gradient descent | The broad idea of taking steps in a direction that reduces loss. |
| Learning rate | How large the updates are allowed to be. |

You do not need to calculate a gradient to use fine-tuning. You do need to understand that the training objective determines what the system rewards.

### How this works for a text answer

In ordinary causal language-model SFT, training rewards the model for assigning probability to the correct next token, repeatedly across the target response.

During training, the model normally sees the correct earlier target tokens when predicting the next one. This is called **teacher forcing**. During use, it has to continue from its own generated tokens, so one mistake can affect later output.

Training implementations can calculate many of these token predictions in parallel. The system does not necessarily generate a full answer, read it, and then receive a human-style grade.

### Why a lower loss is not the whole goal

The loss measures the objective you implemented. It does not automatically measure truth, usefulness, good judgment, or customer satisfaction.

A model can improve at imitating flawed examples. It can also get very good at a narrow training set while becoming worse on real requests. This is why evaluation is part of training, not an optional final decoration.

## 4. Pretraining, post-training, and inference

These terms describe different stages or activities.

| Term | What happens | Analogy |
|---|---|---|
| Pretraining | Learn broad patterns from a large dataset, often using targets derived from the data itself. | General education. |
| Continued pretraining | Continue broad language-model training on additional data, often from a domain. | Extensive reading in a specialty. |
| Supervised fine-tuning | Learn desired outputs from examples of tasks. | Practice with worked examples. |
| Preference or reward training | Adjust behavior using comparisons or scores. | Coaching based on which attempts worked better. |
| Post-training | An umbrella for adaptation after initial pretraining, often including SFT and preference/reward methods. | Further preparation for a role. |
| Inference | Use the resulting model to produce predictions or responses. | Doing the job. |

These are not mandatory steps in one fixed sequence. Terminology also varies between papers and providers.

A **base model** is typically a pretrained model before instruction-oriented post-training. An **instruct/chat model** has already received training intended to make it respond to instructions or conversations.

For a focused assistant experiment, starting from a capable instruct model is often a reasonable practical choice. Starting from a base model can require more work to establish basic assistant behavior. The choice should still be tested on the intended task.

**Transfer learning** is the broader idea of reusing what a model learned for another task. Fine-tuning is one way to do that.

## 5. Fine-tuning versus prompting, RAG, and tools

These solve different problems and can be combined.

| Approach | What changes? | Useful when… | Main limitation |
|---|---|---|---|
| Prompting | Instructions in the current request. | You can describe the behavior clearly. | Instructions consume context and may not be followed consistently. |
| Few-shot prompting | Add worked examples to the request. | A few examples communicate the pattern. | Examples use context and must be supplied when needed. |
| RAG | Retrieve relevant information and provide it with the request. | Answers depend on documents or changing knowledge. | Bad retrieval can omit the necessary evidence. |
| Tools | Allow the application to call search, code, databases, or services. | You need exact calculations, live state, or actions. | Tool access and results need reliable application handling. |
| Fine-tuning | Persistently adjust learned parameters. | You need repeatable behavior across many requests. | Requires data, training, evaluation, and maintenance. |

**RAG** means retrieval-augmented generation. The application finds relevant material, then the model answers using it. The foundational RAG work combines retrieval with generation to support knowledge-intensive tasks. [RAG paper](https://arxiv.org/abs/2005.11401)

### A useful rule of thumb

- “Use this tone and output structure repeatedly” suggests prompting, then possibly fine-tuning.
- “Answer from the current employee handbook” suggests retrieval.
- “Tell me the current order status” suggests a database or API tool.
- “Do all three” suggests a combined system.

These are design heuristics, not hard boundaries. Fine-tuning can teach facts, and RAG can provide examples of behavior. The question is which mechanism gives you the reliability and update process you need.

### Why training on documents does not create a perfect database

Training compresses patterns into parameters. It does not guarantee exact retrieval of each sentence, correct citations, or selective replacement of outdated facts.

For example, training on last year's refund policy may make outdated answers more likely. Retrieving the currently approved policy makes the evidence easier to update and inspect.

Fine-tuning can still teach the assistant **how to use** that evidence: cite it, avoid unsupported claims, and say when the answer is missing.

### Diagnose before choosing a method

If a system fails, ask what was missing:

1. Did it lack the information?
2. Did it have the information but misuse it?
3. Did it understand the task but use the wrong format?
4. Was the required operation outside the model's capabilities?

These failures can look similar to a user but need different fixes.

## 6. A map of fine-tuning types

The labels become much easier once you separate two questions.

**Question A: What learning signal do you use?**

| Family | Training material | What the model practices |
|---|---|---|
| SFT | Inputs with desired outputs. | Producing a good answer. |
| Continued pretraining | Raw text or other domain data. | Modeling the domain's patterns. |
| Preference optimization | Preferred and less-preferred responses. | Favoring one kind of response over another. |
| Reinforcement learning | Generated attempts with rewards. | Producing attempts that earn higher scores. |

**Question B: Which parameters do you train?**

| Family | What gets adjusted |
|---|---|
| Full fine-tuning | All or essentially all model parameters. |
| Partial fine-tuning | Selected layers or components. |
| Head-only training | A task-specific output component. |
| Parameter-efficient fine-tuning, or PEFT | A small subset of parameters or added trainable components. |

**SFT and LoRA are not alternatives.** SFT describes the learning objective. LoRA describes how you represent the trainable changes.

You can do SFT with LoRA, preference training with LoRA, or SFT with full fine-tuning. QLoRA additionally changes how the frozen base weights are stored during training.

## 7. Learning from examples: SFT

**Supervised fine-tuning**, or **SFT**, teaches by demonstration: “For this input, this is a desirable output.”

It can train summarization, extraction, classification, conversational responses, and structured outputs. **Instruction tuning** is SFT organized around following instructions, often across multiple tasks.

Conceptual example:

```json
{
  "messages": [
    {"role": "system", "content": "Classify support requests. Return a category only."},
    {"role": "user", "content": "I cannot sign in after resetting my password."},
    {"role": "assistant", "content": "account_access"}
  ]
}
```

This illustrates a conversation record; the exact accepted schema depends on the trainer.

Training can score just assistant outputs or include other tokens, depending on configuration. **Loss masking** decides which token positions count toward the training loss. Masking user tokens does not hide the user's message from the model; it changes which predictions are graded. [TRL SFT documentation](https://huggingface.co/docs/trl/en/sft_trainer)

### Demonstrations teach more than you intend

If every answer begins with an apology, the model may apologize unnecessarily. If every example answers confidently, it may fail to express uncertainty. If every answer is long, it may become verbose.

Include examples that demonstrate decisions, not only wording:

- Clear requests with direct answers.
- Ambiguous requests that require clarification.
- Requests where the supplied evidence is insufficient.
- Valid edge cases and unusual phrasing.
- Multi-turn conversations if those occur in the real application.

The model learns recurring patterns. Your dataset should make the intended patterns easier to learn than accidental shortcuts.

## 8. Learning a domain through continued pretraining

**Continued pretraining** trains on additional text using a language-modeling objective. The material might be scientific writing, technical manuals, or domain-specific code rather than question-answer pairs.

**Domain-adaptive pretraining**, often shortened to **DAPT**, focuses on a subject area. **Task-adaptive pretraining**, or **TAPT**, uses unlabeled text more closely related to a particular task.

The intuition is repeated exposure to a specialty's vocabulary, style, and relationships. Research has found benefits from such adaptation in studied settings; it does not guarantee a gain on every model or dataset. [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964)

This does not directly demonstrate how an assistant should respond to a user. A project might use domain adaptation followed by SFT, but the extra stage adds work and may be unnecessary.

For a first experiment, establish whether the existing model actually struggles with the domain. If it already understands the material when given context, evidence access or task demonstrations may be the more useful intervention.

## 9. Learning preferences and rewards

### Preference data

Sometimes you cannot write one perfect response, but you can judge which of two responses is better.

```text
Prompt: Explain this error to a beginner.

Preferred: Plain-language explanation with one useful next step.
Less preferred: Unexplained jargon and an unsupported diagnosis.
```

“Less preferred” need not mean completely false. Two answers can both be valid while one better matches the evaluation criteria.

### DPO: Direct Preference Optimization

DPO trains on preferred/rejected response pairs. In its standard form, it uses a reference policy to guide how relative preferences change. It avoids separately fitting an explicit reward model and running the classic reinforcement-learning loop.

The intuition is: increase the relative preference for the better answer, with the reference helping anchor the change. It is not simply SFT with a negative label attached to the bad answer. [DPO paper](https://arxiv.org/abs/2305.18290)

Preference labels need consistent criteria. If annotators mostly favor longer answers, the system may learn length rather than usefulness. [TRL DPO documentation](https://huggingface.co/docs/trl/en/dpo_trainer)

### RLHF: Reinforcement Learning from Human Feedback

A classic pipeline is:

1. Train an assistant with demonstrations.
2. Collect human comparisons of responses.
3. Train a **reward model** to predict human preferences.
4. Train the assistant to generate responses that score well, usually with controls on how far behavior changes.

The reward model is a learned evaluator, not an oracle of truth. **PPO**, or Proximal Policy Optimization, is one algorithm used to update the assistant in this kind of pipeline. [InstructGPT research](https://arxiv.org/abs/2203.02155)

People sometimes use “RLHF” broadly for preference-based alignment, including methods that do not run a classic RL loop. Check what a particular author means.

### RLAIF: AI feedback

**Reinforcement Learning from AI Feedback** uses AI-generated judgments for some feedback. This describes where feedback comes from more than one fixed optimizer.

It can reduce human labeling effort, but a model judge can carry systematic biases or miss errors. AI approval is evidence to examine, not proof of correctness.

### Verifiable rewards and GRPO

Some tasks have automatically checkable outcomes: a program passes tests, a puzzle satisfies rules, or an answer matches a reliable verifier. This is often described as **reinforcement learning with verifiable rewards**, or **RLVR**.

**GRPO**, Group Relative Policy Optimization, samples several responses for a prompt and uses their relative rewards to guide learning. In its usual formulation it avoids a separately learned value model, though it still needs generation, scoring, and training infrastructure. [TRL GRPO documentation](https://huggingface.co/docs/trl/en/grpo_trainer)

### Reward hacking

A reward is a proxy for what you want. A model may find a way to score well without doing the intended job.

For example, if your evaluator rewards the presence of citations without checking them, fabricated citations may score well. If tests cover only easy cases, code can pass while remaining wrong elsewhere.

Design the grader carefully and evaluate with checks that were not used as the training reward. Advanced optimization cannot repair an objective that consistently rewards the wrong outcome.

## 10. Full, partial, and parameter-efficient tuning

### Full fine-tuning

Full fine-tuning updates the whole model. It gives training considerable flexibility, but requires storing and updating many parameters and their associated training state.

It may be appropriate when substantial adaptation is needed and resources allow it. More flexibility does not guarantee a better result: data, optimization, and evaluation still matter.

### Partial fine-tuning and head-only training

You can freeze most of the network and train selected layers. **Frozen** means those parameters do not receive optimizer updates.

In image classification, for example, you can reuse a pretrained visual network and train a new output head for your categories. Later, you might unfreeze more layers if the task benefits. [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

### PEFT

**Parameter-efficient fine-tuning** is a family of methods that trains relatively few parameters, such as added adapters or selected existing weights. The pretrained model supplies most of the computation and capability.

PEFT can greatly reduce the storage and training-state cost per task. It does not eliminate the need to load and run the underlying model. [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/en/index)

## 11. LoRA and QLoRA

### LoRA: learn a compact adjustment

**LoRA** means **Low-Rank Adaptation**. It freezes the original weights and trains smaller components that represent changes to selected weight matrices.

Imagine a large mixing desk with thousands of controls. For one task, many useful changes might be coordinated. Rather than separately adjusting every control, you learn a smaller set of coordinated adjustments.

That is an intuition for “low rank”: restrict the update to a more compact structure. The actual implementation uses two smaller numerical tables, or matrices, whose combination produces an update to a larger one. [LoRA paper](https://arxiv.org/html/2106.09685v2)

```text
Frozen base computation ──────┐
                             ├── Combined result
Trainable LoRA adjustment ────┘
```

### The main LoRA settings

| Setting | Intuition |
|---|---|
| Rank, `r` | Capacity of the compact update. Larger ranks use more trainable parameters. |
| Alpha | Scales the adapter's contribution; its interaction with rank depends on the variant. |
| Target modules | Which model components receive adapters. |
| Dropout | A training regularizer used by some configurations. |

Higher rank is not automatically better. A poorly prepared dataset remains poor at any rank.

A LoRA adapter generally requires the matching base model. It is a learned adjustment, not a complete independent chatbot. Compatible adapters can often be merged into base weights for serving, though quantization and deployment details affect the workflow. [PEFT LoRA guide](https://huggingface.co/docs/peft/main/conceptual_guides/lora)

### QLoRA: store the base more compactly

**Quantization** represents numbers using fewer bits. Think of recording measurements with fewer digits: it saves space while introducing approximation.

**QLoRA** combines a frozen, typically 4-bit base model with trainable LoRA adapters. Its core contribution is making adaptation use much less memory. Computation and trainable adapter values can use higher precision; “4-bit training” does not mean every operation uses 4-bit numbers. [QLoRA paper](https://arxiv.org/abs/2305.14314)

QLoRA is distinct from **quantization-aware training**, where training explicitly accounts for a target quantized computation. The words are easy to confuse.

### Comparing the choices

| Approach | Base weights updated? | Extra trainable adapters? | Practical emphasis |
|---|---|---|---|
| Full tuning | Yes | Usually no | Broad flexibility, more training state. |
| LoRA | No, in the standard setup | Yes | Compact task-specific updates. |
| QLoRA | No; base stored in low precision | Yes | Further base-weight memory savings. |

A smaller adapter does not mean proportionally faster training. The base model still participates in forward and backward computation. Actual speed depends on hardware, kernels, sequence lengths, and the training setup. [PEFT quantization guide](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/quantization.md)

## 12. Other methods and related concepts

These names are useful to recognize; you do not need to master them before running SFT.

| Term | High-level idea |
|---|---|
| Bottleneck adapters | Insert small trainable modules between existing computations. |
| Soft prompt tuning | Learn numerical prompt vectors while keeping the base model frozen. These are not ordinary readable instructions. |
| Prefix tuning | Learn conditioning vectors that influence internal attention computations. |
| IA3 | Learn compact scaling vectors that adjust selected internal activations. |
| DoRA | A LoRA-related approach that treats weight magnitude and direction separately. |

These are examples within a broader and growing family of adaptation methods. [PEFT method catalog](https://huggingface.co/docs/peft/en/index), [PEFT LoRA variants](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/lora.md)

### Distillation

A **teacher** model supplies outputs or other learning signals to a **student** model, often smaller. The student learns to reproduce useful parts of the teacher's behavior.

You might generate candidate demonstrations with a strong model, review them, and fine-tune a smaller model. Distillation describes the source and transfer of the signal; it can use SFT as the training procedure.

Errors can transfer too. Student quality should be measured independently of whether it agrees with the teacher.

### Multi-task and continual learning

**Multi-task learning** mixes several tasks in training. This can help preserve breadth, but task balance matters: an abundant easy task can dominate a smaller important one.

**Continual learning** concerns adaptation over time while retaining useful earlier abilities. Ordinary sequential fine-tuning does not automatically solve this retention problem.

### Model merging

Merging combines compatible model updates or weights. It is different from an ensemble, which runs multiple models and combines outputs. Merging is not guaranteed to combine the best skills cleanly; updates can interfere.

## 13. Preparing good training data

Your data is the curriculum. Training settings determine how the model studies it, but cannot make contradictory lessons coherent.

### Write down the task first

A useful task definition specifies:

- The input the deployed model will actually receive.
- The desired output and any required format.
- What counts as a mistake.
- What to do when information is absent or ambiguous.
- Which cases are outside scope.

For example: “Given a customer message, choose one supported routing category, or `needs_review` if no category is justified.” This is easier to train and evaluate than “be better at customer service.”

### Match training to real use

If deployment includes retrieved passages, include passages in training. If deployment includes conversation history, train on realistic histories. If customers write fragments and typos, do not train exclusively on polished textbook questions.

Do not let training examples contain clues unavailable during use. A hidden category name in a filename or a target accidentally included in the prompt can make performance look excellent for the wrong reason.

### Clean and review

Remove duplicate and near-duplicate examples, fix incorrect labels, reconcile inconsistent instructions, and inspect unusually long or short records. Use data you are authorized to train on and exclude secrets that the model should never reproduce.

Synthetic data needs the same review as human data. Thousands of fluent but wrong examples can teach a systematic error very efficiently.

### Training, validation, and test sets

| Set | Purpose | Analogy |
|---|---|---|
| Training | Used to update parameters. | Practice exercises. |
| Validation/development | Used to choose settings and checkpoints. | Practice exam used to guide study. |
| Test | Used for a final independent comparison. | Sealed final exam. |

An 80/10/10 split is an illustrative starting point, not a rule. A small or heavily imbalanced dataset may need a different design.

**Split related material together.** Questions from the same document, records from the same customer, or paraphrases of one example should not casually cross the boundary between training and testing.

For document question answering, split source documents before generating question-answer pairs. Otherwise, a test question can be “new” while its underlying answer has already appeared repeatedly in training.

If the intended use is answering future requests, a time-based holdout can be more informative than a random split.

### How much data do you need?

There is no universal minimum. A narrow formatting task and broad domain adaptation have very different requirements.

A practical experiment is to train on increasing subsets of a reviewed dataset and plot held-out quality against data size. If performance improves steadily, more coverage may help. If it stalls, inspect failure types before collecting more of the same examples.

Count tokens and distinct situations as well as rows. Ten thousand near-identical examples do not provide ten thousand different lessons.

### Chat templates and special tokens

A **chat template** serializes messages into the exact token layout expected by the model, including role boundaries. A mismatch between training and inference can break behavior even if the text looks reasonable.

Use the model's appropriate tokenizer and template consistently, and inspect the resulting training text. End-of-message and end-of-sequence tokens also affect when the model stops. [Transformers chat templates](https://huggingface.co/docs/transformers/en/chat_templating)

## 14. Training settings in plain English

These settings are called **hyperparameters**: choices you set, rather than weights learned from data.

| Setting | What it means | What can go wrong |
|---|---|---|
| Learning rate | Size of parameter updates. | Too large can destabilize training; too small can learn too slowly. |
| Batch size | Examples or token workload processed together. | Larger batches need more memory and change optimization behavior. |
| Gradient accumulation | Combine gradients from several small batches before updating. | Helps fit training in memory but adds work before each update. |
| Epochs | Passes through the training dataset. | Repeated exposure can eventually overfit. |
| Maximum steps | Limit on optimizer updates. | Too few may undertrain; too many may waste resources or overfit. |
| Sequence length | Maximum length processed per example or packed sequence. | Truncation can remove the answer; long sequences can be expensive. |
| Warmup | Gradually raise the learning rate at the beginning. | An unsuitable schedule can make early updates unhelpful. |
| Learning-rate schedule | How update size changes over training. | A schedule must fit the actual run length. |
| Weight decay | Regularization that discourages some weight growth. | Too much can impair learning. |
| Dropout | Temporarily disable some activations during training. | Too much can make learning harder. |
| Gradient clipping | Limit unusually large gradients. | Can stabilize spikes but does not fix incorrect data or code. |
| Evaluation interval | How often to check held-out performance. | Too infrequent can miss a useful stopping point. |
| Checkpoint interval | How often to save training state. | Frequent saves use time and disk; sparse saves reduce recovery options. |

### A tiny counting example

Suppose there are 800 examples and an effective batch of 8 examples per update. Roughly 100 updates make one epoch, and three epochs make roughly 300 updates.

Packing, distributed sampling, partial batches, and token-based batching can change the exact count. The example is just to connect “examples,” “batches,” and “steps.”

### Padding, packing, and masking

**Padding** adds filler tokens so examples can share tensor shapes. Padding tokens should normally be excluded from the relevant loss.

**Packing** puts multiple shorter examples into a longer sequence to use space efficiently. Pay attention to example boundaries and whether the implementation allows attention across unrelated examples.

**Truncation** cuts an example to fit. Always inspect what gets cut: removing the answer while keeping the prompt can quietly ruin the intended learning signal.

### Change one thing for a reason

Begin with a documented configuration compatible with the chosen model and trainer. Then change settings in response to evidence. Changing learning rate, data, adapter rank, and template simultaneously makes it hard to know why the result changed.

## 15. Hardware, memory, and cost

### Why training needs more memory than loading a model

Training may need space for:

- Model weights.
- Gradients for trainable parameters.
- Optimizer state used to calculate updates.
- Intermediate activations saved for backpropagation.
- Temporary buffers and framework overhead.

LoRA reduces much of the trainable parameter state. Quantizing the base reduces its storage. Neither removes all activation memory.

For a rough **weights-only** illustration, seven billion parameters at two bytes each need about 14 GB in decimal units. Four bits each would be about 3.5 GB before quantization metadata and other overhead. Neither number is a full training-memory requirement. [Transformers memory discussion](https://huggingface.co/docs/transformers/en/llm_tutorial_optimization)

### Common memory techniques

| Technique | Intuition | Tradeoff |
|---|---|---|
| Smaller microbatches | Process fewer examples at once. | More accumulation may be needed. |
| Shorter sequences | Store and compute less context. | May exclude useful task information. |
| Mixed precision | Use appropriate lower-precision arithmetic for parts of training. | Hardware and numerical stability matter. |
| Gradient checkpointing | Recompute some intermediate results instead of saving them. | Less memory, more computation. |
| Sharding | Split model/training state across devices. | More coordination and communication. |
| Offloading | Move some state to CPU memory or storage. | Transfers can slow training substantially. |

**VRAM** is accelerator memory. System RAM and disk are different resources; extra disk space does not by itself make a model fit in VRAM.

### Estimate the whole project

Cost includes data creation, review, experimental runs, evaluation, storage, engineering, and ongoing inference. A cheaper training run can still produce an expensive serving system.

Run a short pilot to measure actual memory and throughput on the intended configuration. Use those measurements to estimate the full run, leaving room for evaluation and checkpoint overhead. Avoid trusting a generic “model X fits on GPU Y” claim without checking sequence length and training settings.

## 16. How to tell whether it worked

**Compare against the starting model using the same held-out tasks and deployment conditions.** Otherwise you cannot attribute a gain to training.

Useful comparisons include the original model with a reasonable prompt, a few-shot prompt, and retrieval where the task requires evidence. A tuned model should earn its added complexity.

### Choose metrics that match the job

| Task | Useful measurements |
|---|---|
| Classification | Accuracy, per-class precision/recall, confusion matrix. |
| Structured extraction | Valid format, correct fields, missing or invented values. |
| Summarization | Supported claims, coverage, misleading omissions. |
| Document QA | Answer correctness, citation support, appropriate abstention. |
| Coding | Independent tests, robustness on edge cases. |
| Search | Whether relevant material appears near the top. |
| Assistant behavior | Human preference under a clear rubric, task success, regressions. |

**Precision** asks: “When the model chose this category, how often was it right?” **Recall** asks: “Of all real cases in this category, how many did it find?”

**Perplexity** measures how well a language model predicts text under a particular evaluation setup. Lower is better for that prediction objective; it does not establish better instruction-following or factuality.

### Check behavior, not only averages

An average can hide failure on rare but important categories. Inspect ambiguous inputs, long examples, missing evidence, unusual wording, and requests outside the training distribution.

Test retained capabilities too. A model that becomes excellent at one response format might start applying that format where it does not belong.

### Keep comparison conditions fair

Use matching prompts, retrieval results, output limits, and decoding settings where possible. Random generation can vary between attempts, so evaluate enough examples and repeat measurements when randomness could affect the conclusion.

If 2 extra answers are correct in a test of 20, that is weak evidence of a dependable improvement. Use a larger independent evaluation or report the uncertainty instead of claiming a decisive win.

### Human and model judges

Human reviewers need a clear rubric. Model judges need calibration against human review and checks for position, verbosity, and style biases.

Where possible, hide model identity and randomize response order. A fluent answer should not win merely because it sounds authoritative.

Choose checkpoints with validation data. Once you repeatedly use test failures to make training decisions, that test set has effectively become development data; a new independent check is needed.

## 17. Common failures and how to investigate them

| Symptom | Possible explanation | Useful next check |
|---|---|---|
| Training improves, validation gets worse | Overfitting or a distribution mismatch. | Inspect duplicates, data coverage, and earlier checkpoints. |
| Neither improves | Weak signal, bad labels, unsuitable model, or training configuration error. | Inspect actual tokenized examples, masks, and trainable parameters. |
| Fluent but wrong answers | Training rewards plausible wording more than evidence. | Check factual labels, context use, and task-level evaluation. |
| Repeats one response | Low diversity, duplicated targets, or excessive adaptation. | Inspect output distribution and dataset balance. |
| Ignores the user | Template, loss-mask, or data construction problem. | Decode several exact training records. |
| Never stops generating | Incorrect stop-token/template handling. | Compare training and serving token conventions. |
| Outputs invalid JSON sometimes | Learned formatting is imperfect. | Validate outputs; use constrained decoding where supported. |
| Original capabilities decline | Catastrophic forgetting or adapter interference. | Run regression tasks and reconsider data mixture or update strength. |
| GPU runs out of memory | Excessive batch/sequence size or training state. | Profile the actual configuration and reduce its memory demands. |
| Reward rises, real usefulness falls | The reward can be exploited. | Inspect high-reward failures and improve independent checks. |

**Overfitting** means learning details of the training set that fail to generalize. **Underfitting** means the model has not learned the task sufficiently. **Catastrophic forgetting** means adaptation damages previously useful abilities.

Freezing the base in LoRA preserves its stored weights, but the combined model can still behave worse on old tasks. Removing the adapter restores the unchanged base only if you actually kept it unchanged and available.

Treat the table as diagnostic possibilities, not automatic explanations. Several different bugs can produce the same symptom.

## 18. A worked example from start to finish

Suppose you want an assistant to route support messages into `billing`, `account_access`, `bug`, or `needs_review`.

### Step 1: Define success

Write category rules and examples of overlaps. Decide how to handle a message that mentions both a failed login and an unexpected charge. Without a policy, labelers will teach contradictory behavior.

Success might mean correct routing, valid JSON, and fewer missed account-access cases. Choose the targets before examining final test results.

### Step 2: Establish a baseline

Try the original model with clear instructions and a few examples. Record its errors on a development set. If the baseline already meets requirements, fine-tuning may not be necessary.

### Step 3: Build a reviewed dataset

For illustration, collect 1,000 representative labeled messages. Include terse messages, misspellings, multiple issues, and ambiguous cases. Keep related customer conversations together when splitting the dataset.

This number is an example budget, not a claim that 1,000 is enough.

### Step 4: Select a simple experiment

A reasonable first candidate is SFT on a capable instruct model using LoRA. If memory is the limiting factor and the stack supports it, consider QLoRA.

Document why you chose the model: baseline quality, language coverage, license, context needs, and serving constraints.

### Step 5: Inspect the training pipeline

Look at a handful of tokenized records. Confirm that inputs include all required information, targets are correct, assistant tokens are scored as intended, and truncation does not remove labels.

Run a short smoke test to confirm the pipeline operates. A tiny training run proves the machinery works, not that the final system is good.

### Step 6: Train and select a checkpoint

Track both loss and task metrics on validation data. Save enough information to reproduce the run. Select the checkpoint that best meets your evaluation criteria, which may not be the final checkpoint.

### Step 7: Evaluate independently

Compare the chosen model with the baseline on the reserved test data. Review disagreements, minority categories, and ambiguous requests. Check whether the gain is large and consistent enough to matter.

### Step 8: Deploy with application checks

Validate the output schema and allowed category values. Route invalid outputs to a defined fallback. Monitor real mistakes and keep a rollback option.

The trained model provides a prediction; the surrounding application is responsible for handling invalid outputs and executing any downstream workflow.

### What the conceptual code does

```python
# Pseudocode only: illustrates responsibilities, not a runnable library API.
model = load_starting_model()
training, validation, test = load_separate_datasets()
baseline = evaluate(model, test, fixed_task_rubric)

model = attach_trainable_adapters(model)
for batch in training_batches(training):
    predictions = model(batch.inputs, batch.correct_previous_tokens)
    loss = score_target_tokens(predictions, batch.targets, batch.loss_mask)
    gradients = calculate_gradients(loss)
    update_trainable_parameters(gradients)
    periodically_evaluate_and_save(model, validation)

chosen_model = choose_checkpoint_using_validation()
result = evaluate(chosen_model, test, fixed_task_rubric)
compare(baseline, result)
```

Keep the test comparison sealed from iterative training decisions. Use the development set to investigate and fix problems before the final comparison.

## 19. What I would do with 1,000 PDFs

If the goal is an assistant that answers questions about 1,000 PDFs, I would build a **RAG system first**, then fine-tune only the component whose failures justify it.

The core design would look like this:

```text
1,000 PDFs
    |
    v
Extract text, tables, page numbers, headings, and metadata
    |
    v
Clean, deduplicate, and split into meaningful passages
    |
    +---------------------> Search index
    |                           |
    |                           v
    |                    Retrieve evidence
    |                           |
    v                           v
Reviewed training examples --> Tuned assistant --> Answer with citations
```

The PDFs play two different roles:

- The **search index** preserves detailed, updateable knowledge.
- The **training examples** teach the model how to answer from retrieved evidence.

I would not simply concatenate all the PDF text and expect fine-tuning to turn the model into a dependable library. It may absorb terminology and recurring patterns, but it will not reliably retrieve every figure, exception, or page-level citation from its weights.

### Step 1: Decide what the system must do

“Know these PDFs” is too vague. I would write concrete tasks such as:

- Answer a question using only the supplied document evidence.
- Quote or cite the supporting document and page.
- Compare rules from two documents.
- Extract named fields from reports.
- Summarize one document or a retrieved set of passages.
- Say that the answer is unavailable when the evidence is insufficient.

The task determines the extraction method, training records, and evaluation. A system for contract clauses needs different handling from one for scanned engineering diagrams.

### Step 2: Inventory the collection

Before training anything, I would measure what is actually in the 1,000 PDFs:

- Born-digital text versus scanned pages needing OCR.
- Languages, page counts, file sizes, and publication dates.
- Tables, forms, diagrams, footnotes, headers, and multi-column layouts.
- Duplicate files and revised editions.
- Access restrictions, confidential content, and documents that should be excluded.

“1,000 PDFs” could mean 5,000 simple pages or 500,000 difficult pages. Page structure and extraction quality often matter more than the file count.

### Step 3: Extract with traceability

For each piece of content, I would preserve at least:

```json
{
  "document_id": "policy-2026-04",
  "title": "Refund Policy",
  "version": "2026-04",
  "page": 17,
  "section": "Exceptions",
  "text": "...extracted passage..."
}
```

Digital PDFs can often yield text directly. Scans need optical character recognition, or OCR. Complex tables and diagrams may need a layout-aware or multimodal extraction path. I would manually inspect a sample from every major document type because plausible-looking extraction can silently scramble columns, symbols, or table rows.

Headers, footers, page numbers, broken hyphenation, repeated navigation text, and OCR artifacts should be cleaned carefully. Cleanup must retain the page and section mapping needed for citations.

### Step 4: Split by document before creating examples

I would assign complete documents or document families to training, validation, and test groups before generating questions.

This prevents a common form of leakage. If one chapter appears in training and a paraphrased question about the same chapter appears in the test set, the result measures familiarity with that document, not generalization to unseen documents.

I would usually maintain two useful evaluations:

- **Seen-document evaluation:** new questions about documents available during development.
- **Unseen-document evaluation:** questions about entirely held-out documents.

The first tests practical coverage of the collection. The second tests whether the method transfers to future documents.

### Step 5: Create retrieval passages

I would split extracted content along headings, paragraphs, list boundaries, and table boundaries where possible. Every passage would keep its document, page, section, date, and permissions metadata.

Chunks should contain enough context to stand on their own without becoming so broad that search loses precision. There is no universally correct chunk size. I would test several strategies against real questions.

Then I would create an embedding index or another searchable index. At query time, the system retrieves candidate passages and may use a reranker to put the most relevant ones first.

### Step 6: Build a small, high-quality evaluation set first

Before generating a huge training set, I would have people write and review representative questions with:

- The correct answer.
- Exact supporting passages and pages.
- Acceptable answer variations.
- Cases where the PDFs do not contain the answer.
- Difficult cases involving tables, multiple documents, conflicting versions, or similar terminology.

This evaluation set tells us whether retrieval, generation, or extraction is failing. Without it, training can improve a loss number while the real product remains unreliable.

### Step 7: Establish the untuned RAG baseline

I would run a capable instruction-following model with retrieved passages and a clear prompt. I would measure:

- Did retrieval include the needed evidence?
- Did the answer agree with that evidence?
- Were citations attached to claims they actually support?
- Did the model abstain when evidence was missing?
- How long, slow, and expensive was each request?

This baseline frequently reveals that the main problem is extraction or retrieval. Fine-tuning the generator cannot recover a passage the system never retrieved.

### Step 8: Generate candidate training records

For generator fine-tuning, each record would resemble the real inference request:

```json
{
  "question": "Which refund requests require manager approval?",
  "evidence": [
    {
      "document": "Refund Policy",
      "page": 17,
      "text": "Refunds above $500 require manager approval..."
    }
  ],
  "answer": "Refunds above $500 require manager approval. [Refund Policy, p. 17]"
}
```

I would include several important variations:

- Answerable questions with one decisive passage.
- Questions requiring evidence from multiple passages.
- Distractor passages that sound relevant but do not support the answer.
- Unanswerable questions whose correct response is to say the evidence is insufficient.
- Conflicting versions where the assistant must use the current or specified edition.
- Requests for structured extraction, if that is part of the product.

A strong model can propose questions and answers, but I would validate them against the source. Synthetic generation can create large volumes of fluent, unsupported labels. Automated checks can catch citation mismatches and malformed records, while humans review samples and high-risk cases.

### Step 9: Fine-tune only after identifying a behavioral gap

If the baseline finds the evidence but answers inconsistently, I would try supervised fine-tuning, usually with LoRA or QLoRA for an initial experiment. The model would learn the desired answer format, citation behavior, use of context, and abstention policy.

If retrieval misses the evidence, I would improve chunking and search first. If needed, I would fine-tune the embedding model or reranker using query, relevant-passage, and carefully checked negative-passage examples.

If the domain uses unusual language that the base model truly handles poorly even when given the right passage, I might test continued pretraining on cleaned domain text. That is a larger experiment and still does not replace retrieval.

### Step 10: Test the complete system

I would evaluate at three levels:

| Level | Main question |
|---|---|
| Extraction | Did the pipeline preserve the correct text, tables, and page references? |
| Retrieval | Did the right evidence appear near the top? |
| Answering | Given the evidence, was the answer correct, supported, and appropriately cautious? |

I would also test the deployed combination: final search index, exact prompt/template, adapter or merged model, quantization, and generation settings. A good model tested with hand-selected evidence can still fail in the complete pipeline.

### Step 11: Plan for updates

When a PDF is added or replaced, updating the search index should usually be enough to make its information available. Retraining should be reserved for new behaviors, recurring failure patterns, or meaningful domain shifts.

I would track document versions so an answer can identify its source and so obsolete editions can be removed from retrieval. This is a major operational advantage of keeping document knowledge in an index instead of relying on memorization in model weights.

### What I would choose in practice

For a first production-minded version, my default would be:

1. Reliable PDF extraction with document/page metadata.
2. Hybrid retrieval, usually combining semantic and keyword search.
3. A reranker if initial retrieval is noisy.
4. A capable instruction model prompted to answer only from evidence and cite it.
5. A human-reviewed test set that includes missing-answer and conflicting-version cases.
6. LoRA/QLoRA SFT only if the baseline shows repeatable behavioral failures that prompting does not solve well.

That design gives you traceable answers, easy document updates, and a clear reason for every training step.

## 20. Fine-tuning a RAG system

A document-answering system often has several components:

```text
Question
   |
   v
Retriever finds candidate passages
   |
   v
Optional reranker reorders candidates
   |
   v
Generator writes an answer using the selected evidence
```

### Tune the component responsible for the failure

| Failure | Component to investigate |
|---|---|
| Relevant passage never appears | Retrieval, indexing, chunking, or query handling. |
| Relevant passage appears but is buried | Ranking/reranking. |
| Correct evidence is supplied but misread | Generator behavior. |
| Answer is right but citations are unsupported | Generator training and citation validation. |
| Source documents lack the answer | Data coverage and appropriate abstention. |

### Embedding fine-tuning

A retrieval embedding model can learn from query/relevant-passage pairs, often with irrelevant passages as contrasts. The objective encourages useful query-document relationships in its numerical representations.

**Hard negatives** are plausible-looking but irrelevant passages. They can teach fine distinctions, provided they are truly irrelevant; mislabeled positives create bad lessons. [Sentence Transformers training overview](https://sbert.net/docs/sentence_transformer/training_overview.html)

If document embeddings change, rebuild the stored index using the compatible embedding model. Mixing old document vectors with a new, incompatible query representation can damage search.

### Generator fine-tuning

Train on examples containing the question, evidence, and a grounded answer. Include irrelevant passages, missing answers, and conflicting material when those occur in practice.

To check whether the generator uses context rather than memorized answers, try paired evaluation cases where the evidence changes and the correct answer must change with it. Also test what happens when the decisive passage is removed.

For a project comparing RAG and fine-tuning, document-level separation matters particularly: distinguish answering about documents seen during training from generalizing to genuinely unseen documents. Both can be useful experiments, but they answer different questions.

## 21. Fine-tuning beyond language models

The same broad idea—adapt a pretrained model—applies elsewhere, but data and objectives differ.

| Model/application | Training example | What adaptation can target |
|---|---|---|
| Image classifier | Image plus category. | Recognizing your product categories or defect types. |
| Object detector | Image plus labeled boxes. | Finding task-specific objects. |
| Segmentation model | Image plus pixel labels. | Identifying precise regions. |
| Speech recognition | Audio plus transcript. | Accent, vocabulary, or acoustic conditions. |
| Embedding model | Related/unrelated items. | Better task-specific search or similarity. |
| Multimodal assistant | Image or audio plus instruction and answer. | Tasks requiring several input types. |

Image-generation adaptation can teach a subject or style using suitable image-caption training. Diffusion models use different training objectives from next-token language modeling; “show input and score prediction” remains a broad analogy, not one identical algorithm.

Some multimodal projects train a connector between components while freezing the rest. Others update more of the system. This is another example of separating the learning objective from the parameters you choose to train.

## 22. Saving, deploying, and maintaining the result

### What should be saved?

- Model weights or adapter weights.
- Exact base model identity and revision.
- Tokenizer, chat template, and special-token settings.
- Training configuration and software versions.
- Dataset version and split definitions.
- Evaluation results and intended-use notes.

For reliable **resumption of training**, you may also need optimizer state, scheduler state, random-number state, and training progress. A lightweight inference adapter alone is not always a complete resumable checkpoint.

### Serving changes can change quality

Quantizing, merging an adapter, changing the template, or switching inference engines can affect behavior. Evaluate the actual deployed configuration, not just the training-time model.

**Temperature** controls randomness during generation. It does not train the model. Lower temperature can make outputs more repeatable but does not guarantee correctness.

Fine-tuning does not inherently make a model smaller or faster. Efficiency gains can come from shorter prompts or replacing a larger model with a smaller tuned model, but measure them in the complete system.

### Maintenance

Monitor changes in incoming requests, category balance, document versions, and failure rates. These changes are forms of **distribution shift** or **data drift**.

Collect reviewed corrections and evaluate whether another training run is justified. Ordinary inference does not automatically incorporate those corrections into weights.

Removing a training record later does not by itself remove its influence from an existing checkpoint. Keep sensitive information out of the training process where possible, and retain traceability of which data produced which models.

## 23. Frequently asked questions

### Do I need to be good at math?

You can understand, run, and evaluate many fine-tuning workflows without deriving equations. Data quality, careful experiments, and debugging are initially more valuable than memorizing formulas. Math becomes more useful when you want to invent methods or explain optimization behavior precisely.

### Does fine-tuning make a model smarter?

It can make it substantially better at selected tasks. It can also improve some capabilities while damaging others. “Smarter” is too broad to evaluate; name the task and test it.

### Can it learn new facts?

Yes, but recall can be incomplete or unreliable. Fine-tuning is not a guarantee of exact knowledge storage or selective updates. Use retrieval or tools when freshness and traceable evidence are central.

### Can I just train on all my PDFs?

You first need to decide what behavior you want. Extracted document text supports one kind of objective; question-context-answer demonstrations support another. PDF extraction errors, tables, duplicate passages, and missing context can all corrupt the curriculum.

### Does it eliminate hallucinations?

No. It can improve grounding and abstention with suitable data and evaluation, but unsupported claims can remain. A dataset full of confident answers can even make the problem worse.

### Is LoRA only for small or simple tasks?

No. It is a method for constraining and storing updates. Whether it is sufficient depends on the model, data, objective, target modules, and evaluation. Benchmark your task rather than assuming either LoRA or full tuning always wins.

### Can I use an adapter with any similar model?

Do not assume so. Architecture compatibility is not enough to guarantee that learned updates work with different base weights. Track the exact base revision and test any transfer explicitly.

### Does a model learn automatically from my chats?

An ordinary model inference call does not update weights. An application may separately store conversation history or run a later training pipeline. Those are different mechanisms and depend on the application.

### Can I train tool use or reasoning?

You can train task-solving behavior, tool-call formats, and responses to tool results. You still need an application that executes tools correctly and enforces its permissions. Longer explanations alone are not evidence of more reliable reasoning; evaluate outcomes and transfer to new cases.

### Is fine-tuning the same as hyperparameter tuning?

No. Fine-tuning updates learned model parameters. Hyperparameter tuning searches for useful training settings, such as learning rate and batch size. You can tune hyperparameters while running fine-tuning experiments.

### Why did copying an online recipe fail?

Recipes depend on the exact model, tokenizer, dataset format, sequence length, hardware, and library version. A working configuration is a starting point, not a portable guarantee.

## 24. A practical learning path

1. Understand tokens, parameters, context, loss, and generalization.
2. Pick one narrow, measurable task.
3. Try clear prompting and a few examples.
4. Build reviewed data and independent evaluation splits.
5. Run one small SFT experiment, with LoRA if appropriate.
6. Inspect failures and compare with the baseline.
7. Learn memory-saving methods when resources require them.
8. Explore preference methods when demonstrations no longer capture the distinction you care about.
9. Explore reward-based training when you have a trustworthy scorer and a reason to accept the added complexity.

Before any run, be able to answer: **What should improve, what evidence teaches it, and how will I know it improved on new cases?**

## 25. Glossary

| Term | Meaning |
|---|---|
| Adapter | Additional learned parameters used with a base model. |
| Alignment | Shaping model behavior toward chosen goals or preferences; the term is broader than any single method. |
| Autoregressive | Generates each next token conditioned on earlier tokens. |
| Base model | Usually a pretrained model before instruction-oriented post-training. |
| Batch | A group of examples or tokens processed together. |
| Checkpoint | Saved model and optionally training state. |
| Contrastive learning | Learn useful representations by contrasting related and unrelated examples. |
| Corpus | A collection of training or evaluation material. |
| Cross-entropy | A common loss that penalizes assigning too little probability to correct targets. |
| Data leakage | Evaluation information accidentally influences training or selection. |
| Decoding | The procedure for choosing output tokens from model predictions. |
| Distillation | Training a student using signals from a teacher. |
| Distribution shift | Real inputs differ from the data previously used. |
| Epoch | One pass through a dataset. |
| Fine-tuning | Additional training of an existing model for adaptation. |
| Frozen | Not being changed by optimizer updates. |
| Generalization | Performance on examples beyond the training set. |
| Grounding | Basing an answer on supplied or retrieved evidence. |
| Inference | Running the model to produce output. |
| Instruction tuning | Training on instructions and desired responses. |
| Logits | Raw model scores before conversion to probabilities. |
| Loss mask | Specifies which prediction positions contribute to loss. |
| Optimizer state | Extra stored information an optimizer uses for updates. |
| Overfitting | Learning training-specific details that do not transfer well. |
| PEFT | Parameter-efficient fine-tuning. |
| Policy | In this context, the model's distribution over possible outputs. |
| Quantization | Representing values using fewer bits. |
| Reference model | A comparison/anchor model used in some training objectives. |
| Regularization | Methods intended to control fitting and improve generalization. |
| Reward model | A learned system that scores outputs. |
| Seed | Initialization for pseudorandom choices; useful for reproducibility. |
| SFT | Supervised fine-tuning. |
| Synthetic data | Training material generated by a program or model. |
| Trainable parameter | A learned number the optimizer is allowed to change. |
| Validation | Evaluation used to guide model and setting selection. |

## 26. Sources and further reading

The explanations and hypothetical examples above are written for this guide. These primary sources and official documentation provide technical detail for the referenced methods. You can use the guide without reading the equations in the papers.

- [Hugging Face fine-tuning overview](https://huggingface.co/docs/transformers/en/training): the basic adaptation workflow.
- [TRL SFT trainer](https://huggingface.co/docs/trl/en/sft_trainer): supervised training formats and controls.
- [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964): domain and task adaptation through additional pretraining.
- [LoRA paper](https://arxiv.org/html/2106.09685v2) and [PEFT LoRA guide](https://huggingface.co/docs/peft/main/conceptual_guides/lora): compact trainable updates.
- [QLoRA paper](https://arxiv.org/abs/2305.14314): adapting a frozen quantized base using LoRA.
- [PEFT documentation](https://huggingface.co/docs/peft/en/index): the wider family of efficient adaptation methods.
- [DPO paper](https://arxiv.org/abs/2305.18290) and [TRL DPO trainer](https://huggingface.co/docs/trl/en/dpo_trainer): learning from preference pairs.
- [InstructGPT research](https://arxiv.org/abs/2203.02155): a classic demonstration, reward-model, and RL pipeline.
- [TRL GRPO trainer](https://huggingface.co/docs/trl/en/grpo_trainer): generation and relative reward optimization.
- [RAG paper](https://arxiv.org/abs/2005.11401): combining retrieval and generation.
- [Sentence Transformers training overview](https://sbert.net/docs/sentence_transformer/training_overview.html): adapting embedding models.
- [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html): full adaptation and frozen feature extraction in vision.
- [Transformers chat templates](https://huggingface.co/docs/transformers/en/chat_templating): consistent message formatting.
- [Transformers memory and speed guide](https://huggingface.co/docs/transformers/en/llm_tutorial_optimization): the resources needed to run language models.
