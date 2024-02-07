# DreamGen fine-tuning scripts

Super simple scripts for supervised fine-tuning and preference-tuning, mostly based on [Unsloth](https://github.com/unslothai/unsloth/tree/main#installation-instructions).
Fork it, and do whatever you want with it.

## Installation

- Install [Unsloth](https://github.com/unslothai/unsloth/tree/main#installation-instructions).
- Install other dependencies `pip install fire`.

## dpo.py

### Tips 

Experiment with different learning rate, beta and DPO / IPO. The actual best value will be highly data dependent.
Start with learning rate that's approximately 10-times smaller than the learning rate you normally use for QLoRA SFT.

Monitor `(train|eval)/rewards/accuracies`, `(train|eval)/rewards/margins` and `train/loss`. They should not improve too fast, if they do, lower learning rate.

Never trust just your `(train|eval)/rewards` metrics alone, perform end-to-end testing on a dataset that does not overlap with your SFT and DPO data.

Here are some example DPO DreamGen V1 7B runs, using different learning rates:

[!rewards](images/dpo-example1-rewards.webp)
[!train-loss](images/dpo-example1-train-loss.webp)
[!learning-rate](images/dpo-example1-learning-rate.webp)

Here are some example DPO Bagel runs:

- [bagel-dpo-34b-v0.2](https://wandb.ai/jondurbin/bagel-dpo-34b-v0.2)
- [bagel-dpo-8x7b-v0.2](https://wandb.ai/jondurbin/bagel-dpo-8x7b-v0.2)
- [bagel-dpo-7b-v0.1](https://wandb.ai/jondurbin/bagel-dpo-7b-v0.1)

### Useful DPO reading material

- [Comparison of DPO, IPO, KTO and various hyper-params](https://huggingface.co/blog/pref-tuning)
- [DPO tutorial from HuggingFace](https://huggingface.co/blog/dpo-trl)
- [DPO tutorial from Maxime Labonne](https://towardsdatascience.com/fine-tune-a-mistral-7b-model-with-direct-preference-optimization-708042745aac)
- [DPO paper](https://arxiv.org/abs/2305.18290)

## End-to-end evals

End-to-end evals on data that's disjoint from your SFT and DPO training data are crucial to assess real improvements.
Ideally, your evals should be as close to what you intend to use the final model for. If you don't have that, you can use one of the existing broad-spectrum auto-evals:

- https://github.com/EleutherAI/lm-evaluation-harness
- https://github.com/mlabonne/llm-autoeval
