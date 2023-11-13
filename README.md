# Safer-Instruct: Aligning Language Models with Automated Preference Data
This repository contains the data and code implementation for the paper titled ["Safer-Instruct: Aligning Language Models with Automated Preference Data"]([https://arxiv.org/abs/2310.15638v1](https://github.com/uscnlp-lime/safer-instruct/edit/main/README.md)).
<img src="img/pipeline.png">

## Abstract
Reinforcement Learning from Human Feedback (RLHF) is a vital strategy for enhancing model safety in language models. However, annotating preference data for RLHF is a resource-intensive and creativity-demanding process, while automatic generation methods face limitations in data diversity and quality. In response, we present Safer-Instruct, a novel pipeline for semi-automatically constructing large-scale preference datasets. Our approach leverages reversed instruction tuning, instruction induction, and expert model evaluation to efficiently generate high-quality preference data without human annotators. We evaluate Safer-Instruct using LLaMA for instruction induction and GPT-4 as an expert model, generating approximately 10K preference samples. Finetuning an Alpaca model on this dataset demonstrates improved harmlessness while maintaining competitive performance on conversation and downstream tasks. Safer-Instruct addresses the challenges in preference data acquisition, advancing the development of safer and more responsible AI systems.

## Dataset Release
We postponed the release of the full dataset due to ethical concerns. The Safer-Instruct process can be easily reversed to train a "bad" LLM. In addition, part of our dataset is collected from X and Reddit. Releasing the data might violate their content distribution policy. However, you can download the sample data [here](https://github.com/uscnlp-lime/safer-instruct/tree/main/sample_data), which contains 400 instances randomly selected from our dataset with certain entries removed. The dataset should be used for academic purposes only. To request the full dataset, please contact one of the authors.

## Usage
### Reversed Instruction Tuning
We use the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) repo to do reversed instruction tuning. The only modification is the prompt template, which is shown below. We use the same prompt for inference. 
```
Below is a response to a certain instruction. Write the instruction that the response is trying to complete.

### response:
{output}

### Instruction:
```

### Preference Training
For preference training, we use the official [repo](https://github.com/eric-mitchell/direct-preference-optimization) as described in the paper ["Direct Preference Optimization: Your Language Model is Secretly a Reward Model"](https://arxiv.org/abs/2305.18290). We first finetuned the model on our data using SFT. We then train the SFT model using DPO.

For SFT training, we use the following command.
```
python -u train.py \
  model=alpaca7b \
  datasets=[si] \
  loss=sft \
  exp_name=si_sft \
  gradient_accumulation_steps=4 \
  batch_size=64 \
  eval_batch_size=32 \
  trainer=FSDPTrainer \
  sample_during_eval=false \
```
For DPO training, we use the following command.
```
python -u train.py \
  model=alpaca7b \
  model.archive=policy.pt \
  datasets=[si] \
  loss=dpo \
  loss.beta=0.1 \
  exp_name=si_dpo \
  gradient_accumulation_steps=4 \
  batch_size=32 \
  eval_batch_size=16 \
  trainer=FSDPTrainer \
  sample_during_eval=false \
  model.fsdp_policy_mp=bfloat16 \
```

## Citation and Contact
Feel free to contact Taiwei Shi at taiweish@usc.edu, if you have any questions about the paper.
