# Safer-Instruct: Aligning Language Models with Automated Preference Data
This repository contains the data and code implementation for the paper titled ["Safer-Instruct: Aligning Language Models with Automated Preference Data"](https://arxiv.org/abs/2311.08685).
<img src="img/pipeline.png">

## Abstract
Reinforcement learning from human feedback (RLHF) is a vital strategy for enhancing model capability in language models. However, annotating preference data for RLHF is a resource-intensive and creativity-demanding process, while existing automatic generation methods face limitations in data diversity and quality. In response, we present Safer-Instruct, a novel pipeline for automatically constructing large-scale preference data. Our approach leverages reversed instruction tuning, instruction induction, and expert model evaluation to efficiently generate high-quality preference data without human annotators. To verify the effectiveness of Safer-Instruct, we apply the pipeline to construct a safety preference dataset as a case study. Finetuning an Alpaca model on this synthetic dataset not only demonstrates improved harmlessness but also outperforms models fine-tuned on human-annotated safety preference data, all the while maintaining a competitive edge in downstream tasks. Importantly, our Safer-Instruct framework is versatile and can be applied to generate preference data across various domains, extending its utility beyond safety preferences. It addresses the challenges in preference data acquisition and advances the development of more capable and responsible AI systems.

## Dataset Release
You can download our dataset [here](https://github.com/uscnlp-lime/safer-instruct/tree/main/sample_data). The dataset contains content that can be offensive or upsetting. It is also important to note that the Safer-Instruct process can be easily reversed to train a harmful LLM, and thus the dataset should be used for academic purposes only. In addition, part of our dataset is collected from X and Reddit. Releasing the data might violate their content distribution policy. Hence, entries that contain the data collected from X and Reddit have been removed. To request the full dataset, please contact one of the authors.

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
If you find this repository helpful, please cite our paper.

```
@misc{shi2023saferinstruct,
      title={Safer-Instruct: Aligning Language Models with Automated Preference Data}, 
      author={Taiwei Shi and Kai Chen and Jieyu Zhao},
      year={2023},
      eprint={2311.08685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

Feel free to contact Taiwei Shi at taiweish@usc.edu, if you have any questions about the paper.
