<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

Our code is implemented in [PyTorch](https://pytorch.org/), using the [Transformers 🤗](https://github.com/huggingface/transformers) libraries. 

</div>

______________________________________________________________________

# KD4Gen

### Official code repository for the ACL'2023 paper: <br> ["A Systematic Study of Knowledge Distillation for Natural Language Generation with Pseudo-Target Training"](https://arxiv.org/abs/2305.02031)


![Study](figures/study_diagram.png)

If you use this code please cite our paper (see [below](#citation)).

______________________________________________________________________

### Install and activate the environment

```bash
conda env create -f environment.yml
conda activate kd
```

______________________________________________________________________

### Notice
In this README we provide examples for running the Shakespeare dataset and T5 model, however, more commands for running other are provided in ...
______________________________________________________________________

### Download the Data

**Option 1:** <br>
Download the data from [here](https://drive.google.com/file/d/1Z2KcNgE36mgNEyTwfgQxAGfOqbl54L_v/view?usp=sharing). Then, unzip it:
```bash
unzip file.zip
```

**Option 2:** <br>
Run the following script:
```bash
python prepare_datasets_main.py --dataset_name shakespeare --train_size 7000 --val_size 750 --val_ppl_size 750 --unlabeled_size 28000 --unlabeled_split True --filter_by_length False --max_input_length 48 --max_labels_length 48 --add_ft_split full --file_name shakespeare_7k.json --output_dir ./datasets/original
```

Note: Make sure the data is in the `datasets/original` folder. <br>
A data file should be a JSON file, each key is a split: `'train', 'dev', 'test', 'unlabeled'.`<br>
Each split is also a dict: `split: {'text': [list of texts], 'target': [list of targets]}`. 

______________________________________________________________________

### Fine-tuning the teacher
Run the following script:

```bash
python ft_main.py --max_input_length 48 --max_labels_length 48 --task_prompt modern: --mixed_precision no --metrics all --num_beams 4 --seed 42 --debug False --do_train True --save_checkpoint_every_n_steps 0.55 --test_every_n_epochs 5 --keep_checkpoint_after_test False --n_patience_epochs 10 --metric_for_best_model sacrebleu --greater_is_better True --model_name t5-large --batch_size 96 --max_gpu_batch_size 96 --learning_rate 5e-05 --epochs 15 --output_dir ./outputs/shakespeare_7k/t5/ft_teacher/0_000050/none/none --dataset_path ./datasets/original/shakespeare_7k.json
```

Note: The `ft_main.py` script is mainly based on the function `src.train_models_accelerator.train_model`. <br>
If you encounter an error while trying to fine-tune a model that is not supported by the code, you should start by modifying this function.

______________________________________________________________________

### Generate pseudo-targets with the teacher
Run the following script:
```bash
python gen_main.py --max_input_length 48 --max_labels_length 48 --task_prompt modern: --mixed_precision no --split_for_generation unlabeled train --num_beams 48 --num_return_sequences 48 --generation_type sampling --add_original_data True --seed 42 --debug False --model_name t5-large --batch_size 96 --output_dir ./datasets/generations/shakespeare_7k/t5/sampling --dataset_path ./datasets/original/shakespeare_7k.json --checkpoint_path ./outputs/shakespeare_7k/t5/ft_teacher/0_000050/none/none/********.pt
```

Note: Replace the `********.pt` with the path to the teacher's checkpoint. <br>
You can also use beam search (`--generation_type beam_search`) and control the number of PTs (`--num_beams X --num_return_sequences Y`) or other arguments, see: `gen_main.py`. <br>
______________________________________________________________________

### Prepare the data for the student (pseudo-targets + original data)
Run the following script:
```bash
python prepare_datasets_main.py --original_dataset_path ./datasets/original/shakespeare_7k.json --generation_dataset_path ./datasets/generations/shakespeare_7k/t5/sampling/generated_datasets.json --n_augmentations_per_example 48 --file_name sampling_48.json --use_unlabeled True --use_labeled True --use_original True --output_dir ./datasets/with_augmentations/shakespeare_7k/t5
```

______________________________________________________________________

### Distill the teacher (student training)
Run the following script:
```bash
python kd_main.py --max_input_length 48 --max_labels_length 48 --task_prompt modern: --mixed_precision no --metrics all --num_beams 4 --seed 42 --debug False --do_train True --save_checkpoint_every_n_steps 0.55 --test_every_n_epochs 8 --keep_checkpoint_after_test True --n_patience_epochs 16 --ft_steps_at_end 10 --epochs 192 --stop_training_after_n_epochs 96 --metric_for_best_model sacrebleu --greater_is_better True --experiment_config_key logits_kd --student_name t5-small --teacher_name t5-large --batch_size 96 --max_gpu_batch_size 96 --learning_rate 0.0005 --output_dir ./outputs/shakespeare_7k/t5/kd/0_000500/logits_kd/sampling_48 --teacher_state_dict_path ./outputs/shakespeare_7k/t5/ft_teacher/0_000050/none/none/********.pt --dataset_path ./datasets/with_augmentations/shakespeare_7k/t5/sampling_48.json
```

Note: The `kd_main.py` script is mainly based on the function `src.kd_models_accelerator_multiple_loaders.train_kd`. <br>
Please see the arguments of `kd_main.py`. <br>
Replace the `********.pt` with the path to the teacher's checkpoint. <br><br>
Important: You can apply different KD techniques by specifying the `--experiment_config_key` argument. For example: `finetuning, logits_kd, noisy_kd, attention_kd, logits_kd_student_gens_co_teaching, logits_kd_student_gens`<br> 
You can also implement your own custom KD technique by modifying the `arg_configs.kd_experiment_configs.yaml` YAML file, see the `logits_kd` key for example. <br> 
You can also distill the teacher without PTs (only the original data) by using the `--dataset_path ./datasets/original/shakespeare_7k.json` argument. <br>
The code and the arguments are highly flexible, and we don't cover the whole experimental setup in this README. <br>
______________________________________________________________________

### Distill GPT-4 (dataset with logits)

First create the dataset with logits:
```bash
python prepare_datasets_main.py --logits_dataset_path ./datasets/for_gpt4/logits/shakespeare_7k.json --file_name 1.json --student_name t5-small --seed 42 --n_augmentations_per_example 1 --output_dir ./datasets/with_logits/shakespeare_7k
```

Then, run the following script:
```bash
python kd_dataset_with_logits_main.py --max_input_length 48 --max_labels_length 48 --task_prompt modern: --mixed_precision no --metrics all --num_beams 4 --seed 42 --debug False --do_train True --save_checkpoint_every_n_steps 0.55 --test_every_n_epochs 8 --keep_checkpoint_after_test True --n_patience_epochs 16 --ft_steps_at_end 10 --epochs 192 --stop_training_after_n_epochs 96 --metric_for_best_model sacrebleu --greater_is_better True --experiment_config_key logits_kd --student_name t5-small --batch_size 96 --max_gpu_batch_size 48 --learning_rate 0.003 --output_dir ./outputs/shakespeare_7k/t5/logits/0_003000/logits_kd/1 --dataset_path ./datasets/with_logits/shakespeare_7k/1.json
```

Note: Please see the arguments and their description in `kd_dataset_with_logits_main.py` file. <br>
______________________________________________________________________


## How to Cite
<a name="citation"/>

```
@article{DBLP:journals/corr/abs-2305-02031,
  author       = {Nitay Calderon and
                  Subhabrata Mukherjee and
                  Roi Reichart and
                  Amir Kantor},
  title        = {A Systematic Study of Knowledge Distillation for Natural Language
                  Generation with Pseudo-Target Training},
  journal      = {CoRR},
  volume       = {abs/2305.02031},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2305.02031},
  doi          = {10.48550/arXiv.2305.02031},
  eprinttype    = {arXiv},
  eprint       = {2305.02031},
  timestamp    = {Fri, 05 May 2023 14:35:02 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2305-02031.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```