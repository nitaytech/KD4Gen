import os

import numpy as np
import torch
import json
from collections import defaultdict
from transformers import PreTrainedTokenizerBase
from torch.utils.data import DataLoader
from datasets import Dataset
from src.constants import *


def sample_dataset(dataset: Dataset, size: int,
                   seed: int = None, return_unsampled: bool = False) -> Union[Tuple[Dataset, Dataset], Dataset]:
    if seed is not None:
        rng = np.random.RandomState(seed)
        indices = rng.choice(list(range(len(dataset))), size=size, replace=False)
    else:
        indices = np.random.choice(list(range(len(dataset))), size=size, replace=False)
    sampled = dataset.select(indices)
    if return_unsampled:
        unsampled = dataset.select(list(set(range(len(dataset))) - set(indices)))
        return sampled, unsampled
    return sampled


def add_id_col(dataset_dict: Dict[str, Dataset]) -> Dict[str, Dataset]:
    for split, dataset in dataset_dict.items():
        dataset = dataset.to_dict()
        dataset[ID_COL] = list(range(len(dataset[TEXT_COL])))
        dataset_dict[split] = Dataset.from_dict(dataset)
    return dataset_dict


def cleanup_datasets_cache(datasets: Dict[str, Dataset]):
    for split, dataset in datasets.items():
        try:
            dataset.cleanup_cache_files()
        except Exception:
            continue


def load_datasets_without_preprocessing(dataset_path: str, use_unlabeled: bool = False,
                                        use_ft: bool = False, use_val_ppl: bool = False,
                                        extra_columns: Optional[List[str]] = None,
                                        debug: bool = False) -> Dict[str, Dataset]:
    columns = [TEXT_COL, TARGET_COL] + (extra_columns if extra_columns is not None else [])
    with open(dataset_path, 'r') as f:
        dataset_dict = json.load(f)
    dataset_dict = {k: Dataset.from_dict({c: v[c] for c in columns if c in v}) for k, v in dataset_dict.items()}
    if not use_unlabeled:
        dataset_dict.pop(SPLIT_UNLABELED, None)
    if not use_ft:
        dataset_dict.pop(SPLIT_FT, None)
    if not use_val_ppl:
        dataset_dict.pop(SPLIT_VAL_PPL, None)
    if debug:
        seed = 42
        dataset_dict = {k: sample_dataset(v, 4 * BATCH_SIZE, seed, return_unsampled=False)
                        for k, v in dataset_dict.items()}
    dataset_dict = add_id_col(dataset_dict)
    print(f"Loaded dataset from {dataset_path}")
    print(f"Dataset splits: {list(dataset_dict.keys())}")
    return dataset_dict


def load_datasets(dataset_name: str, dataset_path: Optional[str] = None,
                  train_size: Optional[int] = None, val_size: Optional[int] = None, test_size: Optional[int] = None,
                  val_ppl_size: Optional[int] = None, unlabeled_size: Optional[int] = None,
                  unlabeled_split: bool = False, filter_by_length: bool = False,
                  max_input_length: Optional[int] = None, max_labels_length: Optional[int] = None,
                  add_ft_split: Union[bool, str] = False,
                  seed: Optional[int] = None, debug: bool = False, **load_dataset_kwargs) -> Dict[str, Dataset]:
    assert unlabeled_size is None or (unlabeled_split and unlabeled_size > 0)
    # load the dataset
    if dataset_name == 'xsum':
        dataset_dict = load_xsum()
    elif dataset_name == 'art':
        dataset_dict = load_art()
    elif dataset_name == 'squad':
        dataset_dict = load_squad()
    elif dataset_name == 'bisect':
        dataset_dict = load_bisect()
    elif dataset_name == 'shakespeare':
        dataset_dict = load_shakespeare()
    elif dataset_name == '' and dataset_path is not None:
        with open(dataset_path, 'r') as f:
            dataset_dict = json.load(f)
        new_dataset_dict = {}
        for k, v in dataset_dict.items():
            if 'logprobs' in v:
                v['logprobs'] = [str(x) for x in v['logprobs']]
            new_dataset_dict[k] = Dataset.from_dict(v)
        dataset_dict = new_dataset_dict
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')
    if dataset_name != '' and dataset_path is not None:
        dataset_dict.update(load_from_path(dataset_path, **load_dataset_kwargs))
    # filter examples by the length
    if max_input_length is not None or max_labels_length is not None:
        max_input_length = max_input_length if max_input_length is not None else float('inf')
        max_labels_length = max_labels_length if max_labels_length is not None else float('inf')
    if filter_by_length:
        dataset_dict = {k: ds.filter(lambda x: (len(x[TEXT_COL].split()) <= max_input_length
                                                and len(x[TARGET_COL].split()) <= max_labels_length), batched=False)
                        for k, ds in dataset_dict.items()}
    if debug:
        train_size, val_size, test_size, unlabeled_size = (
            BATCH_SIZE, max(1, BATCH_SIZE // 2), max(1, BATCH_SIZE // 2), max(1, BATCH_SIZE // 2))
        seed = 42
    # sample examples for each split if sizes are provided
    if train_size is not None and SPLIT_TRAIN in dataset_dict:
        train_size = min(train_size, len(dataset_dict[SPLIT_TRAIN]))
        if unlabeled_split and SPLIT_UNLABELED not in dataset_dict:
            dataset_dict[SPLIT_TRAIN], dataset_dict[SPLIT_UNLABELED] = sample_dataset(
                dataset_dict[SPLIT_TRAIN], train_size, seed, return_unsampled=True)
        else:
            dataset_dict[SPLIT_TRAIN] = sample_dataset(dataset_dict[SPLIT_TRAIN], train_size, seed,
                                                       return_unsampled=False)
    # add split for fine-tuning
    if ((isinstance(add_ft_split, str) and add_ft_split == 'full')
            or (not isinstance(add_ft_split, str) and add_ft_split)):
        dataset_dict[SPLIT_FT] = dataset_dict[SPLIT_TRAIN]
    elif isinstance(add_ft_split, str) and add_ft_split == 'filter':
        assert SPLIT_FT in dataset_dict
        texts = set(dataset_dict[SPLIT_TRAIN][TEXT_COL])
        dataset_dict[SPLIT_FT] = dataset_dict[SPLIT_FT].filter(lambda x: x[TEXT_COL] in texts, batched=False)
    elif isinstance(add_ft_split, str) and add_ft_split == 'split':
        assert SPLIT_FT in dataset_dict
    else:
        dataset_dict.pop(SPLIT_FT, None)
    if not unlabeled_split:
        dataset_dict.pop(SPLIT_UNLABELED, None)
    elif unlabeled_size is not None and SPLIT_UNLABELED in dataset_dict:
        dataset_dict[SPLIT_UNLABELED] = sample_dataset(dataset_dict[SPLIT_UNLABELED], unlabeled_size, seed)
    if val_size is not None and SPLIT_VAL in dataset_dict:
        # first sample the whole dev set, then sample from it to get the desired sizes
        val_size_to_sample = max(val_ppl_size, val_size) if val_ppl_size is not None else val_size
        val_size_to_sample = min(val_size_to_sample, len(dataset_dict[SPLIT_VAL]))
        val_dataset = sample_dataset(dataset_dict[SPLIT_VAL], val_size_to_sample, seed)
        dataset_dict[SPLIT_VAL] = sample_dataset(val_dataset, val_size, seed, return_unsampled=False)
        if val_ppl_size is not None:
            dataset_dict[SPLIT_VAL_PPL] = sample_dataset(val_dataset, val_ppl_size, seed,
                                                         return_unsampled=False)
    if test_size is not None and SPLIT_TEST in dataset_dict:
        test_size = min(test_size, len(dataset_dict[SPLIT_TEST]))
        dataset_dict[SPLIT_TEST] = sample_dataset(dataset_dict[SPLIT_TEST], test_size, seed, return_unsampled=False)
    dataset_dict = add_id_col(dataset_dict)
    return dataset_dict


def prepare_preprocess_function(tokenizer: PreTrainedTokenizerBase, input_col: str = TEXT_COL,
                                labels_col: str = TARGET_COL, max_input_length: int = MAX_INPUT_LENGTH,
                                max_labels_length: int = MAX_LABELS_LENGTH, model_type: str = DECODER,
                                task_prompt: Optional[str] = None, # for T5
                                add_prompt_and_reference: bool = False,  # for evaluation
                                add_eos_to_labels: bool = False,  # for decoder-only models
                                remove_bos_from_labels: bool = False,  # for decoder-only models
                                add_decoder_input_ids: bool = False,  # for encoder-decoder models - for analysis
                                decoder_input_ids_col: str = GEN_COL,  # for encoder-decoder models - for analysis
                                add_logits: bool = False,  # for pre-computed logits (for GPT-4)
                                logits_col: str = LOGITS_COL,  # for pre-computed logits (for GPT-4)
                                token_ids_col: str = TOKENS_COL):
    assert model_type in [DECODER, ENC_DEC, LANGUAGE_MODEL]
    task_prompt = task_prompt if (task_prompt is not None and task_prompt != 'bart') else ''
    if model_type == ENC_DEC and task_prompt != '' and not task_prompt.endswith(' '):
        task_prompt += ' '
    task_prompt_length = len(tokenizer(task_prompt)['input_ids']) if task_prompt != '' else 0
    max_labels_length = max_labels_length + (1 if add_eos_to_labels else 0) - (1 if remove_bos_from_labels else 0)

    def preprocess_function(examples):
        # first we need to truncate the input and leave space for the task_prompt
        tok_inputs = tokenizer(examples[input_col], max_length=max_input_length - task_prompt_length, truncation=True)
        input_texts = tokenizer.batch_decode(tok_inputs['input_ids'],
                                             skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # now we add the task_prompt and tokenize the inputs
        if model_type in [DECODER, LANGUAGE_MODEL] and task_prompt != '':
            input_texts = [text + task_prompt for text in input_texts]
        elif task_prompt != '':
            input_texts = [task_prompt + text for text in input_texts]
        tok_inputs = tokenizer(input_texts, max_length=max_input_length, truncation=True)
        # set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():  # this "with clause" is necessary for some models (e.g. MBART)
            tok_labels = tokenizer(examples[labels_col], max_length=max_labels_length, truncation=True)
        if remove_bos_from_labels:
            tok_labels['input_ids'] = [y[1:] for y in tok_labels['input_ids']]
            tok_labels['attention_mask'] = [y[1:] for y in tok_labels['attention_mask']]
        if add_eos_to_labels:
            tok_labels['input_ids'] = [y + [tokenizer.eos_token_id] for y in tok_labels['input_ids']]
            tok_labels['attention_mask'] = [y + [1] for y in tok_labels['attention_mask']]
        # preparing the input_ids and attention_mask:
        # for decoder-only and language-model we concat the input together with the labels
        # preparing the labels:
        # for decoder-only we replace the input tokens with -100, for language-model the labels are the inputs
        if model_type in [DECODER, LANGUAGE_MODEL]:
            input_ids = [x + y for x, y in zip(tok_inputs['input_ids'], tok_labels['input_ids'])]
            attention_mask = [x + y for x, y in zip(tok_inputs['attention_mask'], tok_labels['attention_mask'])]
            if model_type == DECODER:
                labels = [[LOSS_IGNORE_ID] * len(x) + y for x, y in
                          zip(tok_inputs['input_ids'], tok_labels['input_ids'])]
            else:
                labels = input_ids
        else:  # model is encoder-decoder
            input_ids = tok_inputs['input_ids']
            attention_mask = tok_inputs['attention_mask']
            labels = tok_labels['input_ids']
        batch_encoding = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        if add_prompt_and_reference:
            # preparing the references
            batch_encoding['reference'] = tok_labels['input_ids']
            # preparing the prompts for generation (input + task_prompt, not including labels)
            batch_encoding['prompt_input_ids'] = tok_inputs['input_ids']
            batch_encoding['prompt_attention_mask'] = tok_inputs['attention_mask']
        if add_decoder_input_ids:
            with tokenizer.as_target_tokenizer():  # this "with clause" is necessary for some models (e.g. MBART)
                tok_dec = tokenizer(examples[decoder_input_ids_col], max_length=max_labels_length, truncation=True)
            batch_encoding['decoder_input_ids'] = tok_dec['input_ids']
        if add_logits:
            batch_encoding['tokens_of_logits'] = examples[token_ids_col]
            batch_encoding['logits_of_tokens'] = examples[logits_col]
        return batch_encoding

    return preprocess_function


def prepare_collate_function(tokenizer: PreTrainedTokenizerBase, has_prompt_and_reference: bool = False,
                             has_id_col: bool = False, has_logits: bool = False):
    keys_for_pad = ['input_ids', 'attention_mask', 'token_type_ids', 'special_tokens_mask']

    def collate_function(examples):
        # huggingface does not support labels padding, so we need to do this "hack"
        pad_labels = tokenizer.pad([{'input_ids': example['labels']} for example in examples],
                                   padding="longest", return_tensors="pt")['input_ids']
        pad_labels[pad_labels[:, :] == tokenizer.pad_token_id] = LOSS_IGNORE_ID
        no_labels_examples = [{k: v for k, v in example.items() if k in keys_for_pad} for example in examples]
        batch_encoding = tokenizer.pad(no_labels_examples, padding="longest", return_tensors="pt")
        batch_encoding['labels'] = pad_labels
        if has_prompt_and_reference:
            # pad references
            pad_reference = tokenizer.pad([{'input_ids': example['reference']} for example in examples],
                                          padding="longest", return_tensors="pt")['input_ids']
            # padding prompts (note that model.generate() knows how to handle position_ids when there is paddings)
            pad_prompts = tokenizer.pad([{'input_ids': example['prompt_input_ids'],
                                          'attention_mask': example['prompt_attention_mask']}
                                         for example in examples], padding="longest", return_tensors="pt")
            batch_encoding['reference'] = pad_reference
            batch_encoding['prompt_input_ids'] = pad_prompts['input_ids']
            batch_encoding['prompt_attention_mask'] = pad_prompts['attention_mask']
        if has_id_col:
            batch_encoding[ID_COL] = torch.tensor([example[ID_COL] for example in examples],
                                                  device=batch_encoding['input_ids'].device)
        if has_logits:
            # first we find the batch size, the labels' length, and the vocab size
            batch_size, labels_length = batch_encoding['labels'].shape
            vocab_size = tokenizer.vocab_size
            # next, we create a tensor of shape (batch_size, labels_length, vocab_size)
            logits = torch.ones((batch_size, labels_length, vocab_size),
                                device=batch_encoding['input_ids'].device, dtype=torch.float32)
            logits *= -10000.0  # we set the logits to a very small number
            # example['logits_of_tokens'] is a list of tensors (for each example in the batch),
            # each tensor is a list of logits (for each token in the input),
            # each logits has the same size as the corresponding example['tokens_of_logits']
            # what we need to do is to copy the logits to the right place in the `logits` tensor
            for i in range(batch_size):
                example_tokens = examples[i]['tokens_of_logits'][:labels_length]
                example_logits = examples[i]['logits_of_tokens'][:labels_length]
                for j, (tokens_ids, tokens_logits) in enumerate(zip(example_tokens, example_logits)):
                    logits[i, j, tokens_ids] = torch.tensor(tokens_logits, device=logits.device, dtype=logits.dtype)
            batch_encoding['logits'] = logits
        return batch_encoding

    return collate_function


def is_training_split(split: str, include_ft: bool = True):
    return split.startswith(SPLIT_TRAIN) or (include_ft and split.startswith(SPLIT_FT))


def prepare_dataloaders(tokenizer: PreTrainedTokenizerBase, datasets: Dict[str, Dataset],
                        max_input_length: int = MAX_INPUT_LENGTH, max_labels_length: int = MAX_LABELS_LENGTH,
                        model_type: str = DECODER, task_prompt: str = TASK_PROMPT,
                        batch_size: int = BATCH_SIZE, eval_batch_size: Optional[int] = None,
                        shuffle_train: bool = False, include_id_col: bool = False, include_logits: bool = False,
                        return_list_of_train_loaders: bool = False) -> Dict[str, Union[DataLoader, List[DataLoader]]]:
    eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size

    preprocess_function = prepare_preprocess_function(
        tokenizer, max_input_length=max_input_length, max_labels_length=max_labels_length, model_type=model_type,
        task_prompt=task_prompt, add_prompt_and_reference=True,
        remove_bos_from_labels=True if model_type in [DECODER, LANGUAGE_MODEL] else False,
        add_eos_to_labels=True if model_type in [DECODER, LANGUAGE_MODEL] else False,
        add_decoder_input_ids=False, add_logits=False)
    logits_preprocess_function = prepare_preprocess_function(
        tokenizer, max_input_length=max_input_length, max_labels_length=max_labels_length, model_type=model_type,
        task_prompt=task_prompt, add_prompt_and_reference=True,
        remove_bos_from_labels=True if model_type in [DECODER, LANGUAGE_MODEL] else False,
        add_eos_to_labels=True if model_type in [DECODER, LANGUAGE_MODEL] else False,
        add_decoder_input_ids=False, add_logits=True)

    collate_function = prepare_collate_function(tokenizer, has_prompt_and_reference=True,
                                                has_id_col=include_id_col, has_logits=False)
    logits_collate_function = prepare_collate_function(tokenizer, has_prompt_and_reference=True,
                                                         has_id_col=include_id_col, has_logits=True)

    include_columns = ['input_ids', 'attention_mask', 'labels'] + ([ID_COL] if include_id_col else []) + \
                        (['logits'] if include_logits else [])

    dataloaders = {}

    for split, dataset in datasets.items():
        columns_to_remove = [c for c in dataset.column_names if c not in include_columns]
        if include_logits and is_training_split(split, include_ft=False):
            split_preprocess_function = logits_preprocess_function
            split_collate_function = logits_collate_function
        else:
            split_preprocess_function = preprocess_function
            split_collate_function = collate_function
        tokenized_dataset = dataset.map(split_preprocess_function, batched=True, remove_columns=columns_to_remove)
        split_shuffle = shuffle_train if is_training_split(split, include_ft=True) else False
        split_batch_size = batch_size if is_training_split(split, include_ft=True) else eval_batch_size
        dataloaders[split] = DataLoader(tokenized_dataset, shuffle=split_shuffle,
                                        batch_size=split_batch_size, collate_fn=split_collate_function)

    if return_list_of_train_loaders:
        splits = list(dataloaders.keys())
        train_loaders = []
        for split in splits:
            if is_training_split(split, include_ft=False):
                train_loaders.append(dataloaders[split])
                dataloaders.pop(split)
        dataloaders[SPLIT_TRAIN] = train_loaders
    return dataloaders


# ---------------------------------------------------------------------------------------------------------------------#
# -------------------------------------- Project datasets Loading Functions -------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#


def _extract_n_augmentations_per_example(dataset: Dict[str, List[str]],
                                         n_augmentations_per_example: int,
                                         divide_to_list: bool = False,
                                         extra_columns: Optional[List[str]] = None):
    columns = [TEXT_COL, TARGET_COL] + (extra_columns if extra_columns is not None else [])
    indices_per_example = defaultdict(list)
    for i, text in enumerate(dataset[TEXT_COL]):
        indices_per_example[text].append(i)
    aug_datasets = [{col: [] for col in columns}
                    for _ in range(n_augmentations_per_example if divide_to_list else 1)]
    for text, indices in indices_per_example.items():
        for dataset_index, example_index in enumerate(indices[:n_augmentations_per_example]):
            dataset_index = dataset_index if divide_to_list else 0
            aug_datasets[dataset_index][TEXT_COL].append(text)
            aug_datasets[dataset_index][TARGET_COL].append(dataset[GEN_COL][example_index])
            if extra_columns is not None:
                for col in extra_columns:
                    aug_datasets[dataset_index][col].append(dataset[col][example_index])
    return aug_datasets if divide_to_list else aug_datasets[0]


def load_from_path(dataset_path,
                   use_unlabeled: bool = False,
                   use_labeled: bool = True,
                   use_original: bool = True,
                   n_augmentations_per_example: int = 1) -> Dict[str, Dataset]:
    with open(dataset_path, 'r') as f:
        dataset_dict = json.load(f)
    original_train_dataset = set(zip(dataset_dict[SPLIT_TRAIN][TEXT_COL], dataset_dict[SPLIT_TRAIN][TARGET_COL]))
    original_train_dataset = {TEXT_COL: [text for text, _ in original_train_dataset],
                              TARGET_COL: [target for _, target in original_train_dataset]}
    train_dataset = {TEXT_COL: [], TARGET_COL: []}
    if use_original:
        train_dataset[TEXT_COL].extend(original_train_dataset[TEXT_COL])
        train_dataset[TARGET_COL].extend(original_train_dataset[TARGET_COL])
    if use_unlabeled:
        unlabeled_gens = _extract_n_augmentations_per_example(dataset_dict[SPLIT_UNLABELED],
                                                              n_augmentations_per_example)
        train_dataset[TEXT_COL].extend(unlabeled_gens[TEXT_COL])
        train_dataset[TARGET_COL].extend(unlabeled_gens[TARGET_COL])
    if use_labeled:
        labeled_gens = _extract_n_augmentations_per_example(dataset_dict[SPLIT_TRAIN],
                                                            n_augmentations_per_example)
        train_dataset[TEXT_COL].extend(labeled_gens[TEXT_COL])
        train_dataset[TARGET_COL].extend(labeled_gens[TARGET_COL])
    return {SPLIT_TRAIN: Dataset.from_dict(train_dataset), SPLIT_FT: Dataset.from_dict(original_train_dataset)}


def load_xsum() -> Dict[str, Dataset]:
    from datasets import load_dataset

    dataset_dict = load_dataset('xsum')
    dataset_dict[SPLIT_VAL] = dataset_dict.pop('validation')

    dataset_dict = {split: dataset_dict[split].rename_columns(
        {'document': TEXT_COL, 'summary': TARGET_COL}).remove_columns(['id']) for split in dataset_dict}
    return dataset_dict


def load_squad() -> Dict[str, Dataset]:
    from datasets import load_dataset

    dataset_dict = load_dataset('squad')
    dataset_dict[SPLIT_VAL], dataset_dict[SPLIT_TEST] = sample_dataset(dataset_dict.pop('validation'),
                                                                       1570, 42, return_unsampled=True)

    def preprocessing_func(example):
        new_example = {TEXT_COL: f"{example['context']}; answer: {example['answers']['text'][0]}",
                       TARGET_COL: example['question']}
        return new_example

    columns_to_remove = ['id', 'title', 'context', 'answers', 'question']
    dataset_dict = {split: dataset_dict[split].map(preprocessing_func, batched=False, remove_columns=columns_to_remove)
                    for split in dataset_dict}
    return dataset_dict


def load_art() -> Dict[str, Dataset]:
    from datasets import load_dataset

    dataset_dict = load_dataset('GEM/ART')
    dataset_dict[SPLIT_VAL] = dataset_dict.pop('validation')

    def preprocessing_func(example):
        new_example = {TEXT_COL: f"{example['observation_1']}; outcome: {example['observation_2']}",
                       TARGET_COL: example['target']}
        return new_example

    columns_to_remove = ['gem_id', 'observation_1', 'observation_2', 'references']
    dataset_dict = {split: dataset_dict[split].map(preprocessing_func, batched=False, remove_columns=columns_to_remove)
                    for split in dataset_dict}
    return dataset_dict


def load_bisect() -> Dict[str, Dataset]:
    from datasets import load_dataset, concatenate_datasets

    dataset_dict = load_dataset('GEM/BiSECT')
    dataset_dict[SPLIT_VAL] = dataset_dict.pop('validation')

    def filtering_len_func(example):
        return 20 <= len(example[TEXT_COL].split()) <= 50 and 20 <= len(example[TARGET_COL].split()) <= 50

    def filtering_overlap_func(example):
        text_set = set(example[TEXT_COL].split())
        target_set = set(example[TARGET_COL].split())
        overlap = len(text_set.intersection(target_set)) / len(text_set)
        return 0.45 <= overlap <= 0.85

    columns_to_remove = ['gem_id', 'references']
    dataset_dict = {split: dataset_dict[split].rename_columns(
        {'source': TEXT_COL, 'target': TARGET_COL}).remove_columns(columns_to_remove) for split in dataset_dict}
    dataset_dict = {split: dataset_dict[split].filter(filtering_len_func, batched=False) for split in dataset_dict}
    dataset_dict = {split: dataset_dict[split].filter(filtering_overlap_func, batched=False) for split in dataset_dict}
    dataset_dict[SPLIT_TRAIN] = sample_dataset(dataset_dict[SPLIT_TRAIN], 100000, 42, return_unsampled=False)
    dataset_dict[SPLIT_TEST] = concatenate_datasets([dataset_dict['test'], dataset_dict['challenge_bisect'],
                                                     dataset_dict['challenge_hsplit']])
    return dataset_dict


def load_shakespeare():
    shakespeare_data_path = "datasets/raw/shakespeare/shakespeare_splits.json"
    with open(shakespeare_data_path, 'r') as f:
        dataset_dict = json.load(f)
    dataset_dict['unlabeled']['modern'] = dataset_dict['unlabeled']['original']
    dataset_dict['labeled'] = {}
    dataset_dict['labeled']['original'] = dataset_dict['train']['original'] + dataset_dict['dev']['original'] + \
                                          dataset_dict['test']['original']
    dataset_dict['labeled']['modern'] = dataset_dict['train']['modern'] + dataset_dict['dev']['modern'] + \
                                        dataset_dict['test']['modern']
    dataset_dict = {split: Dataset.from_dict(dataset_dict[split]) for split in ['labeled', 'unlabeled']}
    dataset_dict = {split: dataset_dict[split].rename_columns(
        {'original': TEXT_COL, 'modern': TARGET_COL}) for split in dataset_dict}

    def filtering_len_func(example):
        return 8 <= len(example[TEXT_COL].split()) <= 50 and 8 <= len(example[TARGET_COL].split()) <= 50

    dataset_dict = {split: dataset_dict[split].filter(filtering_len_func, batched=False) for split in dataset_dict}
    labeled = dataset_dict.pop('labeled')
    dataset_dict[SPLIT_TRAIN], labeled = sample_dataset(labeled, 7000, 42, return_unsampled=True)
    dataset_dict[SPLIT_VAL], dataset_dict[SPLIT_TEST] = sample_dataset(labeled, 750, 42, return_unsampled=True)
    return dataset_dict


# ---------------------------------------------------------------------------------------------------------------------#
# --------------------------------------- Prepare and Save Datasets as JSON -------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#


def prepare_and_save_original_dataset(output_dir: str, file_name: str,
                                      dataset_name: str, dataset_path: Optional[str] = None,
                                      train_size: Optional[int] = None, val_size: Optional[int] = None,
                                      test_size: Optional[int] = None, val_ppl_size: Optional[int] = None,
                                      unlabeled_size: Optional[int] = None,
                                      unlabeled_split: bool = False, filter_by_length: bool = False,
                                      max_input_length: Optional[int] = None, max_labels_length: Optional[int] = None,
                                      add_ft_split: Union[bool, str] = False,
                                      seed: Optional[int] = None, debug: bool = False, **load_dataset_kwargs):
    datasets = load_datasets(dataset_name, dataset_path, train_size, val_size, test_size,
                             val_ppl_size, unlabeled_size, unlabeled_split, filter_by_length, max_input_length,
                             max_labels_length, add_ft_split, seed, debug, **load_dataset_kwargs)
    datasets = {split: {TEXT_COL: dataset[TEXT_COL], TARGET_COL: dataset[TARGET_COL]}
                for split, dataset in datasets.items()}
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, file_name)
    with open(save_path, 'w') as f:
        json.dump(datasets, f, indent=4)
    return datasets


def prepare_and_save_with_generations(output_dir: str, file_name: str,
                                      generation_dataset_path: str, original_dataset_path: str,
                                      use_unlabeled: bool = False, use_labeled: bool = True,
                                      use_original: bool = True, n_augmentations_per_example: int = 1):
    with open(original_dataset_path, 'r') as f:
        original_dataset = json.load(f)
    with open(generation_dataset_path, 'r') as f:
        generation_dataset = json.load(f)
    new_generation_dataset = {TEXT_COL: [], GEN_COL: []}
    if use_labeled:
        new_generation_dataset[TEXT_COL] += generation_dataset[SPLIT_TRAIN][TEXT_COL]
        new_generation_dataset[GEN_COL] += generation_dataset[SPLIT_TRAIN][GEN_COL]
    if use_unlabeled:
        new_generation_dataset[TEXT_COL] += generation_dataset[SPLIT_UNLABELED][TEXT_COL]
        new_generation_dataset[GEN_COL] += generation_dataset[SPLIT_UNLABELED][GEN_COL]
    generation_datasets = _extract_n_augmentations_per_example(new_generation_dataset, n_augmentations_per_example,
                                                               divide_to_list=True)
    if use_original:
        original_train = original_dataset.pop(SPLIT_TRAIN)
        generation_datasets = [{TEXT_COL: original_train[TEXT_COL] + generation_dataset[TEXT_COL],
                                TARGET_COL: original_train[TARGET_COL] + generation_dataset[TARGET_COL]}
                               for generation_dataset in generation_datasets]
    new_datasets = original_dataset
    for i, generation_dataset in enumerate(generation_datasets):
        new_datasets[f'{SPLIT_TRAIN}_{i}'] = generation_dataset
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, file_name)
    with open(save_path, 'w') as f:
        json.dump(new_datasets, f, indent=4)
    return new_datasets


def prepare_and_save_with_logits(output_dir: str, file_name: str, dataset_path: str,
                                 train_size: Optional[int] = None, val_size: Optional[int] = None,
                                 test_size: Optional[int] = None, filter_by_length: bool = False,
                                 max_input_length: Optional[int] = None, max_labels_length: Optional[int] = None,
                                 student_name: str = STUDENT_MODEL_NAME, teacher_new_word_char: str = ' ',
                                 n_augmentations_per_example: int = 1, use_reference_as_target: bool = False,
                                 seed: Optional[int] = None, **ignore_kwargs):
    cols = ['text', 'target', 'prediction', 'logprobs', 'tokens']
    datasets = load_datasets('', dataset_path, train_size, val_size, test_size,
                             val_size, None, False, filter_by_length, max_input_length,
                             max_labels_length, 'full', seed, False)
    new_datasets = {}
    for split, dataset in datasets.items():
        dataset = {col: dataset[col] for col in cols if col in dataset.column_names}
        if 'logprobs' in dataset:
            dataset['logprobs'] = [eval(x) for x in dataset['logprobs']]
        new_datasets[split] = dataset
    datasets = new_datasets

    for split, dataset in datasets.items():
        dataset[TEXT_COL] = dataset['text'].copy()
        dataset[GEN_COL] = dataset['prediction'].copy()
        dataset['reference'] = dataset['target'].copy()
        if split == SPLIT_TEST or use_reference_as_target:
            dataset[TARGET_COL] = dataset['target'].copy()
            if split in [SPLIT_TRAIN, SPLIT_UNLABELED]:
                dataset[GEN_COL] = dataset['target'].copy()  # for the _extract_n function
        else:
            dataset[TARGET_COL] = dataset['prediction'].copy()

        if split not in [SPLIT_TRAIN, SPLIT_UNLABELED]:
            datasets[split] = {col: dataset[col] for col in [TEXT_COL, TARGET_COL, GEN_COL, 'reference']}

    from tqdm import tqdm
    from src.align_tokens import prepare_for_kd
    from src.modeling_utils import load_tokenizer

    tokenizer = load_tokenizer(student_name)
    if SPLIT_UNLABELED in datasets:
        # concat unlabeled and train
        for c in [TEXT_COL, TARGET_COL, GEN_COL, 'logprobs']:
            if c in datasets[SPLIT_UNLABELED] and c in datasets[SPLIT_TRAIN]:
                datasets[SPLIT_TRAIN][c] += datasets[SPLIT_UNLABELED][c]
        datasets.pop(SPLIT_UNLABELED)

    # prepare the logits
    extra_columns = ['reference', 'tokens', 'logprobs']
    extra_columns = [c for c in extra_columns if c in datasets[SPLIT_TRAIN]]
    n_augmentations_per_example = 1 if n_augmentations_per_example is None else n_augmentations_per_example

    datasets[SPLIT_TRAIN]['logprobs'] = [[{w: p for w, p in token.items() if p is not None} for token in example]
                                         for example in datasets[SPLIT_TRAIN]['logprobs']]
    train_datasets = _extract_n_augmentations_per_example(datasets[SPLIT_TRAIN], n_augmentations_per_example,
                                                          divide_to_list=True,
                                                          extra_columns=extra_columns)
    for i, dataset in enumerate(train_datasets):
        new_dataset = {TEXT_COL: [], TARGET_COL: [], 'reference': [], LOGITS_COL: [], TOKENS_COL: []}
        for j in tqdm(range(len(dataset[TEXT_COL]))):
            tokens, logprobs = dataset['tokens'][j], dataset['logprobs'][j]
            text, reference = dataset[TEXT_COL][j], dataset['reference'][j]
            target, _, a_logprobs = prepare_for_kd(tokens, logprobs, tokenizer, teacher_new_word_char,
                                                   prepare_for_loader=True)
            new_dataset[TEXT_COL].append(text)
            new_dataset[TARGET_COL].append(target)
            new_dataset['reference'].append(reference)
            new_dataset[LOGITS_COL].append(a_logprobs[LOGITS_COL])
            new_dataset[TOKENS_COL].append(a_logprobs[TOKENS_COL])
        train_datasets[i] = new_dataset

    for i, dataset in enumerate(train_datasets):
        datasets[f'{SPLIT_TRAIN}_{i}'] = dataset
    datasets.pop(SPLIT_TRAIN)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, file_name)
    with open(save_path, 'w') as f:
        json.dump(datasets, f, indent=4)
    return datasets