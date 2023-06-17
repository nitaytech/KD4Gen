import pandas as pd
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import ModuleList
from accelerate import Accelerator
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase,
                          PreTrainedModel, AutoModelForSeq2SeqLM)
from transformers import T5ForConditionalGeneration as hf_T5, BartForConditionalGeneration as hf_Bart
from src.hf_transformers.modeling_t5 import T5ForConditionalGeneration
from src.hf_transformers.modeling_bart import BartForConditionalGeneration
from src.constants import *


def save_scores(scores: Dict[str, Any], split: str, epoch: int, step: int, csv_file_path: str, mode: str = 'append'):
    scores = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
    scores.update({'split': split, 'epoch': epoch, 'step': step})
    new_scores = pd.DataFrame(scores, index=[0])
    if mode == 'append' and os.path.exists(csv_file_path):
        scores_df = pd.read_csv(csv_file_path)
        scores_df = pd.concat([scores_df, new_scores], axis=0)
    else:
        scores_df = new_scores
    first_columns = ['split', 'epoch', 'step']
    scores_df = scores_df[first_columns + sorted([c for c in scores_df.columns if c not in first_columns])]
    scores_df.to_csv(csv_file_path, index=False)


def save_generations(predictions: List[str], references: List[str],
                     split: str, epoch: int, step: int, json_file_path: str, mode: str = 'append'):
    key = f'{split}__epoch{epoch}__step{step}'
    new_texts = {key: {'split': split, 'epoch': epoch, 'step': step,
                       'predictions': predictions, 'references': references}}
    if mode == 'append' and os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            all_texts = json.load(f)
        all_texts.update(new_texts)
    else:
        all_texts = new_texts
    with open(json_file_path, 'w') as f:
        json.dump(all_texts, f, indent=4)


def get_model_type(model_name: str) -> str:
    if 't5' in model_name or 'bart' in model_name:
        return ENC_DEC
    elif 'gpt2' in model_name or 'opt' in model_name:
        return DECODER
    else:
        raise ValueError(f'Unknown model type: {model_name}')


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    model_name = model_name.split('_prune')[0]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False if 'opt' in model_name else True)
    if tokenizer.pad_token_id is None:
        if "~~~~~~~~~~~~~~~~" in tokenizer.vocab:
            tokenizer.pad_token_id = tokenizer.vocab["~~~~~~~~~~~~~~~~"]
            tokenizer.pad_token = "~~~~~~~~~~~~~~~~"
        else:  # this option is not that good: pad_tokens are replaced with -100 thus the eos_token will not be learnt
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, model_type: Optional[str] = None,
               state_dict_path: Optional[str] = None, load_hf_seq2seq: bool = False) -> PreTrainedModel:
    if model_type is None:
        model_type = get_model_type(model_name)
    if '_prune' in model_name:
        prune_layers = True
        model_name, prune_str = model_name.split('_prune')
    else:
        prune_layers = False
        prune_str = ''
    if model_type in [DECODER, LANGUAGE_MODEL]:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif 't5' in model_name and not load_hf_seq2seq:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    elif 't5' in model_name and load_hf_seq2seq:
        model = hf_T5.from_pretrained(model_name)
    elif 'bart' in model_name and not load_hf_seq2seq:
        model = BartForConditionalGeneration.from_pretrained(model_name)
    elif 'bart' in model_name and load_hf_seq2seq:
        model = hf_Bart.from_pretrained(model_name)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if prune_layers:
        # for legacy
        if 'bart' in model_name and prune_str == '':
            model.model.decoder.layers = ModuleList([model.model.decoder.layers[0], model.model.decoder.layers[-1]])
        else:
            if prune_str == '':
                enc_layers_to_keep = None
                dec_layers_to_keep = [0, -1]
            elif model_type == ENC_DEC and ':' in prune_str:
                enc_str, dec_str = prune_str.split(':')
                enc_layers_to_keep = [int(i) for i in enc_str.split('_') if i != ''] if enc_str != '' else None
                dec_layers_to_keep = [int(i) for i in dec_str.split('_') if i != ''] if dec_str != '' else None
            else:
                enc_layers_to_keep = None
                dec_layers_to_keep = [int(i) for i in prune_str.split('_') if i != ''] if prune_str != '' else None
            if enc_layers_to_keep is not None:
                # bart like
                if (hasattr(model, 'model') and hasattr(model.model, 'encoder') and
                        hasattr(model.model.encoder, 'layers')):
                    enc_layers_to_keep = [i if i >= 0 else len(model.model.encoder.layers) + i
                                          for i in enc_layers_to_keep]
                    model.model.encoder.layers = ModuleList([model.model.encoder.layers[i] for i in enc_layers_to_keep])
                # t5 like
                elif hasattr(model, 'encoder') and hasattr(model.encoder, 'block'):
                    enc_layers_to_keep = [i if i >= 0 else len(model.encoder.block) + i
                                          for i in enc_layers_to_keep]
                    model.encoder.block = ModuleList([model.encoder.block[i] for i in enc_layers_to_keep])
                else:
                    raise ValueError(f'Unsupported model architecture for prunning: {model_name+prune_str}')
            if dec_layers_to_keep is not None:
                # bart and opt like
                if (hasattr(model, 'model') and hasattr(model.model, 'decoder')
                        and hasattr(model.model.decoder, 'layers')):
                    dec_layers_to_keep = [i if i >= 0 else len(model.model.decoder.layers) + i
                                          for i in dec_layers_to_keep]
                    model.model.decoder.layers = ModuleList([model.model.decoder.layers[i] for i in dec_layers_to_keep])
                # t5 like
                elif hasattr(model, 'decoder') and hasattr(model.decoder, 'block'):
                    dec_layers_to_keep = [i if i >= 0 else len(model.decoder.block) + i
                                          for i in dec_layers_to_keep]
                    model.decoder.block = ModuleList([model.decoder.block[i]
                                                            for i in dec_layers_to_keep])
                # gpt like
                elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                    dec_layers_to_keep = [i if i >= 0 else len(model.transformer.h) + i
                                          for i in dec_layers_to_keep]
                    model.transformer.h = ModuleList([model.transformer.h[i] for i in dec_layers_to_keep])
                else:
                    raise ValueError(f'Unsupported model architecture for prunning: {model_name+prune_str}')
    if state_dict_path is not None:
        model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    return model


def generate_texts(accelerator: Accelerator, model: PreTrainedTokenizerBase,
                   tokenizer: PreTrainedTokenizerBase, dataloader: DataLoader,
                   model_type: str = DECODER, return_references: bool = True, return_ids: bool = False,
                   **generate_kwargs) -> Union[List[str], Tuple]:
    num_return_sequences = generate_kwargs.get('num_return_sequences', 1)
    model.eval()
    predictions, references, ids = [], [], []
    progress_bar = tqdm(dataloader, desc=f'Generating')
    pad_token = tokenizer.pad_token
    for batch in progress_bar:
        batch = batch.to(accelerator.device)
        input_ids, attention_mask = batch['prompt_input_ids'], batch['prompt_attention_mask']
        with torch.no_grad():
            outputs = accelerator.unwrap_model(model).generate(input_ids=input_ids,
                                                               attention_mask=attention_mask, **generate_kwargs)
            if model_type in [DECODER, LANGUAGE_MODEL]:
                outputs = outputs[:, input_ids.shape[1]:]  # truncating the prompt if decoder-only model
            outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
            outputs = accelerator.gather(outputs)
            generated_text = tokenizer.batch_decode(outputs,
                                                    skip_special_tokens=True, clean_up_tokenization_spaces=True)
            generated_text = [t.replace(pad_token, '') for t in generated_text]
            predictions.extend(generated_text)
        if return_references:
            reference = batch['reference']
            reference = accelerator.pad_across_processes(reference, dim=1, pad_index=tokenizer.pad_token_id)
            reference = accelerator.gather(reference)
            reference[reference[:, :] == LOSS_IGNORE_ID] = tokenizer.pad_token_id
            decoded_reference = tokenizer.batch_decode(reference,
                                                       skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_references = [t for t in decoded_reference for _ in range(num_return_sequences)]
            decoded_references = [t.replace(pad_token, '') for t in decoded_references]
            references.extend(decoded_references)
        if return_ids:
            batch_ids = batch[ID_COL]
            batch_ids = accelerator.gather(batch_ids)
            batch_ids = [_id for _id in batch_ids.tolist() for _ in range(num_return_sequences)]
            ids.extend(batch_ids)
    if not return_references and not return_ids:
        return predictions
    if return_references and not return_ids:
        return predictions, references
    if return_ids and not return_references:
        return predictions, ids
    return predictions, references, ids
