import argparse
import torch
import numpy as np
import pandas as pd
import os
from typing import Callable, Optional, List, Union
from transformers import PreTrainedTokenizerBase, PreTrainedModel, AutoTokenizer
from src.modeling_utils import load_model as load_model_util


def get_total_memory(device):
    return torch.cuda.get_device_properties(device).total_memory // 1000000


def get_peak_memory(device):
    stats = torch.cuda.memory_stats_as_nested_dict(device=device)
    if 'allocated_bytes' not in stats:
        return 0
    return stats['allocated_bytes']['all']['peak'] // 1000000


def measure_latency_and_memory(model_call_function: Callable, device: Optional[Union[int, str]],
                               n_calls: int, warmup_calls: int = 10):
    memories, latencies = [], []
    # warm up
    with torch.no_grad():
        for _ in range(warmup_calls):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            model_call_function()
    with torch.no_grad():
        for _ in range(n_calls):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            before_memory = get_peak_memory(device)
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            model_call_function()
            ender.record()
            torch.cuda.synchronize()
            latency = starter.elapsed_time(ender)

            after_memory = get_peak_memory(device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            memories.append(after_memory - before_memory)
            latencies.append(latency)

    return {'time': np.mean(latencies), 'time_std': np.std(latencies),
            'mem': np.mean(memories), 'mem_std': np.std(memories)}


def measure_flops(model, tokenizer: PreTrainedTokenizerBase,
                  batch_size: int, input_length: int, target_length: int,
                  for_decoder_only: bool):
    inputs = prepare_model_inputs(tokenizer, batch_size, input_length, target_length, model.device, for_decoder_only,
                                  for_generation=False, for_flops=True, num_beams=1)

    from torchprofile import profile_macs

    forward_input_names = model.forward.__code__.co_varnames[:model.forward.__code__.co_argcount]
    max_input_idx = max([i for i, n in enumerate(forward_input_names) if n in inputs])
    input_args = [inputs.get(name, None) for name in forward_input_names[:max_input_idx + 1] if name != 'self']
    input_args = tuple(input_args)

    model = model.eval()
    macs = profile_macs(model, input_args)
    return {'FLOPs': macs * 2}


def prepare_model_inputs(tokenizer: PreTrainedTokenizerBase,
                         batch_size: int, input_length: int, target_length: int,
                         device: Optional[Union[int, str]],
                         for_decoder_only: bool, for_generation: bool, for_flops: bool = False,
                         num_beams: int = 1):
    assert not (for_generation and for_flops)
    input_texts = [['='] * input_length] * batch_size
    tok_inputs = tokenizer(input_texts, add_special_tokens=False, is_split_into_words=True)
    if for_generation:
        return {'input_ids': torch.tensor(tok_inputs['input_ids'], device=device, dtype=torch.long),
                'attention_mask': torch.tensor(tok_inputs['attention_mask'], device=device, dtype=torch.long),
                'max_new_tokens': target_length,
                'min_length': target_length if not for_decoder_only else input_length + target_length,
                'num_return_sequences': 1, 'num_beams': num_beams, 'num_beam_groups': 1, 'do_sample': False}
    # set up the tokenizer for targets
    target_texts = [['='] * target_length] * batch_size
    with tokenizer.as_target_tokenizer():  # this "with clause" is necessary for some models (e.g. MBART)
        tok_labels = tokenizer(target_texts, add_special_tokens=False, is_split_into_words=True)
    # preparing the input_ids and attention_mask:
    # for decoder-only we concat the input together with the labels
    # preparing the labels:
    # for decoder-only we replace the input tokens with -100
    if for_decoder_only:
        input_ids = [x + y for x, y in zip(tok_inputs['input_ids'], tok_labels['input_ids'])]
        attention_mask = [x + y for x, y in zip(tok_inputs['attention_mask'], tok_labels['attention_mask'])]
        labels = [[-100] * len(x) + y for x, y in zip(tok_inputs['input_ids'], tok_labels['input_ids'])]
    else:  # model is encoder-decoder
        input_ids = tok_inputs['input_ids']
        attention_mask = tok_inputs['attention_mask']
        labels = tok_labels['input_ids']
    input_ids = torch.tensor(input_ids, device=device, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, device=device, dtype=torch.long)
    labels = torch.tensor(labels, device=device, dtype=torch.long)
    if not for_flops:
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    else:
        if for_decoder_only:
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            decoder_attention_mask = torch.tensor(tok_labels['attention_mask'], device=device, dtype=torch.long)
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'decoder_input_ids': labels,
                    'decoder_attention_mask': decoder_attention_mask}


def profile_forward(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                    batch_size: int, input_length: int, target_length: int,
                    device: Optional[Union[int, str]], for_decoder_only: bool,
                    n_calls: int = 100, warmup_calls: int = 10):
    model = model.to(device)
    inputs = prepare_model_inputs(tokenizer, batch_size, input_length, target_length, device, for_decoder_only,
                                  for_generation=False, for_flops=False, num_beams=1)

    def forward_func():
        return model(**inputs)

    forward_stats = measure_latency_and_memory(forward_func, device, n_calls, warmup_calls)
    return {f'fw_{k}': v for k, v in forward_stats.items()}


def profile_generate(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                     batch_size: int, input_length: int, target_length: int,
                     device: Optional[Union[int, str]], for_decoder_only: bool, num_beams: int,
                     n_calls: int = 100, warmup_calls: int = 10):
    model = model.to(device)
    inputs = prepare_model_inputs(tokenizer, batch_size, input_length, target_length, device, for_decoder_only,
                                  for_generation=True, for_flops=False, num_beams=num_beams)

    def generate_func():
        gen_ids = model.generate(**inputs)
        assert gen_ids.shape[1] >= target_length

    generate_stats = measure_latency_and_memory(generate_func, device, n_calls, warmup_calls)
    return {f'gen_{k}': v for k, v in generate_stats.items()}


def find_max_batch_size_for_gen(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                                input_length: int, target_length: int, device: Optional[Union[int, str]],
                                for_decoder_only: bool, num_beams: int, power_of_two: bool = False,
                                return_list: bool = False) -> Union[int, List[int]]:
    model = model.to(device)
    possible_sizes = []
    prev_size, batch_size, addition, max_size = 0, 0, 1, None
    while True:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        try:
            batch_size = prev_size + addition
            if max_size is not None and batch_size >= max_size:
                raise RuntimeError('Max size reached')
            inputs = prepare_model_inputs(tokenizer, batch_size, input_length, target_length, device, for_decoder_only,
                                          for_generation=True, for_flops=False, num_beams=num_beams)
            gen_ids = model.generate(**inputs)
            assert gen_ids.shape[1] >= target_length
            memory = get_peak_memory(device)
            # print(prev_size, batch_size, addition, memory, available_memory)
            addition *= 2
            possible_sizes.append(batch_size)
        except RuntimeError as e:
            prev_size = prev_size if not power_of_two else prev_size + addition // 2
            if power_of_two or addition <= 1:
                print(f'Batch size {prev_size} is the maximum for generation')
                return prev_size if not return_list else possible_sizes
            max_size = batch_size
            addition = 1
            print(f'Batch size {max_size} is too big for generation, reducing to {prev_size}')


def find_max_batch_size_for_gen_v2(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                                   input_length: int, target_length: int, device: Optional[Union[int, str]],
                                   for_decoder_only: bool, num_beams: int,
                                   maximum_memory: Optional[int] = None) -> Union[int, List[int]]:
    model = model.to(device)
    if maximum_memory is None:
        maximum_memory = int(0.95 * get_total_memory(device))
    batch_one_mem = int(profile_generate(model, tokenizer, 1, input_length, target_length, device, for_decoder_only,
                                         num_beams, n_calls=1, warmup_calls=1)['gen_mem'])
    model_mem = get_peak_memory(device)
    max_batch_size = int((maximum_memory - model_mem) // batch_one_mem)

    def check_batch_size(batch_size: int) -> bool:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        try:
            inputs = prepare_model_inputs(tokenizer, batch_size, input_length, target_length, device, for_decoder_only,
                                          for_generation=True, for_flops=False, num_beams=num_beams)
            gen_ids = model.generate(**inputs)
            assert gen_ids.shape[1] >= target_length
            memory = get_peak_memory(device)
            print(f'Batch size {batch_size} is OK, memory {memory}')
            return memory < maximum_memory
        except RuntimeError as e:
            print(f'Batch size {batch_size} is too big')
            return False

    # conduct binary search to find the maximum batch size
    low, high = 1, max_batch_size
    while low < high:
        mid = (low + high + 1) // 2
        if check_batch_size(mid):
            low = mid
        else:
            high = mid - 1
    print(f'Batch size {low} is the maximum for generation')
    return low


def measure_model_memory(model, device: Optional[Union[int, str]]):
    model_device = next(model.parameters()).device
    model = model.to(device).cpu()  # need to init the device
    summary = {'before_empty': get_peak_memory(device)}
    torch.cuda.reset_max_memory_allocated(device)
    torch.cuda.reset_max_memory_cached(device)
    torch.cuda.empty_cache()
    before_memory = get_peak_memory(device)
    model = model.to(device)
    model = model.cpu()
    after_memory = get_peak_memory(device)
    torch.cuda.reset_max_memory_allocated(device)
    torch.cuda.reset_max_memory_cached(device)
    torch.cuda.empty_cache()
    summary['after_empty'] = get_peak_memory(device)
    summary['before'] = before_memory
    summary['after'] = after_memory
    # print(iter_summary)
    model = model.to(model_device)
    return after_memory - before_memory


def model_num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_prune_str_to_model_name(model_name: str,
                                enc_layers_to_keep: Optional[List[int]] = None,
                                dec_layers_to_keep: Optional[List[int]] = None):
    enc_layers_to_keep = None if enc_layers_to_keep is None else '_'.join([str(x) for x in enc_layers_to_keep])
    dec_layers_to_keep = None if dec_layers_to_keep is None else '_'.join([str(x) for x in dec_layers_to_keep])
    if enc_layers_to_keep is not None and dec_layers_to_keep is not None:
        model_name += f'_prune' + enc_layers_to_keep + ':' + dec_layers_to_keep
    elif dec_layers_to_keep is not None:
        model_name += f'_prune' + dec_layers_to_keep
    elif enc_layers_to_keep is not None:
        model_name += f'_prune' + enc_layers_to_keep + ':'
    return model_name


def load_model(model_name: str,
               enc_layers_to_keep: Optional[List[int]] = None,
               dec_layers_to_keep: Optional[List[int]] = None):
    return load_model_util(add_prune_str_to_model_name(model_name, enc_layers_to_keep, dec_layers_to_keep),
                           load_hf_seq2seq=True)


def load_tokenizer(model_name: str):
    model_name = model_name.split('_prune')[0]
    # the add_prefix_space is needed for bart/opt/gpt models for using `is_split_into_words`.
    # DONT use tokenizers initialized with `is_split_into_words=True` for training!!!
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False if 'opt' in model_name else True,
                                              add_prefix_space=True)
    return tokenizer


def main(args):
    model_names = args.model_name if isinstance(args.model_name, list) else [args.model_name]
    batch_sizes = args.batch_size if isinstance(args.batch_size, list) else [args.batch_size]
    input_lengths = args.input_length if isinstance(args.input_length, list) else [args.input_length]
    target_lengths = args.target_length if isinstance(args.target_length, list) else [args.target_length]
    is_paired = args.paired_lengths
    if is_paired:
        assert len(input_lengths) == len(target_lengths), "When `is_paired=True`, `input_length` and `target_lengths`" \
                                                          "should contain the same number of lengths."
    num_beams = args.num_beams if isinstance(args.num_beams, list) else [args.num_beams]
    enc_layers_to_keep = args.enc_layers_to_keep
    dec_layers_to_keep = args.dec_layers_to_keep
    n_calls = args.n_calls
    warmup_calls = args.warmup_calls
    maximum_memory = args.maximum_memory
    device = args.device if not args.device.isdigit() else int(args.device)
    output_file_path = args.output_file_path
    if args.find_maximal_batch_size:
        batch_sizes = [-2] + batch_sizes
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    compute_stats = []
    for model_name in model_names:
        model = load_model(model_name, enc_layers_to_keep, dec_layers_to_keep)
        tokenizer = load_tokenizer(model_name)
        for_decoder_only = not model.config.is_encoder_decoder
        model_stats = {'model_name': model_name, 'num_params': model_num_parameters(model), 'device': device}
        if device != 'cpu':
            model_stats['model_mem'] = measure_model_memory(model, device)
        else:
            model_stats['model_mem'] = 0
        model = model.to(device)
        for i, input_length in enumerate(input_lengths):
            for j, target_length in enumerate(target_lengths):
                if is_paired and i != j:
                    continue
                for batch_size in batch_sizes:
                    find_maximal_batch_size = True if batch_size == -2 else False
                    stats_without_gen = model_stats.copy()
                    stats_without_gen.update({'batch_size': batch_size, 'input_length': input_length,
                                              'target_length': target_length})
                    if not find_maximal_batch_size:
                        # flops
                        stats_without_gen.update(measure_flops(model, tokenizer, batch_size, input_length,
                                                               target_length, for_decoder_only))
                        # forward pass
                        stats_without_gen.update(profile_forward(model, tokenizer, batch_size, input_length,
                                                                 target_length, device, for_decoder_only,
                                                                 n_calls, warmup_calls))
                    # generation
                    for num_beam in num_beams:
                        if find_maximal_batch_size:
                            batch_size = find_max_batch_size_for_gen_v2(
                                model, tokenizer, input_length, target_length, device, for_decoder_only,
                                num_beams=num_beam, maximum_memory=maximum_memory)

                        stats = stats_without_gen.copy()
                        stats.update({'batch_size': batch_size})
                        stats.update({'num_beams': num_beam})
                        stats.update(profile_generate(model, tokenizer, batch_size, input_length,
                                                      target_length, device, for_decoder_only, num_beam,
                                                      n_calls, warmup_calls))
                        # examples per minute
                        stats['throughput'] = stats['batch_size'] * 60 / (stats['gen_time'] / 1000)
                        compute_stats.append(stats)
                        print(str(stats)[1:-1])
                        pd.DataFrame(compute_stats).to_csv(output_file_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file_path', type=str, required=True,
                        help='Path to the output file where the benchmark results will be saved.')
    parser.add_argument('--model_name', type=str, nargs='+', required=True,
                        help='Name of the model to benchmark. Can be a list of models.')
    parser.add_argument('--batch_size', type=int, nargs='+', default=[1],
                        help='Batch size to benchmark. Can be a list of batch sizes.')
    parser.add_argument('--input_length', type=int, nargs='+', default=[128],
                        help='Input length to benchmark. Can be a list of input lengths.')
    parser.add_argument('--target_length', type=int, nargs='+', default=[32],
                        help='Target length to benchmark. Can be a list of target lengths.')
    parser.add_argument('--paired_lengths', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='If True, measuring performance for pairs of length after zipping'
                             '`input_length` and `target_length`:'
                             'for x, y in `zip(input_length,target_length):`. '
                             'Otherwise, measuring for all combination of `input_length` and `target_length`:'
                             '`for x in input_length: for y in target_length:`')
    parser.add_argument('--num_beams', type=int, nargs='+', default=[1],
                        help='Number of beams to benchmark. Can be a list of number of beams.')
    parser.add_argument('--enc_layers_to_keep', type=int, nargs='+', default=None,
                        help='Encoder layers to keep when doing prunning, if None all layers are kept.')
    parser.add_argument('--dec_layers_to_keep', type=int, nargs='+', default=None,
                        help='Decoder layers to keep when doing prunning, if None all layers are kept.')
    parser.add_argument('--n_calls', type=int, default=100,
                        help='Number of calls for forward or generate function to measure latency and memory.')
    parser.add_argument('--warmup_calls', type=int, default=10,
                        help='Number of warmup calls for forward or generate function to measure latency and memory.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to benchmark on.')
    parser.add_argument('--find_maximal_batch_size', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='If True, the script will find maximal batch size for the model and input lengths.')
    parser.add_argument('--maximum_memory', type=int, default=None,
                        help='Maximum memory (MB) to use for finding the maximal batch size. If None uses all memory')
    args = parser.parse_args()
    main(args)


"""

example of command line
CUDA_VISIBLE_DEVICES=1 python profile_main.py --output_file_path outputs/profiling_32.csv --model_name t5-small t5-large facebook/bart-base facebook/bart-large gpt2 gpt2-medium gpt2-large facebook/opt-125m facebook/opt-350m facebook/bart-base_prune0_1: facebook/bart-base_prune0_1 --batch_size 1 --input_length 48 320 480 --target_length 32 --num_beams 1 --n_calls 25 --warmup_calls 5 --device 0 --find_maximal_batch_size True --maximum_memory 16000


CUDA_VISIBLE_DEVICES=1 python profile_main.py --output_file_path outputs/profiling_pruning.csv --model_name facebook/bart-base facebook/bart-base_prune0_1: facebook/bart-base_prune0_1 --batch_size 1 --input_length 1 16 32 64 128 192 224 240 256 --target_length 256 240 224 192 128 64 32 16 1 --paired_lengths True --num_beams 1 --n_calls 100 --warmup_calls 5 --device 0 --find_maximal_batch_size True --maximum_memory 16000

"""
