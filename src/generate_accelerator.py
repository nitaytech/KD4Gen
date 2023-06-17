import torch
import gc
import json
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import set_seed, PreTrainedModel
from datasets import Dataset
from src.modeling_utils import load_tokenizer, load_model, generate_texts
from src.loading_utils import prepare_dataloaders
from src.train_models_accelerator import resume_from_checkpoint
from src.constants import *


def get_generation_kwargs(pad_token_id: int, num_beams: int = NUM_BEAMS, num_return_sequences: int = 1,
                          generation_type: str = 'beam_search', max_new_tokens: int = MAX_LABELS_LENGTH) \
        -> Dict[str, Any]:
    generation_kwargs = dict(pad_token_id=pad_token_id, num_return_sequences=num_return_sequences,
                             max_new_tokens=max_new_tokens, num_beams=num_beams,
                             num_beam_groups=1, do_sample=False, diversity_penalty=0.0, temperature=1.0, top_p=0.95)
    gen_modes = {
        'beam_search': dict(),
        'sampling': dict(num_beams=1, do_sample=True),
        'sampling_beam_search': dict(do_sample=True),
        'sampling_low': dict(num_beams=1, do_sample=True, temperature=0.75),
        'sampling_high': dict(num_beams=1, do_sample=True, temperature=1.5),
        'diverse_beam_search': dict(num_beam_groups=num_beams // 2, diversity_penalty=0.1),
        'beam_search_short': dict(length_penalty=0.4),
        'beam_search_long': dict(length_penalty=2),
    }
    if generation_type not in gen_modes:
        raise ValueError(f'Unknown generation type {generation_type}')
    generation_kwargs.update(gen_modes[generation_type])
    return generation_kwargs


def prepare_accelerator(model: PreTrainedModel, dataloaders: Dict[str, DataLoader],
                        mixed_precision: str = 'fp16') -> Tuple:
    # prepare accelerator and optimizer
    accelerator = Accelerator(mixed_precision=mixed_precision)
    model = model.to(accelerator.device)
    loaders_list = [dataloaders[split] for split in dataloaders]
    model, *loaders_list = accelerator.prepare(model, *loaders_list)
    dataloaders = dict(zip(list(dataloaders), loaders_list))
    return accelerator, model, dataloaders


def generate(output_dir: str,
             model_name: str,
             datasets: Dict[str, Dataset],
             checkpoint_path: Optional[str] = None,
             max_input_length: int = MAX_INPUT_LENGTH,
             max_labels_length: int = MAX_LABELS_LENGTH,
             model_type: str = DECODER,
             task_prompt: str = TASK_PROMPT,
             batch_size: int = BATCH_SIZE,
             mixed_precision: str = PRECISION,
             num_beams: int = NUM_BEAMS,
             num_return_sequences: int = 1,
             generation_type: str = 'beam_search',
             add_original_data: bool = False,
             seed: int = SEED) -> Optional[Dict[str, Dict[str, List[Any]]]]:
    set_seed(seed)
    tokenizer = load_tokenizer(model_name)
    dataloaders = prepare_dataloaders(tokenizer, datasets, max_input_length, max_labels_length, model_type,
                                      task_prompt, batch_size, eval_batch_size=batch_size, shuffle_train=False,
                                      include_id_col=True, include_logits=False, return_list_of_train_loaders=False)
    state_dict_path = checkpoint_path if (checkpoint_path is not None and not os.path.isdir(checkpoint_path)) else None
    model = load_model(model_name, model_type, state_dict_path, load_hf_seq2seq=False)
    accelerator, model, dataloaders = prepare_accelerator(model, dataloaders, mixed_precision)
    # potentially load in the weights and states from a previous save
    if checkpoint_path is not None and os.path.isdir(checkpoint_path):
        resume_from_checkpoint(accelerator, checkpoint_path, [dataloaders[list(dataloaders.keys()[0])]])
    generation_kwargs = get_generation_kwargs(tokenizer.pad_token_id, num_beams, num_return_sequences,
                                              generation_type, max_labels_length)
    generated_datasets = {}
    for split, dataloader in dataloaders.items():
        predictions, ids = generate_texts(accelerator, model, tokenizer, dataloader, model_type=model_type,
                                          return_references=False, return_ids=True, **generation_kwargs)
        if add_original_data:
            original_dataset = datasets[split]
            id2loc = {_id: loc for loc, _id in enumerate(original_dataset[ID_COL])}
            original_dataset = original_dataset.select([id2loc[_id] for _id in ids])
            original_dataset = {c: original_dataset[c] for c in original_dataset.column_names if c != ID_COL}
            original_dataset[GEN_COL] = predictions
            generated_datasets[split] = original_dataset
        else:
            generated_datasets[split] = {GEN_COL: predictions}
        accelerator.wait_for_everyone()
    if output_dir is not None and accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'generated_datasets.json'), 'w') as f:
            json.dump(generated_datasets, f, indent=4)
    accelerator.wait_for_everyone()
    del accelerator, model
    gc.collect()
    torch.cuda.empty_cache()
    return generated_datasets
