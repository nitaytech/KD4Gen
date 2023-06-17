import torch
import gc
from tqdm import tqdm
import shutil
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AdamW, get_linear_schedule_with_warmup, set_seed, PreTrainedTokenizerBase, PreTrainedModel
from datasets import Dataset
from azureml.core.run import Run
from src.evaluate import load_metrics
from src.modeling_utils import load_tokenizer, load_model, save_scores, save_generations, generate_texts
from src.loading_utils import prepare_dataloaders
from src.constants import *


def prepare_sizes(batch_size: int, max_gpu_batch_size: int, num_beams: int):
    # if the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > max_gpu_batch_size:
        gradient_accumulation_steps = max(1, batch_size // max_gpu_batch_size)
        batch_size = max_gpu_batch_size
    eval_batch_size = max(1, int(batch_size * 4 / num_beams))
    return batch_size, eval_batch_size, gradient_accumulation_steps


def find_steps_in_epoch(epoch: int, train_dataloaders: List[DataLoader]) -> int:
    return len(train_dataloaders[epoch % len(train_dataloaders)])


def find_total_training_steps(epochs: int, train_dataloaders: Union[DataLoader, List[DataLoader]]) -> int:
    if not isinstance(train_dataloaders, list):
        train_dataloaders = [train_dataloaders]
    return sum([find_steps_in_epoch(i, train_dataloaders) for i in range(epochs)])


def prepare_dataloaders_for_accelerator(accelerator: Accelerator,
                                        dataloaders: Dict[str, Union[DataLoader, List[DataLoader]]]) \
        -> Dict[str, Union[DataLoader, List[DataLoader]]]:
    for split, loaders in dataloaders.items():
        if isinstance(loaders, list):
            dataloaders[split] = [accelerator.prepare_data_loader(loader) for loader in loaders]
        else:
            dataloaders[split] = accelerator.prepare_data_loader(loaders)
    return dataloaders


def prepare_accelerator(model: PreTrainedModel, teacher: Optional[PreTrainedModel],
                        dataloaders: Dict[str, Union[DataLoader, List[DataLoader]]],
                        gradient_accumulation_steps: int = 1,
                        learning_rate: float = LEARNING_RATE,
                        optimizer_eps: float = OPTIMIZER_EPS,
                        weight_decay: float = WEIGHT_DECAY,
                        epochs: int = NUM_EPOCHS,
                        ft_steps_at_end: int = 0,
                        mixed_precision: str = 'fp16',
                        do_warmup: bool = False) -> Tuple:
    # prepare accelerator and optimizer
    accelerator = Accelerator(mixed_precision=mixed_precision)
    model = model.to(accelerator.device)
    optimizer = AdamW(params=model.parameters(), lr=learning_rate, eps=optimizer_eps, weight_decay=weight_decay)
    if teacher is not None:
        teacher = teacher.to(accelerator.device)
        teacher.eval()
    num_training_steps = find_total_training_steps(epochs, dataloaders[SPLIT_TRAIN]) // gradient_accumulation_steps
    if ft_steps_at_end > 0:
        num_training_steps += (len(dataloaders[SPLIT_FT]) * ft_steps_at_end) // gradient_accumulation_steps
    num_warmup_steps = min(WARMUP_STEPS, max(0, num_training_steps - WARMUP_STEPS)) if do_warmup else 0
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=num_training_steps)
    if teacher is not None:
        model, teacher, optimizer, lr_scheduler = accelerator.prepare(
            model, teacher, optimizer, lr_scheduler)
    else:
        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler)
    dataloaders = prepare_dataloaders_for_accelerator(accelerator, dataloaders)
    if teacher is not None:
        return accelerator, model, teacher, optimizer, lr_scheduler, dataloaders
    else:
        return accelerator, model, optimizer, lr_scheduler, dataloaders


def resume_from_checkpoint(accelerator: Accelerator, checkpoint_path: str,
                           train_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None) \
        -> Tuple[int, int, Optional[float], Optional[str]]:
    if os.path.isdir(checkpoint_path):
        checkpoint_path = checkpoint_path.rstrip('/')
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.wait_for_everyone()
        accelerator.load_state(checkpoint_path)
        accelerator.wait_for_everyone()
    elif '.' in checkpoint_path:
        checkpoint_path = os.path.splitext(checkpoint_path)[0]
    path = os.path.basename(checkpoint_path)
    try:
        splitted = os.path.splitext(path)[0].split('__')
        if len(splitted) == 3:
            epoch, step, score = splitted
        elif len(splitted) == 4:
            epoch, step, score, _ = splitted
        else:
            raise ValueError
        epoch = int(epoch.replace('epoch', ''))
        step = int(step.replace('step', ''))
        score = float(score.replace('score', '').replace('_', '.'))
        steps_in_epoch = find_steps_in_epoch(epoch, train_dataloaders) if train_dataloaders is not None else None
        if steps_in_epoch is not None and step >= steps_in_epoch - 1:
            epoch += 1
            step = 0
        else:
            step += 1
        accelerator.print(f"Resumed from step {step} and epoch {epoch} with score {score}")
        return epoch, step, score, checkpoint_path
    except ValueError:
        accelerator.print(f"Failed to resume from the step and epoch of checkpoint: {checkpoint_path}")
        return 0, 0, None, None


def save_checkpoint(accelerator: Accelerator,
                    new_checkpoint_path: str, checkpoint_path_to_delete: Optional[str] = None):
    accelerator.save_state(new_checkpoint_path)
    if checkpoint_path_to_delete is not None:
        if accelerator.is_main_process and os.path.exists(checkpoint_path_to_delete):
            shutil.rmtree(checkpoint_path_to_delete)
    accelerator.wait_for_everyone()


def save_model_delete_checkpoint(accelerator: Accelerator,
                                 checkpoint_path: Optional[str] = None, model_path: Optional[str] = None) -> str:
    path_to_return = checkpoint_path
    if accelerator.is_main_process and checkpoint_path is not None and os.path.exists(checkpoint_path):
        old_model_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
        new_model_path = model_path if model_path is not None else (checkpoint_path + '_pytorch_model.bin')
        if os.path.exists(old_model_path):
            shutil.copy(old_model_path, new_model_path)
            shutil.rmtree(checkpoint_path)
            path_to_return = new_model_path
        else:
            accelerator.print(f"Failed to save model from checkpoint: {checkpoint_path}")
    accelerator.wait_for_everyone()
    return path_to_return


def loss_step(model: PreTrainedTokenizerBase,
              input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor,
              step_type: str = 'forward', pad_token_id: Optional[int] = None, **kwargs):
    if pad_token_id is not None:
        labels[labels[:, :] == pad_token_id] = LOSS_IGNORE_ID
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
    return outputs.loss


def evaluate_perplexity(accelerator: Accelerator, model: PreTrainedTokenizerBase,
                        dataloader: DataLoader, pad_token_id: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    progress_bar = tqdm(dataloader, desc=f'Perplexity')
    total_loss, loss_steps = 0, 0
    for batch in progress_bar:
        batch = batch.to(accelerator.device)
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        with torch.no_grad():
            loss = loss_step(model, input_ids, attention_mask, labels, step_type='forward', pad_token_id=pad_token_id)
        total_loss += accelerator.gather(loss).detach().mean().item()
        loss_steps += 1
        progress_bar.set_postfix({'ppl': total_loss / loss_steps})
    return {'ppl': total_loss / loss_steps}


def evaluate_model(accelerator: Accelerator, model: PreTrainedTokenizerBase,
                   tokenizer: PreTrainedTokenizerBase, dataloader: DataLoader,
                   metrics: Optional[Union[List[str], str, Dict[str, Callable]]] = None,
                   metric_for_best_model: Optional[str] = METRIC_FOR_EVAL, max_new_tokens: int = MAX_LABELS_LENGTH,
                   num_beams: int = NUM_BEAMS,
                   calculate_perplexity: bool = False, perplexity_dataloader: Optional[DataLoader] = None,
                   model_type: str = DECODER, return_generations: bool = False, ignore_bertscore: bool = True):
    scores = {}
    if not isinstance(metrics, dict):
        metric_funcs = load_metrics(metrics, metric_for_best_model)
    else:
        metric_funcs = metrics
    if len(metric_funcs) > 0:
        predictions, references = generate_texts(accelerator, model, tokenizer, dataloader,
                                                 return_references=True, return_ids=False, num_beams=num_beams,
                                                 model_type=model_type,  max_new_tokens=max_new_tokens,
                                                 pad_token_id=tokenizer.pad_token_id)
        empty_indices = [i for i, (pred, ref) in enumerate(zip(predictions, references))
                         if len(pred) == 0 or len(ref) == 0]
        if len(empty_indices) > 0:
            predictions = [pred for i, pred in enumerate(predictions) if i not in empty_indices]
            references = [ref for i, ref in enumerate(references) if i not in empty_indices]
        for metric_name, metric_func in metric_funcs.items():
            if metric_name != 'bertscore' or not ignore_bertscore:
                scores.update(metric_func(predictions, references))
        if return_generations:
            scores['predictions'] = predictions
            scores['references'] = references
    if calculate_perplexity:
        perplexity_dataloader = perplexity_dataloader if perplexity_dataloader is not None else dataloader
        scores.update(evaluate_perplexity(accelerator, model, perplexity_dataloader, tokenizer.pad_token_id))
    return scores


def evaluate_step(accelerator: Accelerator, model: PreTrainedTokenizerBase,
                  tokenizer: PreTrainedTokenizerBase, dataloader: DataLoader,
                  model_type: str = DECODER, metrics: Optional[Union[List[str], str, Dict[str, Callable]]] = None,
                  max_labels_length: int = MAX_LABELS_LENGTH, num_beams: int = NUM_BEAMS,
                  calculate_perplexity: bool = False, perplexity_dataloader: Optional[DataLoader] = None,
                  output_dir: Optional[str] = None,
                  split: Optional[str] = None, epoch: Optional[int] = None, step: Optional[int] = None,
                  save_scores_to_output_dir: bool = True, save_generations_to_output_dir: bool = True,
                  save_checkpoint_to_output_dir: bool = True, best_checkpoint_path: Optional[str] = None,
                  best_score: Optional[float] = None, metric_for_best_model: str = METRIC_FOR_EVAL,
                  greater_is_better: bool = True,
                  azureml_run: Optional[Run] = None) -> Tuple[Dict[str, Any], Optional[float], Optional[str]]:
    if save_scores_to_output_dir or save_generations_to_output_dir or save_checkpoint_to_output_dir:
        assert output_dir is not None and split is not None and epoch is not None and step is not None
    ignore_bertscore = False if (split == SPLIT_TEST and epoch == -1 and step == -1) else True  # only for final test
    scores = evaluate_model(accelerator, model, tokenizer, dataloader, metrics, metric_for_best_model,
                            max_new_tokens=max_labels_length, num_beams=num_beams, model_type=model_type,
                            calculate_perplexity=calculate_perplexity, perplexity_dataloader=perplexity_dataloader,
                            return_generations=True,
                            ignore_bertscore=ignore_bertscore)
    score = scores[metric_for_best_model]
    if output_dir is not None:
        if accelerator.is_main_process:
            scores_to_save = {k: v for k, v in scores.items() if k not in ['predictions', 'references']}
            if save_scores_to_output_dir:
                csv_file_path = os.path.join(output_dir, 'eval_scores.csv')
                save_scores(scores_to_save, split, epoch, step, csv_file_path, mode='append')
            if save_generations_to_output_dir:
                save_generations(scores['predictions'], scores['references'], split, epoch, step,
                                 os.path.join(output_dir, 'eval_generations.json'), mode='append')
        accelerator.wait_for_everyone()
        if save_checkpoint_to_output_dir and (best_score is None or (greater_is_better and score > best_score) or
                                              (not greater_is_better and score < best_score)):
            score_str = f'{score:.3f}'.replace('.', '_')
            new_checkpoint_path = os.path.join(output_dir, f'epoch{epoch}__step{step}__score{score_str}')
            best_score = score
            checkpoint_path_to_delete = best_checkpoint_path
            save_checkpoint(accelerator, new_checkpoint_path, checkpoint_path_to_delete)
            best_checkpoint_path = new_checkpoint_path
    if isinstance(azureml_run, Run) and accelerator.is_main_process:  # log scores
        azureml_run.log('split', split if split is not None else 'None')
        azureml_run.log('epoch', epoch if epoch is not None else -1)
        azureml_run.log('step', step if step is not None else -1)
        for metric_name, metric_value in scores.items():
            if isinstance(metric_value, (str, int, float)):
                azureml_run.log(f'{split}_{metric_name}', metric_value)
    return scores, best_score, best_checkpoint_path


def do_evaluation(step: int, total_steps: int,
                  save_checkpoint_every_n_steps: Optional[Union[int, float]] = None) -> bool:
    # return True if it is the last step or if it is a checkpoint step
    if isinstance(save_checkpoint_every_n_steps, float):
        if save_checkpoint_every_n_steps > 1.0:
            save_checkpoint_every_n_steps = int(save_checkpoint_every_n_steps)
        else:
            save_checkpoint_every_n_steps = int(save_checkpoint_every_n_steps * total_steps)
    is_last_step = step + 1 == total_steps
    is_step_cond = (save_checkpoint_every_n_steps is not None and step > 0
                    and (step + 1) % save_checkpoint_every_n_steps == 0)
    return is_last_step or is_step_cond


def do_test(epoch: int, epochs: int, train_epochs: Optional[int] = None,
            test_every_n_epochs: Optional[Union[int, float]] = None) -> bool:
    return (do_evaluation(epoch, epochs, test_every_n_epochs) or
            (train_epochs is not None and do_evaluation(epoch, train_epochs, test_every_n_epochs)))


def do_early_stopping(best_scores: List[float], n_patience_epochs: Optional[int] = None) -> bool:
    if n_patience_epochs is None or len(best_scores) < n_patience_epochs:
        return False
    return all(score == best_scores[-1] for score in best_scores[-(n_patience_epochs + 1):])


def train_model(output_dir: str,
                model_name: str,
                datasets: Dict[str, Dataset],
                checkpoint_path: Optional[str] = None,
                max_input_length: int = MAX_INPUT_LENGTH,
                max_labels_length: int = MAX_LABELS_LENGTH,
                model_type: str = DECODER,
                task_prompt: str = TASK_PROMPT,
                metric_for_best_model: str = METRIC_FOR_EVAL,
                greater_is_better: bool = True,
                metrics: Optional[Union[str, List[str]]] = None,
                batch_size: int = BATCH_SIZE,
                max_gpu_batch_size: int = MAX_GPU_BATCH_SIZE,
                learning_rate: float = LEARNING_RATE,
                optimizer_eps: float = OPTIMIZER_EPS,
                weight_decay: float = WEIGHT_DECAY,
                epochs: int = NUM_EPOCHS,
                mixed_precision: str = PRECISION,
                num_beams: int = NUM_BEAMS,
                save_checkpoint_every_n_steps: Optional[Union[int, float]] = None,
                test_every_n_epochs: Optional[Union[int, float]] = None,
                keep_checkpoint_after_test: bool = False,
                n_patience_epochs: Optional[int] = None,
                seed: int = SEED,
                azureml_run: Optional[Run] = None) -> Optional[Dict[str, Any]]:
    set_seed(seed)
    batch_size, eval_batch_size, gradient_accumulation_steps = prepare_sizes(batch_size, max_gpu_batch_size, num_beams)
    tokenizer = load_tokenizer(model_name)
    dataloaders = prepare_dataloaders(tokenizer, datasets, max_input_length, max_labels_length, model_type,
                                      task_prompt, batch_size, eval_batch_size,
                                      shuffle_train=True, include_id_col=False, include_logits=False,
                                      return_list_of_train_loaders=True)
    state_dict_path = checkpoint_path if (checkpoint_path is not None and not os.path.isdir(checkpoint_path)) else None
    model = load_model(model_name, model_type, state_dict_path, load_hf_seq2seq=False)
    do_warmup = False if checkpoint_path is not None else True
    accelerator, model, optimizer, lr_scheduler, dataloaders = prepare_accelerator(
        model, None, dataloaders, gradient_accumulation_steps,
        learning_rate, optimizer_eps, weight_decay, epochs, 0, mixed_precision, do_warmup)
    metrics = load_metrics(metrics, metric_for_best_model)
    # potentially load in the weights and states from a previous save
    if checkpoint_path is not None:
        starting_epoch, resume_step, best_score, best_checkpoint_path = resume_from_checkpoint(
            accelerator, checkpoint_path, dataloaders[SPLIT_TRAIN])
    else:
        starting_epoch, resume_step, best_score, best_checkpoint_path = 0, 0, None, None
    best_scores = []
    eval_scores = None
    os.makedirs(output_dir, exist_ok=True)
    # now we train the model
    for epoch in range(starting_epoch, epochs):
        model.train()
        total_loss, loss_steps = 0, 0
        train_dataloader = dataloaders[SPLIT_TRAIN][epoch % len(dataloaders[SPLIT_TRAIN])]
        progress_bar = tqdm(train_dataloader, desc=f'Training e{epoch}')
        for step, batch in enumerate(progress_bar):
            # we need to skip steps until we reach the resumed step
            if epoch == starting_epoch and step < resume_step:
                if step > 0 and (step + 1) % gradient_accumulation_steps == 0:
                    lr_scheduler.step()
                continue
            batch = batch.to(accelerator.device)
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            loss = loss_step(model, input_ids, attention_mask, labels, step_type='forward',
                             pad_token_id=tokenizer.pad_token_id)
            total_loss += accelerator.gather(loss).detach().mean().item()
            loss_steps += 1
            progress_bar.set_postfix({'loss': total_loss / loss_steps})
            if azureml_run is not None and step > 0 and step % 100 == 0:
                azureml_run.log(f'train_loss', total_loss / loss_steps)
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)  # this is important to do before optimizer.step()
            if step > 0 and (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # do evaluation and save checkpoint
            if do_evaluation(step, len(progress_bar), save_checkpoint_every_n_steps):
                eval_scores, best_score, best_checkpoint_path = evaluate_step(
                    accelerator, model, tokenizer, dataloaders[SPLIT_VAL], model_type, metrics, max_labels_length,
                    num_beams, True, dataloaders.get(SPLIT_VAL_PPL, dataloaders[SPLIT_VAL]),
                    output_dir, SPLIT_VAL, epoch, step, True, True, True, best_checkpoint_path,
                    best_score, metric_for_best_model, greater_is_better, azureml_run)
                model.train()
        progress_bar.close()
        best_scores.append(best_score)
        if do_test(epoch, epochs, None, test_every_n_epochs) and SPLIT_TEST in dataloaders:
            model.eval()
            step = find_steps_in_epoch(epoch, dataloaders[SPLIT_TRAIN]) - 1
            eval_scores, _, _ = evaluate_step(
                accelerator, model, tokenizer, dataloaders[SPLIT_TEST], model_type, metrics, max_labels_length,
                num_beams, True, dataloaders[SPLIT_TEST], output_dir, SPLIT_TEST, epoch, step, True, True, False,
                best_checkpoint_path, best_score, metric_for_best_model, greater_is_better, azureml_run)
        if do_early_stopping(best_scores, n_patience_epochs):
            break
    if SPLIT_TEST in dataloaders:
        model.eval()
        epoch = -1
        step = -1
        if best_checkpoint_path is not None and os.path.isdir(best_checkpoint_path):
            resume_from_checkpoint(accelerator, best_checkpoint_path, dataloaders[SPLIT_TRAIN])
        eval_scores, _, _ = evaluate_step(
            accelerator, model, tokenizer, dataloaders[SPLIT_TEST], model_type, metrics, max_labels_length,
            num_beams, True, dataloaders[SPLIT_TEST], output_dir, SPLIT_TEST, epoch, step, True, True, False,
            best_checkpoint_path, best_score, metric_for_best_model, greater_is_better, azureml_run)
    if not keep_checkpoint_after_test and best_checkpoint_path is not None and epochs > 0:
        model_path = best_checkpoint_path + '__model.pt'
        eval_scores['checkpoint_path'] = save_model_delete_checkpoint(accelerator, best_checkpoint_path, model_path)
    elif best_checkpoint_path is not None:
        eval_scores['checkpoint_path'] = best_checkpoint_path
    else:
        eval_scores['checkpoint_path'] = state_dict_path if state_dict_path is not None else checkpoint_path
    del accelerator, model, optimizer, lr_scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return eval_scores
