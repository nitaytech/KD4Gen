import gc
import torch.cuda
from tqdm import tqdm
from transformers import set_seed
from datasets import Dataset
from azureml.core.run import Run
from src.constants import *
from src.kd_utils import loss_step, LossScheduler
from src.loading_utils import prepare_dataloaders, is_training_split
from src.train_models_accelerator import (load_tokenizer, load_model, load_metrics, prepare_dataloaders_for_accelerator,
                                          find_steps_in_epoch, resume_from_checkpoint, prepare_sizes,
                                          save_model_delete_checkpoint, prepare_accelerator, save_checkpoint,
                                          evaluate_step, do_evaluation, do_test, do_early_stopping)
from src.generate_accelerator import get_generation_kwargs


def do_student_generate(step: int, generate_labels_weight: float) -> bool:
    if generate_labels_weight >= 1:
        return True
    elif generate_labels_weight > 0.0:
        # e.g. if generate_labels_weight = 0.5, then generate labels for 50% of the steps
        # this means, every second step, i.e. every_n_steps = int(1 / generate_labels_weight)
        generate_labels_every_n_steps = int(1 / generate_labels_weight)
        if generate_labels_every_n_steps > 0 and (step + 1) % generate_labels_every_n_steps == 0:
            return True
    return False


def train_kd(output_dir: str,
             student_name: str,
             teacher_name: str,
             loss_scheduler: LossScheduler,
             datasets: Dict[str, Dataset],
             student_state_dict_path: Optional[str] = None,
             teacher_state_dict_path: Optional[str] = None,
             checkpoint_path: Optional[str] = None,
             max_input_length: int = MAX_INPUT_LENGTH,
             max_labels_length: int = MAX_LABELS_LENGTH,
             model_type: str = ENC_DEC,
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
             ft_steps_at_end: Optional[int] = None,
             stop_training_after_n_epochs: Optional[int] = None,
             test_at_end: bool = True,
             seed: int = SEED,
             azureml_run: Optional[Run] = None) -> Optional[Dict[str, Any]]:
    set_seed(seed)
    batch_size, eval_batch_size, gradient_accumulation_steps = prepare_sizes(batch_size, max_gpu_batch_size, num_beams)
    tokenizer = load_tokenizer(student_name)
    # we first split the datasets into train and the rest, the rest should include one dataset,
    # just for calculating steps
    train_datasets = {split: dataset for split, dataset in datasets.items()
                      if is_training_split(split, include_ft=False)}
    train_splits = list(train_datasets.keys())
    datasets = {split: dataset for split, dataset in datasets.items() if split not in train_datasets}
    datasets[train_splits[0]] = train_datasets[train_splits[0]]
    dataloaders = prepare_dataloaders(tokenizer, datasets, max_input_length, max_labels_length, model_type,
                                      task_prompt, batch_size, eval_batch_size,
                                      shuffle_train=True, include_id_col=False, return_list_of_train_loaders=True)
    # load the models
    if student_state_dict_path is None and checkpoint_path is not None and not os.path.isdir(checkpoint_path):
        student_state_dict_path = checkpoint_path
    student = load_model(student_name, model_type, student_state_dict_path, load_hf_seq2seq=False)
    teacher = load_model(teacher_name, model_type, teacher_state_dict_path, load_hf_seq2seq=False)
    do_warmup = False if (checkpoint_path is not None or student_state_dict_path is not None) else True
    ft_steps_at_end = ft_steps_at_end if (ft_steps_at_end is not None and ft_steps_at_end > 0
                                          and SPLIT_FT in dataloaders) else 0
    train_epochs = epochs + ft_steps_at_end
    accelerator, student, teacher, optimizer, lr_scheduler, dataloaders = prepare_accelerator(
        student, teacher, dataloaders, gradient_accumulation_steps,
        learning_rate, optimizer_eps, weight_decay, epochs, ft_steps_at_end, mixed_precision, do_warmup)
    metrics = load_metrics(metrics, metric_for_best_model)
    # potentially load in the weights and states from a previous save
    if checkpoint_path is not None:
        starting_epoch, resume_step, best_score, best_checkpoint_path = resume_from_checkpoint(
            accelerator, checkpoint_path, dataloaders[SPLIT_TRAIN])
    else:
        starting_epoch, resume_step, best_score, best_checkpoint_path = 0, 0, None, None
    last_checkpoint_path = None
    best_scores, eval_scores, early_stop = [], None, False
    os.makedirs(output_dir, exist_ok=True)
    loss_scheduler.update_loss_scales(accelerator, student, teacher,
                                      dataloaders[SPLIT_TRAIN][0], tokenizer.pad_token_id)
    # in case we want to stop the training before (and the epochs are used only for lr_scheduler)
    if isinstance(stop_training_after_n_epochs, int):
        epochs = min(epochs, starting_epoch + stop_training_after_n_epochs)
        train_epochs = epochs + ft_steps_at_end
    # now we train the model
    for epoch in range(starting_epoch, train_epochs):
        if epoch == epochs and epoch != starting_epoch:
            # if we are at the end of the training, we do the final fine-tuning
            # we reload the best checkpoint, and continue fine-tuning
            resume_from_checkpoint(accelerator, best_checkpoint_path, dataloaders[SPLIT_TRAIN])
            best_scores, early_stop = [], False
        if early_stop:
            continue
        if epoch < epochs:
            ft_stage = False
            train_split = train_splits[epoch % len(train_splits)]
            accelerator.print(f"Train split for epoch-{epoch}: {train_split}")
            train_dataset = {train_split: train_datasets[train_split]}
            train_dataloader = prepare_dataloaders(tokenizer, train_dataset, max_input_length, max_labels_length,
                                                   model_type, task_prompt, batch_size, eval_batch_size,
                                                   shuffle_train=True, include_id_col=False, include_logits=False,
                                                   return_list_of_train_loaders=True)
            dataloader = prepare_dataloaders_for_accelerator(accelerator, train_dataloader)[SPLIT_TRAIN][0]
        else:
            ft_stage = True
            dataloader = dataloaders[SPLIT_FT]
        student.train()
        teacher.eval()
        total_loss, loss_steps = 0, 0
        progress_bar = tqdm(dataloader, desc=f"{'Training' if not ft_stage else 'FineTuning'} e{epoch}")
        for step, batch in enumerate(progress_bar):
            # we need to skip steps until we reach the resumed step
            if epoch == starting_epoch and step < resume_step:
                if step > 0 and (step + 1) % gradient_accumulation_steps == 0:
                    lr_scheduler.step()
                continue
            loss_kwargs = loss_scheduler.get_loss_kwargs(return_loss_items=False, ft_stage=ft_stage)
            generate_labels_weight = loss_kwargs.pop('generate_labels_weight', 0.0)
            batch = batch.to(accelerator.device)
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            if not ft_stage and do_student_generate(step, generate_labels_weight):
                generate_kwargs = get_generation_kwargs(tokenizer.pad_token_id, num_beams=1, num_return_sequences=1,
                                                        generation_type='sampling', max_new_tokens=max_labels_length)
                with torch.no_grad():
                    student_labels = accelerator.unwrap_model(student).generate(
                        input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
                    student_labels = student_labels[:, 1:].contiguous()  # because the first token is a pad token
                loss = loss_step(student, teacher, input_ids, attention_mask, student_labels,
                                 tokenizer.pad_token_id, **loss_kwargs)
            else:
                loss = loss_step(student, teacher, input_ids, attention_mask, labels,
                                 tokenizer.pad_token_id, **loss_kwargs)
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
                    accelerator, student, tokenizer, dataloaders[SPLIT_VAL], model_type, metrics, max_labels_length,
                    num_beams, True, dataloaders.get(SPLIT_VAL_PPL, dataloaders[SPLIT_VAL]),
                    output_dir, SPLIT_VAL, epoch, step, True, True, True, best_checkpoint_path,
                    best_score, metric_for_best_model, greater_is_better, azureml_run)
                new_score = best_score if greater_is_better else -best_score
                loss_scheduler.change_stage(epoch, new_score, accelerator, student, teacher,
                                            dataloader, tokenizer.pad_token_id)
                student.train()
        progress_bar.close()
        best_scores.append(best_score)
        # we do test evaluation at the end of the KD training, or FT training or every test_every_n_epochs
        if do_test(epoch, epochs, train_epochs, test_every_n_epochs) and SPLIT_TEST in dataloaders:
            student.eval()
            step = find_steps_in_epoch(epoch, dataloaders[SPLIT_TRAIN]) - 1
            eval_scores, _, _ = evaluate_step(
                accelerator, student, tokenizer, dataloaders[SPLIT_TEST], model_type, metrics, max_labels_length,
                num_beams, True, dataloaders[SPLIT_TEST], output_dir, SPLIT_TEST, epoch, step, True, True, False,
                best_checkpoint_path, best_score, metric_for_best_model, greater_is_better, azureml_run)
            # if we are at the last epoch of the KD (not FT stage) - we need to save the last checkpoint
            if epoch == epochs - 1:
                score_str = f'{best_scores[-1]:.3f}'.replace('.', '_')
                last_checkpoint_path = os.path.join(output_dir, f'epoch{epoch}__step{step}__score{score_str}__last')
                save_checkpoint(accelerator, last_checkpoint_path, None)
        if do_early_stopping(best_scores, n_patience_epochs):
            early_stop = True
    if test_at_end and SPLIT_TEST in dataloaders:
        student.eval()
        epoch = -1
        step = -1
        if best_checkpoint_path is not None and os.path.isdir(best_checkpoint_path):
            resume_from_checkpoint(accelerator, best_checkpoint_path, dataloaders[SPLIT_TRAIN])
        eval_scores, _, _ = evaluate_step(
            accelerator, student, tokenizer, dataloaders[SPLIT_TEST], model_type, metrics, max_labels_length,
            num_beams, True, dataloaders[SPLIT_TEST], output_dir, SPLIT_TEST, epoch, step, True, True, False,
            best_checkpoint_path, best_score, metric_for_best_model, greater_is_better, azureml_run)
    if not keep_checkpoint_after_test and best_checkpoint_path is not None and train_epochs > 0:
        student_path = best_checkpoint_path + '__model.pt'
        eval_scores['checkpoint_path'] = save_model_delete_checkpoint(accelerator, best_checkpoint_path, student_path)
    elif best_checkpoint_path is not None:
        eval_scores['checkpoint_path'] = best_checkpoint_path
    else:
        eval_scores['checkpoint_path'] = (student_state_dict_path if student_state_dict_path is not None
                                          else checkpoint_path)
    eval_scores['last_checkpoint_path'] = last_checkpoint_path
    del accelerator, student, teacher, optimizer, lr_scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return eval_scores
