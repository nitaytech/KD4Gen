import argparse
import yaml
from azureml.core.run import Run
from src.loading_utils import load_datasets_without_preprocessing, cleanup_datasets_cache
from src.modeling_utils import get_model_type
from src.kd_models_dataset_with_logits import train_kd
from src.kd_utils import LossScheduler
from src.constants import *
from kd_main import prepare_loss_scheduler


def main(args):
    loss_scheduler = prepare_loss_scheduler(args)
    model_type = get_model_type(args.student_name)
    if args.do_train:
        epochs = args.epochs
    else:
        epochs = 0
    os.makedirs(args.output_dir, exist_ok=True)
    azureml_run = Run.get_context(allow_offline=True)
    azureml_run = azureml_run if isinstance(azureml_run, Run) else None

    datasets = load_datasets_without_preprocessing(
        dataset_path=args.dataset_path, use_unlabeled=False, use_ft=True, use_val_ppl=True,
        extra_columns=[LOGITS_COL, TOKENS_COL], debug=args.debug)
    # if you change the preprocess_function but don't clean the cache, you might load the older data (saved in cache)
    cleanup_datasets_cache(datasets)

    return train_kd(output_dir=args.output_dir, loss_scheduler=loss_scheduler,
                    datasets=datasets, student_name=args.student_name,
                    student_state_dict_path=args.student_state_dict_path,
                    checkpoint_path=args.checkpoint_path, max_input_length=args.max_input_length,
                    max_labels_length=args.max_labels_length, model_type=model_type, task_prompt=args.task_prompt,
                    metric_for_best_model=args.metric_for_best_model, greater_is_better=args.greater_is_better,
                    metrics=args.metrics, batch_size=args.batch_size, max_gpu_batch_size=args.max_gpu_batch_size,
                    learning_rate=args.learning_rate, optimizer_eps=args.optimizer_eps, weight_decay=args.weight_decay,
                    mixed_precision=args.mixed_precision, epochs=epochs, num_beams=args.num_beams,
                    save_checkpoint_every_n_steps=args.save_checkpoint_every_n_steps,
                    test_every_n_epochs=args.test_every_n_epochs,
                    keep_checkpoint_after_test=args.keep_checkpoint_after_test,
                    n_patience_epochs=args.n_patience_epochs, ft_steps_at_end=args.ft_steps_at_end,
                    stop_training_after_n_epochs=args.stop_training_after_n_epochs,
                    seed=args.seed, azureml_run=azureml_run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True,
                        help='Path to the output directory where the checkpoints, the training scores and '
                             'the generations will be saved.')
    parser.add_argument('--experiment_config_key', type=str, required=True,
                        help='Will use this key to get the experimental arguments for the loss scheduler.'
                             'Please see the file arg_configs/kd_experiment_configs.yaml for more details.')
    parser.add_argument('--dataset_path', required=True,
                        help='Path to the dataset (either a path to a file or to a directory).')
    parser.add_argument('--max_input_length', type=int, default=MAX_INPUT_LENGTH,
                        help='Maximum length of the input, If `filter_by_length` = True, then '
                             'filter out examples with a larger length (length is determined by the number of words).'
                             'Also used to truncate the tokenized input to this length.')
    parser.add_argument('--max_labels_length', type=int, default=MAX_LABELS_LENGTH,
                        help='Maximum length of the target/labels, If `filter_by_length` = True, then '
                             'filter out examples with a larger length (length is determined by the number of words).'
                             'Also used to truncate the tokenized input to this length.')
    parser.add_argument('--student_name', type=str, default=STUDENT_MODEL_NAME, help='Name of the student model.')
    parser.add_argument('--student_state_dict_path', type=str, default=None,
                        help='Path to the student model state dict.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the checkpoint')
    parser.add_argument('--task_prompt', type=str, default='summarize: ',
                        help='Task Prompt used as a suffix (for decoder-only) or as a prefix (for encoder-decoder) '
                             'for the input. If the model is a decoder then the input is: input + task_prompt + labels.')
    parser.add_argument('--metric_for_best_model', type=str, default=METRIC_FOR_EVAL,
                        help='Metric for the selecting the best model and saving the checkpoint of it.')
    parser.add_argument('--greater_is_better', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Whether the metric is greater is better (for selecting the best model)')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                        help='A string or a list of strings. Metrics for evaluating the generations.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size.')
    parser.add_argument('--max_gpu_batch_size', type=int, default=MAX_GPU_BATCH_SIZE,
                        help='Maximum batch size for GPU, if batch_size > max_gpu_batch_size'
                             ' then accumulate the gradients for `batch_size // max_gpu_batch_size` steps.'
                             ' When doing distributed training, this is the size of each node, thus the real'
                             'batch size is `(batch_size // max_gpu_batch_size) * max_gpu_batch_size * n_nodes.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate.')
    parser.add_argument('--optimizer_eps', type=float, default=OPTIMIZER_EPS, help='Optimizer epsilon.')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY, help='Weight decay.')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of training epochs.')
    parser.add_argument('--mixed_precision', type=str, default=PRECISION,
                        help='Optional values for `mixed_precision`: "no", "fp16", "bf16".')
    parser.add_argument('--num_beams', type=int, default=NUM_BEAMS, help='Number of beams for generation.')
    parser.add_argument('--save_checkpoint_every_n_steps', type=float, default=0.55,
                        help='Doing an evaluation every n steps (can be either `int` or `float`)'
                             ' and save a checkpoint if the model improves.')
    parser.add_argument('--test_every_n_epochs', type=int, default=None,
                        help='Doing an evaluation every n epochs.')
    parser.add_argument('--keep_checkpoint_after_test', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Whether to keep the checkpoint after doing an evaluation or only the model.')
    parser.add_argument('--n_patience_epochs', type=int, default=None,
                        help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--ft_steps_at_end', type=int, default=None,
                        help='Number of finetuning steps at the end of the KD.')
    parser.add_argument('--stop_training_after_n_epochs', type=int, default=None,
                        help='Number of epochs after which training will be stopped '
                             '(if provided, `epochs` is used only for the lr_scheduler).')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--debug', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Debug mode, sample small datasets.')
    parser.add_argument('--do_train', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='If false, only do generation for test dataset.')
    args = parser.parse_args()

    main(args)
