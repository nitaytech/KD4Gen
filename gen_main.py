import argparse
from src.loading_utils import load_datasets_without_preprocessing, cleanup_datasets_cache
from src.modeling_utils import get_model_type
from src.generate_accelerator import generate
from src.constants import *


def gen_main(args):
    datasets = load_datasets_without_preprocessing(
        dataset_path=args.dataset_path, use_unlabeled=True, use_ft=False, use_val_ppl=False,
        extra_columns=None, debug=args.debug)
    if args.split_for_generation is not None:
        # these cases are only for supporting AML runs (for non AML runs it should be the 'else' case)
        if isinstance(args.split_for_generation, str) and ' ' in args.split_for_generation:
            splits = args.split_for_generation.split(' ')
        elif isinstance(args.split_for_generation, str):
            splits = [args.split_for_generation]
        elif (isinstance(args.split_for_generation, list) and
              len(args.split_for_generation) == 1 and ' ' in args.split_for_generation[0]):
            splits = args.split_for_generation[0].split(' ')
        else:
            splits = args.split_for_generation
        datasets = {split: dataset for split, dataset in datasets.items() if split in splits}
        print(f'Splits for generation: {splits}, found {len(datasets)} in the dataset.')
    # if you change the preprocess_function but don't clean the cache, you might load the older data (saved in cache)
    cleanup_datasets_cache(datasets)
    model_type = get_model_type(args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    return generate(output_dir=args.output_dir, model_name=args.model_name, datasets=datasets,
                    checkpoint_path=args.checkpoint_path, max_input_length=args.max_input_length,
                    max_labels_length=args.max_labels_length, model_type=model_type, task_prompt=args.task_prompt,
                    batch_size=args.batch_size, mixed_precision=args.mixed_precision, num_beams=args.num_beams,
                    num_return_sequences=args.num_return_sequences, generation_type=args.generation_type,
                    add_original_data=args.add_original_data, seed=args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True,
                        help='Path to the output directory where the checkpoints, the training scores and '
                             'the generations will be saved.')
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
    parser.add_argument('--split_for_generation', type=str, nargs='+', default=None,
                        help='Splits which will be used for generation. If None, all splits will be generated.')
    parser.add_argument('--model_name', type=str, default=TEACHER_MODEL_NAME, help='Name of the model.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the checkpoint')
    parser.add_argument('--task_prompt', type=str, default=TASK_PROMPT,
                        help='Task Prompt used as a suffix (for decoder-only) or as a prefix (for encoder-decoder) '
                             'for the input. If the model is a decoder then the input is: input + task_prompt + labels.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size.')
    parser.add_argument('--mixed_precision', type=str, default=PRECISION,
                        help='Optional values for `mixed_precision`: "no", "fp16", "bf16".')
    parser.add_argument('--num_beams', type=int, default=NUM_BEAMS, help='Number of beams for generation.')
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help='Number of sequences to return for each example.')
    parser.add_argument('--generation_type', type=str, default='beam_search',
                        help='Type of decoding method for generation: "beam_search", "sampling",'
                             ' "sampling_beam_search", "sampling_low", "sampling_high", "diverse_beam_search"')
    parser.add_argument('--add_original_data', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Whether to add the original data to the generated data.')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--debug', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Debug mode, sample small datasets.')

    args = parser.parse_args()
    gen_main(args)
