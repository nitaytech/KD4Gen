import argparse
from src.loading_utils import (load_datasets_without_preprocessing, prepare_and_save_original_dataset,
                               prepare_and_save_with_generations, prepare_and_save_with_logits)
from src.constants import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True,
                        help='Path to the output_dir where the json file will be saved.')
    parser.add_argument('--file_name', type=str, default='dataset.json',
                        help='Name of the file that will be saved.')
    parser.add_argument('--dataset_name', type=str, default='xsum',
                        help='Name of the dataset, should be supported by the'
                             ' src.loading_utils.load_datasets function.')
    parser.add_argument('--dataset_path', default=None,
                        help='Path to the dataset (either a path to a file or to a directory).')
    parser.add_argument('--train_size', type=int, default=None,
                        help='Size of the training set. If None the whole train set will be used.')
    parser.add_argument('--val_size', type=int, default=None,
                        help='Size of the validation set. If None the whole val set will be used.')
    parser.add_argument('--test_size', type=int, default=None,
                        help='Size of the test set. If None the whole test set will be used.')
    parser.add_argument('--unlabeled_size', type=int, default=None,
                        help='Size of the unlabeled set. If None the whole unlabeled set will be used.')
    parser.add_argument('--val_ppl_size', type=int, default=None,
                        help='Size of the val set for ppl. If None the whole val set will be used.')
    parser.add_argument('--unlabeled_split', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Whether to use the unlabeled split or not.')
    parser.add_argument('--filter_by_length', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Whether to filter the dataset by lengths specified in'
                             ' `max_input_length` and `max_labels_length`.')
    parser.add_argument('--max_input_length', type=int, default=MAX_INPUT_LENGTH,
                        help='Maximum length of the input, If `filter_by_length` = True, then '
                             'filter out examples with a larger length (length is determined by the number of words).'
                             'Also used to truncate the tokenized input to this length.')
    parser.add_argument('--max_labels_length', type=int, default=MAX_LABELS_LENGTH,
                        help='Maximum length of the target/labels, If `filter_by_length` = True, then '
                             'filter out examples with a larger length (length is determined by the number of words).'
                             'Also used to truncate the tokenized input to this length.')
    parser.add_argument('--add_ft_split', type=str, default='full',
                        help='Can be one of full, filter or split. full for train split, split for loaded.')
    parser.add_argument('--generation_dataset_path', type=str, default=None,
                        help='Path to the generation_dataset json path.')
    parser.add_argument('--original_dataset_path', type=str, default=None,
                        help='Path to the original_dataset json path.')
    parser.add_argument('--logits_dataset_path', type=str, default=None,
                        help='Path to the logits_dataset json path (for GPT-4 KD).')
    parser.add_argument('--use_unlabeled', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--use_labeled', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--use_original', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--n_augmentations_per_example', type=int, default=None,
                        help='Number of augmentations per example in the generation dataset.')
    parser.add_argument('--load_use_unlabeled', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--load_use_ft', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--load_use_val_ppl', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--student_name', type=str, default=STUDENT_MODEL_NAME,
                        help='Name of the student model, used for tokens alignment.')
    parser.add_argument('--teacher_new_word_char', type=str, default=' ',
                        help='The character that is the prefix of words.')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--debug', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Debug mode, sample small datasets.')

    args = parser.parse_args()
    if args.original_dataset_path is None and args.generation_dataset_path is None and args.logits_dataset_path is None:
        prepare_and_save_original_dataset(output_dir=args.output_dir, file_name=args.file_name,
                                          dataset_name=args.dataset_name,
                                          dataset_path=args.dataset_path, train_size=args.train_size,
                                          val_size=args.val_size, test_size=args.test_size,
                                          val_ppl_size=args.val_ppl_size, unlabeled_size=args.unlabeled_size,
                                          unlabeled_split=args.unlabeled_split, filter_by_length=args.filter_by_length,
                                          max_input_length=args.max_input_length,
                                          max_labels_length=args.max_labels_length,
                                          add_ft_split=args.add_ft_split, seed=args.seed, debug=args.debug)
    elif args.original_dataset_path is not None and args.generation_dataset_path is not None:
        prepare_and_save_with_generations(output_dir=args.output_dir, file_name=args.file_name,
                                          generation_dataset_path=args.generation_dataset_path,
                                          original_dataset_path=args.original_dataset_path,
                                          n_augmentations_per_example=args.n_augmentations_per_example,
                                          use_unlabeled=args.use_unlabeled, use_labeled=args.use_labeled,
                                          use_original=args.use_original)
    elif args.logits_dataset_path is not None:
        if args.n_augmentations_per_example < 1:
            args.n_augmentations_per_example = 1
            use_reference_as_target = True
        else:
            use_reference_as_target = False
        prepare_and_save_with_logits(output_dir=args.output_dir, file_name=args.file_name,
                                     dataset_path=args.logits_dataset_path, train_size=args.train_size,
                                     val_size=args.val_size, test_size=args.test_size,
                                     filter_by_length=args.filter_by_length,
                                     max_input_length=args.max_input_length,
                                     max_labels_length=args.max_labels_length, student_name=args.student_name,
                                     teacher_new_word_char=args.teacher_new_word_char,
                                     n_augmentations_per_example=args.n_augmentations_per_example,
                                     use_reference_as_target=use_reference_as_target, seed=args.seed)
    elif args.dataset_path is not None:
        datasets = load_datasets_without_preprocessing(dataset_path=args.dataset_path,
                                                       use_unlabeled=args.load_use_unlabeled,
                                                       use_ft=args.load_use_ft, use_val_ppl=args.load_use_val_ppl,
                                                       debug=args.debug)
    debug = True
