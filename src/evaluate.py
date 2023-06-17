import numpy as np
from datasets import load_metric
from src.wer import compute_wer_score
from src.constants import *


def load_metrics(metrics: Optional[Union[List[str], str]] = None,
                 metric_for_best_model: Optional[str] = None) -> Dict[str, Callable]:

    possible_metrics = ['wer', 'gleu', 'rouge', 'sacrebleu', 'bertscore', 'meteor']
    if metrics is None:
        metrics = []
    elif isinstance(metrics, str):
        metrics = [metrics]
    elif isinstance(metrics, list) and len(metrics) == 1:
        metrics = metrics[0].split()
    if 'all' in metrics:
        metrics = possible_metrics
    if metric_for_best_model is not None:
        for metric in possible_metrics:
            if metric in metric_for_best_model and metric not in metrics:
                metrics.append(metric)
    metric_funcs = {}
    for metric in metrics:
        try:
            if metric == 'wer':
                def wer(predictions, references):
                    return {'wer': np.mean([compute_wer_score(p, r) for p, r in zip(predictions, references)]) * 100}

                metric_funcs[metric] = wer
            if metric == 'gleu':
                hf_gleu = load_metric('google_bleu')

                def gleu(predictions, references):
                    predictions = [t.split() for t in predictions]
                    references = [[t.split()] for t in references]
                    score_dict = {}
                    for i in [1, 2, 4]:
                        score_dict[f'gleu_{i}'] = hf_gleu.compute(predictions=predictions,
                                                                  references=references, max_len=i)['google_bleu'] * 100
                    return score_dict

                metric_funcs[metric] = gleu
            elif metric == 'rouge':
                hf_rouge = load_metric('rouge')

                def rouge(predictions, references):
                    rouge_scores = hf_rouge.compute(predictions=predictions, references=references)
                    score_dict = {}
                    for k in ['rouge1', 'rouge2', 'rougeL']:
                        score_dict.update({f'{k}_p': rouge_scores[k].mid.precision * 100,
                                           f'{k}_r': rouge_scores[k].mid.recall * 100,
                                           f'{k}_f': rouge_scores[k].mid.fmeasure * 100})
                    return score_dict

                metric_funcs[metric] = rouge

            elif metric == 'sacrebleu':
                hf_sacrebleu = load_metric('sacrebleu')

                def sacrebleu(predictions, references):
                    return {'sacrebleu': hf_sacrebleu.compute(predictions=predictions,
                                                              references=[[r] for r in references])['score']}

                metric_funcs[metric] = sacrebleu

            elif metric == 'bleurt':
                hf_bleurt = load_metric('bleurt')

                def bleurt(predictions, references):
                    return {'bleurt': np.mean(hf_bleurt.compute(predictions=predictions,
                                                                references=references)['scores'] * 100)}

                metric_funcs[metric] = bleurt

            elif metric == 'bertscore':
                hf_bertscore = load_metric('bertscore')

                def bertscore(predictions, references):
                    scores = hf_bertscore.compute(predictions=predictions, references=references,
                                                  model_type=BERTSCORE_MODEL_NAME)
                    return {f'bertscore_{k}': np.mean(scores[k]) * 100 for k in ['precision', 'recall', 'f1']}

                metric_funcs[metric] = bertscore

            elif metric == 'meteor':
                hf_meteor = load_metric('meteor')

                def meteor(predictions, references):
                    return {'meteor': hf_meteor.compute(predictions=predictions, references=references)['meteor'] * 100}

                metric_funcs[metric] = meteor
        except Exception as e:
            print(f'Error while loading metric {metric} (ignoring it):\n{type(e)}:{e}')
    return metric_funcs


def evaluate_multiple_generations(generations: Dict[str, Dict],
                                  metrics: Optional[Union[List[str], str]] = None) -> DataFrame:
    import pandas as pd
    from tqdm import tqdm
    metrics = metrics if metrics is not None else 'all'
    metric_funcs = load_metrics(metrics)
    scores = []
    progress_bar = tqdm(generations.items(), desc='Evaluating:')
    for model_name, outputs in progress_bar:
        if 'predictions' not in outputs or 'references' not in outputs:
            continue
        model_scores = {'model': model_name}
        predictions, references = outputs['predictions'], outputs['references']
        for metric_name, metric_func in metric_funcs.items():
            try:
                model_scores.update(metric_func(predictions, references))
            except Exception as e:
                print(f'Error while evaluating {metric_name} for {model_name} (ignoring it):\n{type(e)}:{e}')
        scores.append(model_scores)
    return pd.DataFrame(scores)
