from Levenshtein import editops
from string import punctuation
from src.constants import *


def tokenize_text(text: str) -> List[str]:
    puncs_trans = str.maketrans({k: f' {k} ' for k in punctuation})
    return ' '.join([c for c in text.translate(puncs_trans).split(' ') if c != '']).strip().split()


def get_wer_edits_fast(predictions: List[str], references: List[str]) -> List[Tuple[str, int, str, str]]:
    # create a vocab of unicodes
    vocab = {w: chr(i) for i, w in enumerate(set([w.lower() for w in predictions + references]))}
    p_uni, r_uni = ''.join([vocab[w.lower()] for w in predictions]), ''.join([vocab[w.lower()] for w in references])
    # calculate WER editops
    ops = editops(p_uni, r_uni)
    # for same location: insert will always come before replace
    # for subsequents: delete will always come before replace
    # delete and insert may never come together or be subsequents.
    ops = [(op, ip,
            predictions[ip] if op in ['replace', 'delete'] else None,
            references[ir] if op in ['replace', 'insert'] else None) for op, ip, ir in ops]
    return ops


def compute_wer_score(predictions: str, references: str):
    predictions, references = tokenize_text(predictions), tokenize_text(references)
    ops = get_wer_edits_fast(predictions, references)
    return len(ops) / len(references)

