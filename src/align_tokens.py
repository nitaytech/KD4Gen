from Levenshtein import editops
from transformers import PreTrainedTokenizerBase
from src.modeling_utils import load_tokenizer
from src.constants import *


def tokens_to_str(tokens: List[str], space_char: Optional[str] = None,
                  tokenizer: Optional[PreTrainedTokenizerBase] = None) -> str:
    assert space_char is not None or tokenizer is not None
    if tokenizer is not None:
        return tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True)
    else:
        return ''.join(tokens).replace(space_char, ' ').lstrip()


def align_tokens(student_tokens: List[str],
                 teacher_tokens: List[str],
                 new_word_char: Optional[str] = None) -> List[Tuple]:
    # create a vocab of unicodes
    vocab = {w: chr(i) for i, w in enumerate(set(student_tokens + teacher_tokens))}
    s_uni, t_uni = ''.join([vocab[w] for w in student_tokens]), ''.join([vocab[w] for w in teacher_tokens])
    # calculate WER editops
    edits = editops(s_uni, t_uni)
    # for same location: insert will always come before replace
    # for subsequents: delete will always come before replace
    # delete and insert may never come together or be subsequents.
    edits = [(op, s_i, t_i,
              student_tokens[s_i] if op in ['replace', 'delete'] else None,
              teacher_tokens[t_i] if op in ['replace', 'insert'] else None) for op, s_i, t_i in edits]
    # adding equals
    m = len(student_tokens)
    n = len(teacher_tokens)
    s_is_equal = {i: True for i in range(m)}
    t_is_equal = {i: True for i in range(n)}
    for op, s_i, t_i, _, _ in edits:
        if op in ['replace', 'delete']:
            s_is_equal[s_i] = False
        if op in ['replace', 'insert']:
            t_is_equal[t_i] = False
    # we now add tuples of equal tokens: (equal, s_i, t_i, s_w, t_w)
    # for each s_is_equal[i] == True we need to find a j such that t_is_equal[j] == True and
    # student_tokens[i] == teacher_tokens[j], and also a j that we haven't used yet and is larger than the last j.
    last_j = -1
    for i in range(m):
        if s_is_equal[i]:
            for j in range(last_j + 1, n):
                if t_is_equal[j] and student_tokens[i] == teacher_tokens[j]:
                    edits.append(('equal', i, j, student_tokens[i], teacher_tokens[j]))
                    last_j = j
                    break
    # sort edits
    priority = {'insert': 0, 'delete': 1, 'replace': 2, 'equal': 3}
    edits = sorted([(s_i, priority[op], t_i, op, w, new_w) for i, (op, s_i, t_i, w, new_w) in enumerate(edits)])
    edits = [(op, s_i, t_i, w, new_w) for (s_i, _, t_i, op, w, new_w) in edits]
    # matches: equals and replaces that the student token starts with the teacher token or vice versa
    alignment = []
    for op, s_i, t_i, s_w, t_w in edits:
        if op == 'equal':
            alignment.append((op, s_i, t_i, s_w, t_w))
        elif op == 'replace':
            if new_word_char is not None:
                s_wc, t_wc = s_w.replace(new_word_char, ''), t_w.replace(new_word_char, '')
            else:
                s_wc, t_wc = s_w, t_w
            if s_wc.startswith(t_wc) or t_wc.startswith(s_wc):
                alignment.append((op, s_i, t_i, s_w, t_w))
    return alignment


def preprocess_tokens(teacher_tokens: List[str], student_tokenizer: PreTrainedTokenizerBase,
                      teacher_new_word_char: str = ' ',
                      student_new_word_char: Optional[str] = None):
    if student_new_word_char is None:
        student_new_word_char = student_tokenizer.tokenize('the')[0][0]
    text = tokens_to_str(teacher_tokens, teacher_new_word_char)
    student_tokens = student_tokenizer.tokenize(text, add_special_tokens=True)
    student_str = tokens_to_str(student_tokens, student_new_word_char, student_tokenizer)
    student_tokens = student_tokenizer.tokenize(student_str, add_special_tokens=True)
    teacher_tokens = [t.replace(teacher_new_word_char, student_new_word_char) for t in teacher_tokens]
    return student_tokens, teacher_tokens


def prepare_for_kd(teacher_tokens: List[str], logprobs: List[Dict[str, float]],
                   student_tokenizer: PreTrainedTokenizerBase,
                   teacher_new_word_char: str = ' ',
                   student_new_word_char: Optional[str] = None,
                   prepare_for_loader: bool = False) -> Tuple[str, List[str], Union[List[Dict], Dict[str, List]]]:
    if student_new_word_char is None:
        student_new_word_char = student_tokenizer.tokenize('the')[0][0]
    student_vocab = student_tokenizer.get_vocab().keys()
    student_tokens, teacher_tokens = preprocess_tokens(teacher_tokens, student_tokenizer,
                                                       teacher_new_word_char, student_new_word_char)
    student_tokens = [t for t in student_tokens if t in student_vocab]
    alignment = align_tokens(student_tokens, teacher_tokens, student_new_word_char)
    indices_alignment = {i: j for _, i, j, _, _ in alignment}
    teacher_alignment = {i: t_w for _, i, _, _, t_w in alignment}
    # valid tokens are teacher tokens (from the logprobs) that are in the vocab of the student tokenizer
    student_tokens_replaced = set([x.replace(student_new_word_char, teacher_new_word_char)
                                   for x in student_tokenizer.get_vocab().keys()])
    valid_tokens = [t for t_logprob in logprobs for t in t_logprob.keys() if t in student_tokens_replaced]
    aligned_logprobs = []
    student_str = tokens_to_str(student_tokens, student_new_word_char, student_tokenizer)
    for i, s_w in enumerate(student_tokens):
        if i not in indices_alignment:
            aligned_logprobs.append({s_w: -1e-6})
        else:
            t_w = teacher_alignment[i].replace(student_new_word_char, teacher_new_word_char)
            t_logprob = logprobs[indices_alignment[i]]
            t_logprob = {t: p for t, p in t_logprob.items() if t in valid_tokens or t == t_w}
            s_p = t_logprob.pop(t_w, -1e-6)
            t_logprob = {t.replace(teacher_new_word_char, student_new_word_char): p for t, p in t_logprob.items()}
            t_logprob[s_w] = s_p
            aligned_logprobs.append(t_logprob)
    # a few assertions to make sure everything is correct
    for t_logprob in aligned_logprobs:
        for t in t_logprob.keys():
            assert t in student_vocab
    assert student_tokenizer.tokenize(student_str, add_special_tokens=True) == student_tokens
    for i, t in enumerate(student_tokens):
        assert t in aligned_logprobs[i]
    if prepare_for_loader:
        loader_data = {TOKENS_COL: [[] for _ in range(len(aligned_logprobs))],
                       LOGITS_COL: [[] for _ in range(len(aligned_logprobs))]}
        for i, tokens_logprobs in enumerate(aligned_logprobs):
            for t, p in tokens_logprobs.items():
                loader_data[TOKENS_COL][i].append(student_tokenizer.convert_tokens_to_ids(t))
                loader_data[LOGITS_COL][i].append(p)
        aligned_logprobs = loader_data
    return student_str, student_tokens, aligned_logprobs


#
# from transformers import AutoTokenizer
# import json
#
# with open("/data/home/nitay/dev_msft/KD4Gen/datasets/for_gpt4/logits/squad_17k.json", 'r') as f:
#     d = json.load(f)['train']
#
# tokenizer = AutoTokenizer.from_pretrained('t5-small')
#
# for tokens, logprobs in zip(d['tokens'], d['logprobs']):
#     s, student_tokens, aligned_logprobs = prepare_for_kd(tokens, logprobs, tokenizer, teacher_new_word_char=' ',
#                                                          prepare_for_loader=True)
#     debug = True