import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch import Tensor
from src.hf_transformers.top_p_logits_warper import TopPLogitsWarper
from src.constants import *


def _self_relations(states: Tensor) -> Tensor:
    # states.shape = (batch_size, seq_length, dim)
    state_dim = states.shape[-1]
    return torch.matmul(states, states.transpose(-1, -2)) / (state_dim ** 0.5)


def cosine_similarity(tensor1: Tensor, tensor2: Tensor, eps: float = 1e-8):
    tensor1_norm = tensor1.norm(dim=2).unsqueeze(2)
    tensor2_norm = tensor2.norm(dim=2).unsqueeze(2)
    tensor1_normed = tensor1 / torch.max(tensor1_norm, eps * torch.ones_like(tensor1_norm))
    tensor2_normed = tensor2 / torch.max(tensor2_norm, eps * torch.ones_like(tensor2_norm))
    return torch.bmm(tensor1_normed, tensor2_normed.transpose(1, 2))


def construct_attention_mask_for_relations_matrix(attention_mask: Tensor, causal_mask: bool = False) -> Tensor:
    with torch.no_grad():
        n = attention_mask.shape[1]
        attention_mask_matrix = torch.einsum('bi, bj -> bij', attention_mask, attention_mask)
        if causal_mask:  # need to mask future tokens
            causal_matrix = torch.tril(torch.ones((n, n), dtype=attention_mask_matrix.dtype,
                                                  device=attention_mask_matrix.device)).view(1, n, n)
            attention_mask_matrix *= causal_matrix
    return attention_mask_matrix


def construct_attention_mask_for_logits(attention_mask: Tensor, logits: Tensor) -> Tensor:
    assert logits.ndim == 3  # (batch_size, seq_length, vocab_size)
    assert attention_mask.shape[0] == logits.shape[0]
    assert attention_mask.shape[1] == logits.shape[1]
    with torch.no_grad():
        attention_mask_matrix = torch.ones_like(logits, device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask_matrix *= attention_mask.unsqueeze(-1)
    return attention_mask_matrix


def rmse_with_masks(student_tensor: Tensor, teacher_tensor: Tensor, attention_mask: Tensor, causal_mask: bool = False,
                    is_relations: bool = True):
    assert student_tensor.ndim == 3  # (batch_size, seq_length, dim)
    assert student_tensor.size() == teacher_tensor.size()
    assert student_tensor.shape[0] == attention_mask.shape[0]
    assert student_tensor.shape[1] == attention_mask.shape[1]
    teacher_tensor = teacher_tensor.type(student_tensor.dtype)
    if is_relations:
        assert student_tensor.shape[1] == student_tensor.shape[2]
        attention_mask_matrix = construct_attention_mask_for_relations_matrix(attention_mask, causal_mask)
    else:
        attention_mask_matrix = construct_attention_mask_for_logits(attention_mask, student_tensor)
    n_relations = attention_mask_matrix.sum(dim=(1, 2)).view(-1, 1, 1)
    student_tensor *= attention_mask_matrix
    teacher_tensor *= attention_mask_matrix
    loss = torch.mean((torch.sum((student_tensor - teacher_tensor) ** 2 / n_relations, dim=(1, 2))) ** 0.5)
    return loss


def kl_with_masks(student_tensor: Tensor, teacher_tensor: Tensor, attention_mask: Tensor, causal_mask: bool = False,
                  is_relations: bool = True):
    assert student_tensor.ndim == 3  # (batch_size, seq_length, seq_length/vocab_size)
    assert student_tensor.size() == teacher_tensor.size()
    assert student_tensor.shape[0] == attention_mask.shape[0]
    assert student_tensor.shape[1] == attention_mask.shape[1]
    teacher_tensor = teacher_tensor.type(student_tensor.dtype)
    if is_relations:
        assert student_tensor.shape[1] == student_tensor.shape[2]
        attention_mask_matrix = construct_attention_mask_for_relations_matrix(attention_mask, causal_mask)
    else:
        attention_mask_matrix = construct_attention_mask_for_logits(attention_mask, student_tensor)
    # we apply softmax on the relations, therefore we multiply the masked tokens with -10000.0 (e^-10000 ~= 0)
    student_tensor += (1 - attention_mask_matrix) * -10000.0
    teacher_tensor += (1 - attention_mask_matrix) * -10000.0
    # now we calculate the pointwise KL (i.e., t[b,i,j] * (log(t[b,i,j]) - log(s[b,i,j])))
    pw_kl = torch.kl_div(torch.log_softmax(student_tensor, dim=-1), torch.softmax(teacher_tensor, dim=-1))
    # next we calculate the normalized KL for each token (sum over the pointwise KL)
    pw_kl *= attention_mask_matrix
    pw_kl = pw_kl.sum(dim=-1)  # shape = (batch_size, seq_length)
    # finally, we calculate the mean KL of *each example* in the batch, while ignoring the masked tokens KL scores
    seq_length = attention_mask.sum(dim=-1)
    seq_length = torch.max(seq_length, torch.ones_like(seq_length))
    pw_kl = pw_kl.sum(dim=-1) / seq_length  # shape = (batch_size, )
    return pw_kl.mean()  # only here we take the mean over the batch


def distil_logits_loss(student_logits: Tensor, teacher_logits: Tensor, labels_mask: Tensor, temperature: float = 1.0):
    assert teacher_logits.size() == student_logits.size()
    assert student_logits.shape[0] == labels_mask.shape[0]
    assert student_logits.shape[1] == labels_mask.shape[1]
    teacher_logits = teacher_logits.type(student_logits.dtype)
    if temperature > 0:  # if temperature is positive: soften probabilities (higher temperature -> more soft)
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
    else:  # if negative, we do nucleus adjustment (i.e. taking only the cumulative top_p probabilities of teacher)
        top_p_warper = TopPLogitsWarper(top_p=abs(temperature))
        teacher_logits = top_p_warper(input_ids=None, scores=teacher_logits)
    loss_logits = kl_with_masks(student_logits, teacher_logits, labels_mask, causal_mask=False, is_relations=False)
    return loss_logits * (temperature ** 2)


def _relations_loss_helper(student_relations: Tensor, teacher_relations: Tensor, attention_mask: Tensor,
                           causal_mask: bool = False, kl_loss: bool = False):
    if kl_loss:
        return kl_with_masks(student_relations, teacher_relations, attention_mask, causal_mask, is_relations=True)
    else:
        return rmse_with_masks(student_relations, teacher_relations, attention_mask, causal_mask, is_relations=True)


def distil_hidden_loss(student_hidden: Tensor, teacher_hidden: Tensor,
                       attention_mask: Tensor, causal_mask: bool = False, kl_loss: bool = False):
    # hidden.shape = (batch_size, seq_length, h_dim)
    # attention_mask.shape = (batch_size, seq_length)
    assert student_hidden.shape[0] == teacher_hidden.shape[0] and student_hidden.shape[0] == attention_mask.shape[0]
    assert student_hidden.shape[1] == teacher_hidden.shape[1] and student_hidden.shape[1] == attention_mask.shape[1]
    teacher_hidden = teacher_hidden.type(student_hidden.dtype)
    student_relations = _self_relations(student_hidden)
    teacher_relations = _self_relations(teacher_hidden)
    return _relations_loss_helper(student_relations, teacher_relations, attention_mask, causal_mask, kl_loss)


def distil_attention_relations_loss(student_states: Tensor, teacher_states: Tensor,
                                    attention_mask: Tensor, causal_mask: bool = False, kl_loss: bool = False):
    # states.shape = (batch_size, n_heads, seq_length, dim_per_head)
    teacher_heads = teacher_states.shape[1]
    assert student_states.shape[0] == teacher_states.shape[0]
    assert student_states.shape[2] == teacher_states.shape[2]
    teacher_states = teacher_states.type(student_states.dtype)
    student_states = student_states.permute(0, 2, 1, 3)  # (batch_size, seq_length, student_heads, student_dim_per_head)
    student_states = student_states.contiguous()
    student_states = student_states.view(student_states.shape[0], student_states.shape[1], teacher_heads, -1)
    student_states = student_states.permute(0, 2, 1, 3)  # (batch_size, teacher_heads, seq_length, adj_dim_per_head)
    student_relations = _self_relations(student_states)  # (batch_size, teacher_heads, seq_length, seq_length)
    teacher_relations = _self_relations(teacher_states)  # (batch_size, teacher_heads, seq_length, seq_length)
    # flatting the heads: (batch_size * teacher_heads, seq_length, seq_length)
    student_relations = student_relations.view(-1, student_relations.shape[-2], student_relations.shape[-1])
    teacher_relations = teacher_relations.view(-1, teacher_relations.shape[-2], teacher_relations.shape[-1])
    # duplicate attention_mask teacher_heads times
    attention_mask = torch.repeat_interleave(attention_mask, teacher_heads, dim=0)
    return _relations_loss_helper(student_relations, teacher_relations, attention_mask, causal_mask, kl_loss)


def loss_step(student: PreTrainedModel, teacher: Union[PreTrainedModel, Dict],
              input_ids: Tensor, attention_mask: Tensor, labels: Tensor, pad_token_id: Optional[int] = None,
              forward_weight: Optional[float] = None,
              logits_weight: Optional[float] = None,
              hidden_weight: Optional[float] = None,
              attention_relation_weight: Optional[float] = None,
              temperature: float = 1.0, kl_loss: bool = False, loss_scales: Optional[Dict[str, float]] = None,
              student_enc_hidden_layer: int = -1, teacher_enc_hidden_layer: int = -1,
              student_dec_hidden_layer: int = -1, teacher_dec_hidden_layer: int = -1,
              student_enc_sa_layer: int = -1, teacher_enc_sa_layer: int = -1,
              student_dec_sa_layer: int = -1, teacher_dec_sa_layer: int = -1,
              student_dec_ca_layer: int = -1, teacher_dec_ca_layer: int = -1,
              return_loss_items: bool = False,
              dropout: bool = False, **kwargs):
    if pad_token_id is not None:
        labels[labels[:, :] == pad_token_id] = LOSS_IGNORE_ID
    labels_mask = torch.zeros_like(labels, dtype=attention_mask.dtype, device=attention_mask.device)
    labels_mask[labels != LOSS_IGNORE_ID] = 1
    forward_weight = 0.0 if forward_weight is None else forward_weight
    logits_weight = 0.0 if logits_weight is None else logits_weight
    hidden_weight = 0.0 if hidden_weight is None else hidden_weight
    attention_relation_weight = 0.0 if attention_relation_weight is None else attention_relation_weight
    loss_scales = loss_scales if loss_scales is not None else {}
    kwargs = kwargs.copy()
    kwargs.pop('generate_labels_weight', None)
    if hidden_weight > 0.0:
        kwargs['output_hidden_states'] = True
    if attention_relation_weight > 0.0:
        kwargs['use_cache'] = True
    student_outputs = student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
    if isinstance(teacher, PreTrainedModel):
        with torch.no_grad():
            if dropout:
                teacher.train()
            teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
            teacher.eval()
    else:
        student_logits = student_outputs.logits
        teacher_logits = teacher.get('logits', None)
        if teacher_logits is not None:  # sometimes the
            if student_logits.shape[2] < teacher_logits.shape[2]:
                teacher_logits = teacher_logits[:, :, :student_logits.shape[2]]
            elif student_logits.shape[2] > teacher_logits.shape[2]:
                x, y, z = teacher_logits.shape
                z = student_logits.shape[2] - z
                extra_logits = torch.ones((x, y, z), device=teacher_logits.device, dtype=teacher_logits.dtype)
                extra_logits *= -10000.0
                teacher_logits = torch.cat([teacher_logits, extra_logits], dim=-1)
        teacher_outputs = Seq2SeqLMOutput(
            loss=teacher.get('loss', None),
            logits=teacher_logits,
            past_key_values=teacher.get('past_key_values', None),
            decoder_hidden_states=teacher.get('decoder_hidden_states', None),
            decoder_attentions=teacher.get('decoder_attentions', None),
            cross_attentions=teacher.get('cross_attentions', None),
            encoder_last_hidden_state=teacher.get('encoder_last_hidden_state', None),
            encoder_hidden_states=teacher.get('encoder_hidden_states', None),
            encoder_attentions=teacher.get('encoder_attentions', None),
        )
    losses = {}
    loss = None
    if forward_weight > 0.0:
        forward_loss = student_outputs.loss
        losses['forward_loss'] = forward_loss.detach().item()
        forward_loss = forward_weight * forward_loss * loss_scales.get('forward_loss', 1.0)
        loss = forward_loss if loss is None else (loss + forward_loss)
    if logits_weight > 0.0:
        student_logits = student_outputs.logits  # (batch_size, seq_length, vocab)
        teacher_logits = teacher_outputs.logits  # (batch_size, seq_length, vocab)
        logits_loss = distil_logits_loss(student_logits, teacher_logits, labels_mask, temperature)
        losses['logits_loss'] = logits_loss.detach().item()
        logits_loss = logits_weight * logits_loss * loss_scales.get('logits_loss', 1.0)
        loss = logits_loss if loss is None else (loss + logits_loss)
    if hidden_weight > 0.0:
        teacher_enc_hidden = teacher_outputs.encoder_hidden_states[teacher_enc_hidden_layer]
        teacher_dec_hidden = teacher_outputs.decoder_hidden_states[teacher_dec_hidden_layer]
        student_enc_hidden = student_outputs.encoder_hidden_states[student_enc_hidden_layer]
        student_dec_hidden = student_outputs.decoder_hidden_states[student_dec_hidden_layer]
        hidden_enc_loss = distil_hidden_loss(student_enc_hidden, teacher_enc_hidden, attention_mask,
                                             causal_mask=False, kl_loss=kl_loss)
        hidden_dec_loss = distil_hidden_loss(student_dec_hidden, teacher_dec_hidden, labels_mask,
                                             causal_mask=True, kl_loss=kl_loss)
        losses['hidden_enc_loss'] = hidden_enc_loss.detach().item()
        losses['hidden_dec_loss'] = hidden_dec_loss.detach().item()
        hidden_loss = hidden_weight * (hidden_enc_loss * loss_scales.get('hidden_enc_loss', 1.0) +
                                       hidden_dec_loss * loss_scales.get('hidden_dec_loss', 1.0)) / 2
        loss = hidden_loss if loss is None else (loss + hidden_loss)
    if attention_relation_weight > 0.0:
        student_dec_sa_k, student_dec_sa_v, student_dec_sa_q = student_outputs.past_key_values[student_dec_sa_layer][:3]
        student_dec_ca_k, student_dec_ca_v, student_dec_ca_q = student_outputs.past_key_values[student_dec_ca_layer][3:]
        student_enc_sa_k, student_enc_sa_v, student_enc_sa_q = student_outputs.encoder_past_key_values[
                                                                   student_enc_sa_layer][:3]
        teacher_dec_sa_k, teacher_dec_sa_v, teacher_dec_sa_q = teacher_outputs.past_key_values[teacher_dec_sa_layer][:3]
        teacher_dec_ca_k, teacher_dec_ca_v, teacher_dec_ca_q = teacher_outputs.past_key_values[teacher_dec_ca_layer][3:]
        teacher_enc_sa_k, teacher_enc_sa_v, teacher_enc_sa_q = teacher_outputs.encoder_past_key_values[
                                                                   teacher_enc_sa_layer][:3]
        dec_sa_k_loss = distil_attention_relations_loss(student_dec_sa_k, teacher_dec_sa_k, labels_mask,
                                                        causal_mask=True, kl_loss=kl_loss)
        losses['dec_sa_k_loss'] = dec_sa_k_loss.detach().item()
        dec_sa_k_loss = dec_sa_k_loss * loss_scales.get('dec_sa_k_loss', 1.0)
        dec_sa_v_loss = distil_attention_relations_loss(student_dec_sa_v, teacher_dec_sa_v, labels_mask,
                                                        causal_mask=True, kl_loss=kl_loss)
        losses['dec_sa_v_loss'] = dec_sa_v_loss.detach().item()
        dec_sa_v_loss = dec_sa_v_loss * loss_scales.get('dec_sa_v_loss', 1.0)
        dec_sa_q_loss = distil_attention_relations_loss(student_dec_sa_q, teacher_dec_sa_q, labels_mask,
                                                        causal_mask=True, kl_loss=kl_loss)
        losses['dec_sa_q_loss'] = dec_sa_q_loss.detach().item()
        dec_sa_q_loss = dec_sa_q_loss * loss_scales.get('dec_sa_q_loss', 1.0)
        dec_ca_k_loss = distil_attention_relations_loss(student_dec_ca_k, teacher_dec_ca_k, attention_mask,
                                                        causal_mask=False, kl_loss=kl_loss)
        losses['dec_ca_k_loss'] = dec_ca_k_loss.detach().item()
        dec_ca_k_loss = dec_ca_k_loss * loss_scales.get('dec_ca_k_loss', 1.0)
        dec_ca_v_loss = distil_attention_relations_loss(student_dec_ca_v, teacher_dec_ca_v, attention_mask,
                                                        causal_mask=False, kl_loss=kl_loss)
        losses['dec_ca_v_loss'] = dec_ca_v_loss.detach().item()
        dec_ca_v_loss = dec_ca_v_loss * loss_scales.get('dec_ca_v_loss', 1.0)
        dec_ca_q_loss = distil_attention_relations_loss(student_dec_ca_q, teacher_dec_ca_q, labels_mask,
                                                        causal_mask=False, kl_loss=kl_loss)
        losses['dec_ca_q_loss'] = dec_ca_q_loss.detach().item()
        dec_ca_q_loss = dec_ca_q_loss * loss_scales.get('dec_ca_q_loss', 1.0)
        enc_sa_k_loss = distil_attention_relations_loss(student_enc_sa_k, teacher_enc_sa_k, attention_mask,
                                                        causal_mask=False, kl_loss=kl_loss)
        losses['enc_sa_k_loss'] = enc_sa_k_loss.detach().item()
        enc_sa_k_loss = enc_sa_k_loss * loss_scales.get('enc_sa_k_loss', 1.0)
        enc_sa_v_loss = distil_attention_relations_loss(student_enc_sa_v, teacher_enc_sa_v, attention_mask,
                                                        causal_mask=False, kl_loss=kl_loss)
        losses['enc_sa_v_loss'] = enc_sa_v_loss.detach().item()
        enc_sa_v_loss = enc_sa_v_loss * loss_scales.get('enc_sa_v_loss', 1.0)
        enc_sa_q_loss = distil_attention_relations_loss(student_enc_sa_q, teacher_enc_sa_q, attention_mask,
                                                        causal_mask=False, kl_loss=kl_loss)
        losses['enc_sa_q_loss'] = enc_sa_q_loss.detach().item()
        enc_sa_q_loss = enc_sa_q_loss * loss_scales.get('enc_sa_q_loss', 1.0)
        rel_loss = attention_relation_weight * (dec_sa_k_loss + dec_sa_v_loss + dec_sa_q_loss + dec_ca_k_loss +
                                                dec_ca_v_loss + dec_ca_q_loss + enc_sa_k_loss + enc_sa_v_loss +
                                                enc_sa_q_loss) / 9
        loss = rel_loss if loss is None else (loss + rel_loss)
    if return_loss_items:
        return losses
    else:
        return loss


def distil_logits(student: PreTrainedModel, teacher: PreTrainedModel,
                  input_ids: Tensor, attention_mask: Tensor, labels: Tensor, pad_token_id: Optional[int] = None,
                  temperature: float = 1.0, **kwargs):
    return loss_step(student, teacher, input_ids, attention_mask, labels, pad_token_id,
                     forward_weight=0.0, logits_weight=1.0, hidden_weight=0.0, attention_relation_weight=0.0,
                     temperature=temperature, kl_loss=False, **kwargs)


def distil_hidden(student: PreTrainedModel, teacher: PreTrainedModel,
                  input_ids: Tensor, attention_mask: Tensor, labels: Tensor, pad_token_id: Optional[int] = None,
                  **kwargs):
    return loss_step(student, teacher, input_ids, attention_mask, labels, pad_token_id,
                     forward_weight=0.0, logits_weight=0.0, hidden_weight=1.0, attention_relation_weight=0.0,
                     temperature=1.0, kl_loss=False, **kwargs)


def distil_attention_relations(student: PreTrainedModel, teacher: PreTrainedModel,
                               input_ids: Tensor, attention_mask: Tensor, labels: Tensor,
                               pad_token_id: Optional[int] = None,
                               kl_loss: bool = False, **kwargs):
    return loss_step(student, teacher, input_ids, attention_mask, labels, pad_token_id,
                     forward_weight=0.0, logits_weight=0.0, hidden_weight=0.0, attention_relation_weight=1.0,
                     temperature=1.0, kl_loss=kl_loss, **kwargs)


class LossScheduler:
    def __init__(self,
                 list_of_loss_kwargs: List[Dict[str, Any]],
                 change_every_n_epochs: Optional[int] = None,
                 change_every_n_evaluation_steps: Optional[int] = None,
                 change_no_improvements_for: Optional[int] = None,
                 keep_last: bool = True,
                 random_sample: bool = False):
        self.list_of_loss_kwargs = list_of_loss_kwargs
        self.change_every_n_epochs = change_every_n_epochs
        self.change_every_n_evaluation_steps = change_every_n_evaluation_steps
        self.change_no_improvements_for = change_no_improvements_for
        self.keep_last = keep_last
        self.random_sample = random_sample
        self.rnp = np.random.RandomState(seed=0)
        self.current_loss_idx = 0
        self.change_epoch = -1
        self.scores = []
        self.loss_scales = {}

    @property
    def is_last_idx(self):
        return self.current_loss_idx == len(self.list_of_loss_kwargs) - 1

    def update_loss_scales(self, accelerator: Accelerator, student: PreTrainedModel,
                           teacher: Optional[PreTrainedModel],
                           dataloader: DataLoader, pad_token_id: Optional[int] = None,
                           sampling_steps: int = SAMPLIMG_STEPS):
        loss_scale = self.list_of_loss_kwargs[self.current_loss_idx].get('loss_scale', None)
        if teacher is None or loss_scale is None or not loss_scale:  # when teacher is None we do not need to scale
            self.loss_scales = None
            return
        assert (accelerator is not None and student is not None and teacher is not None
                and dataloader is not None)
        student_training = student.training
        teacher_training = teacher.training
        student.eval()
        teacher.eval()
        losses = []
        progress_bar = tqdm(dataloader, desc=f'Finding Scales')
        for step, batch in enumerate(progress_bar):
            # we need to skip steps until we reach the resumed step
            if step >= sampling_steps:
                break
            batch = batch.to(accelerator.device)
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            kd_loss_step_kwargs = self.get_loss_kwargs(return_loss_items=True)
            losses.append(loss_step(student, teacher, input_ids, attention_mask, labels, pad_token_id,
                                    **kd_loss_step_kwargs))
        # calculate the mean loss for each type of loss
        loss_scales = {}
        for lossed_dict in losses:
            for key, value in lossed_dict.items():
                if key not in loss_scales:
                    loss_scales[key] = []
                if value is not None and not np.isnan(value):
                    loss_scales[key].append(value)
        # scale the losss to the same scale (1.0)
        self.loss_scales = {key: 1 / max(np.mean(value), 0.025) for key, value in loss_scales.items()}
        # set the model back to the original state
        if student_training:
            student.train()
        if teacher_training:
            teacher.train()

    def check_if_change(self, epoch: Optional[int] = None, scores: Optional[List[float]] = None) -> bool:
        if isinstance(self.change_every_n_epochs, int) and epoch is not None:
            if epoch - self.change_epoch >= self.change_every_n_epochs:
                return True
        scores = scores if scores is not None else self.scores
        if isinstance(self.change_every_n_evaluation_steps, int) and scores is not None:
            if len(scores) > 0 and len(scores) >= self.change_every_n_evaluation_steps:
                return True
        if isinstance(self.change_no_improvements_for, int) and scores is not None:
            if len(scores) > self.change_no_improvements_for:
                if max(scores[-self.change_no_improvements_for:]) < max(scores[:self.change_no_improvements_for]):
                    return True
        return False

    def change_stage(self, epoch: Optional[int] = None, new_score: Optional[float] = None,
                     accelerator: Optional[Accelerator] = None,
                     student: Optional[PreTrainedModel] = None, teacher: Optional[PreTrainedModel] = None,
                     dataloader: Optional[DataLoader] = None, pad_token_id: Optional[int] = None,
                     sampling_steps: int = SAMPLIMG_STEPS):
        if new_score is not None:
            self.scores.append(new_score)
        if self.check_if_change(epoch):
            self.change_epoch = epoch
            self.current_loss_idx += 1
            if self.current_loss_idx >= len(self.list_of_loss_kwargs):
                self.current_loss_idx = (len(self.list_of_loss_kwargs) - 1) if self.keep_last else 0
            if self.random_sample:
                self.current_loss_idx = self.rnp.randint(0, len(self.list_of_loss_kwargs))
            self.scores = []
            self.update_loss_scales(accelerator, student, teacher, dataloader, pad_token_id, sampling_steps)
            print(f"Changing loss stage to {self.current_loss_idx}. New loss kwargs:")
            for k, v in self.list_of_loss_kwargs[self.current_loss_idx].items():
                print(f"\t{k}: {v}")

    def get_loss_kwargs(self, return_loss_items: bool = False, ft_stage: bool = False) -> Dict[str, Any]:
        kd_loss_step_kwargs = self.list_of_loss_kwargs[self.current_loss_idx].copy()
        kd_loss_step_kwargs['return_loss_items'] = return_loss_items
        kd_loss_step_kwargs['loss_scales'] = self.loss_scales
        if ft_stage:
            kd_loss_step_kwargs['kd_loss_type'] = 'forward'
            kd_loss_step_kwargs['generate_labels_weight'] = 0.0
        return self.prepare_loss_step_kwargs(**kd_loss_step_kwargs)

    @staticmethod
    def prepare_loss_step_kwargs(kd_loss_type: Optional[str] = None,
                                 **kd_loss_step_kwargs) -> Dict:
        kd_loss_step_kwargs = kd_loss_step_kwargs.copy()
        if kd_loss_type is None or kd_loss_type == 'multi':
            forward_weight = kd_loss_step_kwargs.pop('forward_weight', 0.0)
            logits_weight = kd_loss_step_kwargs.pop('logits_weight', 0.0)
            hidden_weight = kd_loss_step_kwargs.pop('hidden_weight', 0.0)
            attention_relation_weight = kd_loss_step_kwargs.pop('attention_relation_weight', 0.0)
        else:
            assert kd_loss_type in ['logits', 'hidden', 'attention_relation', 'forward']
            forward_weight = 1.0 if kd_loss_type == 'forward' else 0.0
            logits_weight = 1.0 if kd_loss_type == 'logits' else 0.0
            hidden_weight = 1.0 if kd_loss_type == 'hidden' else 0.0
            attention_relation_weight = 1.0 if kd_loss_type == 'attention_relation' else 0.0
        forward_weight = 0.0 if forward_weight is None else forward_weight
        logits_weight = 0.0 if logits_weight is None else logits_weight
        hidden_weight = 0.0 if hidden_weight is None else hidden_weight
        attention_relation_weight = 0.0 if attention_relation_weight is None else attention_relation_weight
        return dict(forward_weight=forward_weight, logits_weight=logits_weight,
                    hidden_weight=hidden_weight, attention_relation_weight=attention_relation_weight,
                    temperature=kd_loss_step_kwargs.pop('temperature', 1.0),
                    kl_loss=kd_loss_step_kwargs.pop('kl_loss', False),
                    loss_scales=kd_loss_step_kwargs.pop('loss_scales', None),
                    student_enc_hidden_layer=kd_loss_step_kwargs.get('student_enc_hidden_layer', -1),
                    teacher_enc_hidden_layer=kd_loss_step_kwargs.get('teacher_enc_hidden_layer', -1),
                    student_dec_hidden_layer=kd_loss_step_kwargs.get('student_dec_hidden_layer', -1),
                    teacher_dec_hidden_layer=kd_loss_step_kwargs.get('teacher_dec_hidden_layer', -1),
                    student_enc_sa_layer=kd_loss_step_kwargs.get('student_enc_sa_layer', -1),
                    teacher_enc_sa_layer=kd_loss_step_kwargs.get('teacher_enc_sa_layer', -1),
                    student_dec_sa_layer=kd_loss_step_kwargs.get('student_dec_sa_layer', -1),
                    teacher_dec_sa_layer=kd_loss_step_kwargs.get('teacher_dec_sa_layer', -1),
                    student_dec_ca_layer=kd_loss_step_kwargs.get('student_dec_ca_layer', -1),
                    teacher_dec_ca_layer=kd_loss_step_kwargs.get('teacher_dec_ca_layer', -1),
                    return_loss_items=kd_loss_step_kwargs.get('return_loss_items', False),
                    dropout=kd_loss_step_kwargs.get('dropout', False),
                    generate_labels_weight=kd_loss_step_kwargs.get('generate_labels_weight', 0.0))
