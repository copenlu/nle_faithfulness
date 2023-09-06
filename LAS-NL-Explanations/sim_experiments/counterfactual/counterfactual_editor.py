import random

import ComVE_data_utils
import NLI_data_utils
import QA_data_utils
import argparse
import os
import json
import torch
import numpy as np
from functools import partial
from models.T5ForMC import T5ModelForMC
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, T5Model, T5ForConditionalGeneration
from utils import str2bool
import utils
from collections import defaultdict
import copy
from string import punctuation
from dataset import collate_snli, collate_comve, collate_qa, collate_snli_test, collate_explanation_student, add_insertions, add_insertions_qa, add_insertions_comve

_SPECIAL_TOKEN_INFILL = '<extra_id_0>'

def enforce_reproducibility(seed: int = 42):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def eval_examples(args, device, examples, model, tokenizer, sample_exps):
    """ Runs one epoch. returns stats_dict. updates model parameters if training. """
    data_tensors = prep_function(args, examples=examples,
                                tokenizer=tokenizer,
                                max_seq_length=args.max_seq_length,
                                condition_on_explanations=args.condition_on_explanations,
                                multi_explanation=args.multi_explanation,
                                explanations_only=args.explanations_only)
    dataloader = DataLoader(TensorDataset(*data_tensors), shuffle=False,
                                batch_size=args.train_batch_size if args.do_train else args.dev_batch_size,
                                num_workers=4, pin_memory=True)

    model.eval()
    ST_RA = (args.condition_on_explanations and args.multi_explanation)

    # ignore these in decoding
    ignore_tokens_list = [tokenizer.pad_token, '[UNK]']

    # init stat vars
    label_strs, sample_strs, multi_sample_strs = [], [], []
    preds_list = []
    for step, batch in enumerate(dataloader):
        # unpack batch vars
        batch = [item.to(device) for item in batch]
        task_input_ids, task_input_masks, \
        task_answer_ids, task_answer_masks, task_answer_labels, \
        task_output_ids, task_output_masks, task_output_labels, task_choice_labels, \
        explanation_input_ids, explanation_input_masks, \
        explanation_output_ids, explanation_output_masks, explanation_output_labels, \
        explanation_context_ids, explanation_only_ids, explanation_lens = batch

        # shape vars
        batch_size = task_output_ids.size(0)
        # num_choices = 3
        num_choices = task_output_labels.shape[1]

        # FORWARD
        grad_req = torch.no_grad()
        with grad_req:
            if args.do_task:
                if 't5' in args.task_pretrained_name and not ST_RA:
                    # print(task_input_ids.shape, task_input_masks.shape)
                    outputs = model(input_ids=task_input_ids, attention_mask=task_input_masks, return_dict=False)
                    encoder_hidden_states = outputs[-1]
                    # now get likelihoods for each choice

                    with torch.no_grad():
                        # add num_choices dim to input_masks and encoder_hidden_states and expand to match task_output_ids shape
                        expand_shape = list(encoder_hidden_states.shape)
                        expand_shape.insert(1, num_choices)
                        encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(expand_shape)
                        task_input_masks = task_input_masks.unsqueeze(1).expand_as(task_output_masks)
                        outputs = model(encoder_hidden_states=encoder_hidden_states,
                                        encoder_attention_mask=task_input_masks,
                                        decoder_input_ids=task_output_ids,
                                        decoder_lm_labels=task_output_labels,
                                        decoder_attention_mask=task_output_masks,
                                        return_dict=False)

                        # choice_losses is of shape: batch_size x num_choices, because task_output_ids had a num_choices dim
                        choice_losses = outputs[0]
                elif 't5' in args.task_pretrained_name and ST_RA:
                    batch_shape = list(task_input_ids.shape)
                    task_input_ids = task_input_ids.view(-1, task_input_ids.size(-1))
                    task_input_masks = task_input_masks.view(-1, task_input_ids.size(-1))
                    outputs = model(input_ids=task_input_ids,
                                    attention_mask=task_input_masks,
                                    return_dict=False)
                    encoder_hidden_states = outputs[-1]
                    # reshape inputs
                    task_input_masks = task_input_masks.view(batch_shape)
                    batch_shape.append(encoder_hidden_states.size(-1))
                    encoder_hidden_states = encoder_hidden_states.view(batch_shape)
                    outputs = model(encoder_hidden_states=encoder_hidden_states,
                                    encoder_attention_mask=task_input_masks,
                                    decoder_input_ids=task_output_ids,
                                    decoder_lm_labels=task_output_labels,
                                    decoder_attention_mask=task_output_masks,
                                    return_dict=False)
                    choice_losses = outputs[0]  # choice_losses is of shape: batch_size x num_choices

                # compute task accuracy
                preds = np.argmin(choice_losses.detach().cpu().numpy(), axis=-1)
                preds_list.extend(preds.tolist())

            if args.do_explain:
                model = model.to(device)
                outputs = model(input_ids=explanation_input_ids,
                                encoder_attention_mask=explanation_input_masks,
                                decoder_input_ids=explanation_output_ids,
                                decoder_lm_labels=explanation_output_labels,
                                decoder_attention_mask=explanation_output_masks,
                                return_dict=False)
                encoder_hidden_states = outputs[-1]
                # print('encoder_hidden_states', encoder_hidden_states)
            # explanation sampling. sample when do_explain is true
            if args.do_explain and sample_exps:
                if args.do_task:  # get predicted contexts
                    use_contexts = torch.stack(
                        [explanation_context_ids[i, preds[i], :] for i in range(batch_size)], dim=0
                    ).unsqueeze(1)
                elif args.multi_explanation:  # use all three contexts
                    use_contexts = explanation_context_ids
                elif not args.multi_explanation:  # take an arbitrary context for each data point (all the same)
                    use_contexts = explanation_context_ids[:, 0, :]

                # sample
                reshape = False
                if use_contexts.dim() == 3:
                    first_two_dims = list(use_contexts.shape)[:2]
                    explanation_input_masks = explanation_input_masks.unsqueeze(1).expand_as(use_contexts)
                    expand_shape = list(encoder_hidden_states.shape)
                    expand_shape.insert(1, use_contexts.size(1))
                    encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(expand_shape)
                    use_contexts = use_contexts.view(-1, use_contexts.size(-1))
                    encoder_hidden_states = encoder_hidden_states.reshape(-1, encoder_hidden_states.size(-2),
                                                                          encoder_hidden_states.size(-1))
                    explanation_input_masks = explanation_input_masks.reshape(-1, explanation_input_masks.size(-1))
                    reshape = True
                samples = utils.T5_sample(model,
                                          encoder_hidden_states=encoder_hidden_states,
                                          decoder_input_ids=use_contexts,
                                          encoder_attention_mask=explanation_input_masks,
                                          tokenizer=tokenizer,
                                          max_sample_len=args.max_sample_len)
                if reshape:
                    samples = samples.view(first_two_dims + [samples.size(-1)])

                if not args.do_task and args.multi_explanation:  # condition where three are sampled per item
                    pred_explanations = [question[task_choice_labels[i].item()] for i, question in
                                         enumerate(samples.tolist())]
                    batch_multi_sample_strs = utils.detok_batch(tokenizer, samples,
                                                                ignore_tokens=ignore_tokens_list,
                                                                eos_token=tokenizer.eos_token)
                    multi_sample_strs.extend(batch_multi_sample_strs)
                else:
                    pred_explanations = samples.squeeze(1).tolist()

                # detokenize expl. labels and predictions
                batch_label_strs = utils.detok_batch(tokenizer, explanation_only_ids,
                                                     ignore_tokens=ignore_tokens_list)
                batch_sample_strs = utils.detok_batch(tokenizer, pred_explanations,
                                                      ignore_tokens=ignore_tokens_list,
                                                      eos_token=tokenizer.eos_token)
                label_strs.extend(batch_label_strs)
                sample_strs.extend(batch_sample_strs)

    expl = None
    if args.do_explain:
        if args.multi_explanation and args.do_task:
            expl = sample_strs
        if args.multi_explanation and not args.do_task:
            explanations = np.array(multi_sample_strs)
            exp_cols = [f't5-multi-exp-{i}-seed{args.seed}' for i in range(num_choices)]
            expl = [[] for _ in range(len(examples))]
            for j, col_name in enumerate(exp_cols):
                expls = explanations[:, j].tolist()
                for i in range(len(expls)):
                    expl[i].append(expls[i])
        if not args.multi_explanation:
            expl = sample_strs

    return preds_list, expl


def load_data(version, args):
    if '1.0' in args.data_dir or 'qa' in args.model_name:
        data_name = 'QA'
    elif 'NLI' in args.data_dir or 'nli' in args.model_name:
        data_name = 'NLI'
        args.max_seq_length = 128  # override to have lower max_seq_len
        args.max_sample_len = 128  # doesn't need to be higher than max_seq_length, naturally
        print("Overriding sequence length to %d and sample_len to %d" % (args.max_seq_length, args.max_sample_len))
    elif 'comve' in args.data_dir or 'comve' in args.model_name:
        data_name = 'COMVE'
        args.max_seq_length = 128  # override to have lower max_seq_len
        args.max_sample_len = 128  # doesn't need to be higher than max_seq_length, naturally
        print("Overriding sequence length to %d and sample_len to %d" % (args.max_seq_length, args.max_sample_len))
    if data_name == 'QA':
        read_function = QA_data_utils.read_CQA
        if 't5' in args.task_pretrained_name or 'bart' in args.task_pretrained_name:
            prep_function = QA_data_utils.get_tensors_for_T5_split
        elif 'bert' in args.task_pretrained_name:
            prep_function = QA_data_utils.get_tensors_for_bert
        extension = 'csv'
    elif data_name == 'NLI':
        read_function = NLI_data_utils.read_NLI
        if 't5' in args.task_pretrained_name or 'bart' in args.task_pretrained_name:
            prep_function = NLI_data_utils.get_tensors_for_T5_split
        elif 'bert' in args.task_pretrained_name:
            prep_function = NLI_data_utils.get_tensors_for_bert
        extension = 'tsv'
    elif data_name == 'COMVE':
        read_function = ComVE_data_utils.read_ComVE
        if 't5' in args.task_pretrained_name or 'bart' in args.task_pretrained_name:
            prep_function = ComVE_data_utils.get_tensors_for_T5_split
        elif 'bert' in args.task_pretrained_name:
            prep_function = ComVE_data_utils.get_tensors_for_bert
        extension = 'csv'
    train_examples = read_function(args,
                                   input_file=os.path.join(args.data_dir, 'train.%s' % extension),
                                   explanations_to_use=args.explanations_to_use,
                                   labels_to_use=args.labels_to_use,
                                   version=version)
    dev_examples = read_function(args,
                                 input_file=os.path.join(args.data_dir, 'dev.%s' % extension),
                                 explanations_to_use=args.explanations_to_use,
                                 labels_to_use=args.labels_to_use,
                                 version=version)
    if args.test_file:
        test_file_path = os.path.join(args.save_dir, args.test_file)
    else:
        test_file_path = os.path.join(args.data_dir, 'test.%s' % extension)
    test_examples = read_function(args,
                                  input_file=test_file_path,
                                  explanations_to_use=args.explanations_to_use,
                                  labels_to_use=None if (
                                          data_name == 'QA' and args.labels_to_use == 'label') else args.labels_to_use,
                                  version=version)

    return train_examples, dev_examples, test_examples, data_name, prep_function


def load_model(args, device, tokenizer, finetuned_path=None):
    if finetuned_path is None:
        print(f"\nLoading non-finetuned model: {args.task_pretrained_name}...")
    elif finetuned_path is not None:
        print(f"\nLoading fine-tuned model: {finetuned_path}...")

    if 't5' in args.task_pretrained_name:
        model_class = T5ModelForMC
        model = model_class.from_pretrained(args.task_pretrained_name,
                                            project_to_small=False,
                                            cache_dir=args.cache_dir)
        if ('NLI' in args.data_dir or 'nli' in args.model_name) \
                and not 'CLM.rationalize' in finetuned_path \
                and not 'ST.RA' in finetuned_path:
            model.resize_token_embeddings(len(tokenizer))
        if 'v1.0' in args.data_dir and 'MT.RE' in args.model_name:
            model.resize_token_embeddings(len(tokenizer))

    if finetuned_path is not None:
        # args for preventing memory leakage across gpus
        model_state_dict = torch.load(finetuned_path, map_location=lambda storage,
                                                                          loc: storage)
        model.load_state_dict(model_state_dict)
        del model_state_dict

    model.to(device)
    return model


def get_bad_words():
    print('Getting bad words for generation...')
    bad_words = []
    bad_words_custom = ['reteta', 'gradini', 'proaspat', 'adica', 'senzati', 'si', 'unk', 'linguri', 'gardinen',
                        'limbgardinen',
                        'abo', 'ganduri', 'prevazut', 'zy', 'lingurila', 'ganduri', 'ble']
    for token in tokenizer.get_vocab():
        token_clean = token.strip('▁')
        if token_clean.lower() != token_clean or \
                len(token_clean) == 1 or \
                token_clean in punctuation or \
                token_clean in bad_words_custom or \
                any(p in token_clean for p in punctuation) or \
                not token.startswith('▁'):
            bad_words.append(token_clean)
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids
    bad_words_ids = [l for l in bad_words_ids if l]
    print('Bad words: ', len(bad_words_ids))
    return bad_words_ids


def eval_dataset(examples, editor_model, tokenizer, args, collate_fn_test, bad_words_ids, write=False):
    with torch.no_grad():
        in_filled_examples = []
        total_success, total_label_success = defaultdict(lambda: 0), defaultdict(lambda: 0)
        instances = []
        for instance in examples:
            indexes = [i for i in range(len(instance.masked_instances))]
            selected_index = random.choice(indexes)
            instances.append(instance.masked_instances[selected_index])
            del instance.masked_instances[selected_index]

        dl = DataLoader(instances, batch_size=args.dev_batch_size,
                            collate_fn=collate_fn_test, shuffle=False)

        for batch in tqdm(dl, desc=f'evaluating {len(instances)} instances'):
            outputs = editor_model.generate(batch[0]['input_ids'],
                                            max_new_tokens=args.max_words_gen,
                                            min_length=1,
                                            num_beams=args.num_beams,
                                            # top_k=30,
                                            # top_p=0.92,
                                            # length_penalty=0.5,
                                            # do_sample=True,
                                            bad_words_ids=bad_words_ids,
                                            num_return_sequences=args.num_return_sequences)

            replacements = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            k = 0

            for b_i in range(len(batch[1])):
                for s_i in range(args.num_return_sequences):
                    example = copy.deepcopy(batch[1][b_i])
                    if example.masked == 'hypothesis':
                        example.hypothesis = example.hypothesis.replace(_SPECIAL_TOKEN_INFILL, f'{replacements[k]}')
                    elif example.masked == 'premise':
                        example.premise = example.premise.replace(_SPECIAL_TOKEN_INFILL, f'{replacements[k]}')
                    elif example.masked == 'question':
                        example.question = example.question.replace(_SPECIAL_TOKEN_INFILL, f'{replacements[k]}')
                    elif example.masked == 'sent1':
                        example.sent1 = example.sent1.replace(_SPECIAL_TOKEN_INFILL, f'{replacements[k]}')
                    elif example.masked == 'sent2':
                        example.sent2 = example.sent2.replace(_SPECIAL_TOKEN_INFILL, f'{replacements[k]}')
                    example.replacement = replacements[k]
                    in_filled_examples.append(example)
                    k += 1

        teacher_explanations, teacher_predictions = model_specific_predictions_explanations(args, in_filled_examples,
                                                                                            tokenizer)
        with open(f'{data_name}_{prefinetuned_name}_counterfactual.jsonl', 'a') as outfile:
            for i in range(len(in_filled_examples)):
                if int(in_filled_examples[i].original_label) != int(teacher_predictions[i]):

                    total_label_success[in_filled_examples[i].idx] += 1
                    inserted_tokens = [t.lower().replace('▁', '') for t in
                                       tokenizer.tokenize(in_filled_examples[i].replacement)]
                    inserted_tokens = [t for t in inserted_tokens if t.strip()]
                    outfile.write('counterfactual\t'+json.dumps(vars(in_filled_examples[i]), cls=NpEncoder) + '\n')

                    if inserted_tokens and all(
                            inserted_token not in teacher_explanations[i].lower() for inserted_token in inserted_tokens):

                        in_filled_examples[i].teacher_predictions = teacher_predictions[i]
                        in_filled_examples[i].teacher_explanations = teacher_explanations[i]
                        # print('-----\n', vars(in_filled_examples[i]))
                        outfile.write('advcounterfactual\t'+json.dumps(vars(in_filled_examples[i]), cls=NpEncoder)+'\n')
                        total_success[in_filled_examples[i].idx] += 1
 
        print("Total counterfactual success:", len(total_success), len(examples),
              len(total_success) / len(examples), (len(total_success) / len(total_label_success)) if len(total_label_success) else 0)
        print("Total label success:", len(total_label_success), len(examples),
              len(total_label_success) / len(examples))

        del batch
        del outputs
        success = len(total_success) / len(total_label_success) if len(total_label_success) else 0
        return success, len(total_label_success) / len(examples), total_success, total_label_success

 
def model_specific_predictions_explanations(args, in_filled_examples, tokenizer):
    # print('Getting teacher predictions...', len(in_filled_examples))
    if args.model_name.startswith('MT'):
        if args.model_name == 'MT.RE':
            args.multi_explanation = False
        else:
            args.multi_explanation = True
            args.task_coef = 0.5
        teacher_predictions, teacher_explanations = eval_examples(examples=in_filled_examples,
                                                                  args=args,
                                                                  device=device,
                                                                  model=explanation_model,
                                                                  tokenizer=tokenizer,
                                                                  sample_exps=True
                                                                  )
    elif args.model_name == 'ST.RE':
        # generate explanation
        args.do_task = False
        args.do_explain = True
        args.task_coef = 0
        args.multi_explanation = False
        args.model_name = 'CLM.reason'

        _, teacher_explanations = eval_examples(examples=in_filled_examples,
                                                args=args,
                                                device=device,
                                                model=explanation_model,
                                                tokenizer=tokenizer,
                                                sample_exps=True)

        for i in range(len(in_filled_examples)):
            in_filled_examples[i].explanation = teacher_explanations[i]

        # generate predictions
        args.do_explain = False
        args.do_task = True
        args.task_coef = 1
        args.multi_explanation = False
        args.condition_on_explanations = True
        args.model_name = 'ST.RE'

        teacher_predictions, _ = eval_examples(examples=in_filled_examples,
                                               args=args,
                                               device=device,
                                               model=prediction_model,
                                               tokenizer=tokenizer,
                                               sample_exps=False
                                               )
    elif args.model_name == 'ST.RA':
        args.do_task = False
        args.do_explanation = True
        args.task_coef = 0
        args.multi_explanation = True
        args.model_name = 'CLM.rationalize'
        args.condition_on_explanations = False

        _, teacher_explanations = eval_examples(examples=in_filled_examples,
                                                args=args,
                                                device=device,
                                                model=explanation_model,
                                                tokenizer=tokenizer,
                                                sample_exps=True)

        for i in range(len(in_filled_examples)):
            in_filled_examples[i].explanation = teacher_explanations[i]

        args.do_task = True
        args.do_explanation = False
        args.task_coef = 1
        args.multi_explanation = True
        args.model_name = 'ST.RA'
        args.condition_on_explanations = True

        teacher_predictions, _ = eval_examples(examples=in_filled_examples,
                                               args=args,
                                               device=device,
                                               model=prediction_model,
                                               tokenizer=tokenizer,
                                               sample_exps=False)
        # pick explanation of predicted label
        teacher_explanations = [tp[i] for tp, i in zip(teacher_explanations, teacher_predictions)]
    return teacher_explanations, teacher_predictions


def eval_loop(examples, editor_model, tokenizer, args, collate_fn_test, bad_words_ids, write=False):
    dev_instances_not_ready = copy.deepcopy(examples)
    k = 0
    label_set, explanation_set = set(), set()
    while dev_instances_not_ready and k < args.n_pos:

        adversarial_rate, counterfactual_rate, success_dict, label_success_dict = eval_dataset(dev_instances_not_ready,
                                                                                               editor_model,
                                                                                               tokenizer,
                                                                                               args,
                                                                                               collate_fn_test,
                                                                                               bad_words_ids,
                                                                                               write)

        label_set.update(set(label_success_dict.keys()))
        explanation_set.update(set(success_dict.keys()))
        k += 1

        keep_instances = []
        for instance in dev_instances_not_ready:
            if instance.idx not in success_dict and instance.masked_instances:
                keep_instances.append(instance)
        dev_instances_not_ready = copy.deepcopy(keep_instances)
    total_adversarial_rate = len(explanation_set) / len(label_set) if label_set else 0

    return total_adversarial_rate, label_set, explanation_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--editor_model_path", default='editor_checkpoint', type=str, help='HuggingFace transformer model')

    parser.add_argument("--task_pretrained_name", default='t5-base', type=str, help='HuggingFace transformer model')
    parser.add_argument("--max_seq_length", default=175, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    # hyperparams
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--dev_batch_size", default=20, type=int, help="Total batch size for eval.")
    parser.add_argument("--lr", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--max_sample_len', type=int, default=175,
                        help='Maximum num tokens that can appear in generated explanation')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id to use. -1 defaults to multi-gpu')
    # misc
    parser.add_argument('--seed', type=int, default=21, help="random seed for initialization")
    parser.add_argument('--debug', action='store_true', help='Flag that queues ipdb before training')
    # directories + file paths
    parser.add_argument("--save_dir", default='', required=True, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--data_dir', type=str, default='data/e-SNLI-data/',
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--report_dir", default='training_reports/', type=str,
                        help="The output directory where the model training reports will be written.")
    parser.add_argument("--cache_dir", default='', required=True, type=str,
                        help="Directory for cacheing pretrained models.")
    parser.add_argument('--model_name', type=str, default='unnamed',
                        help="Save and/or load name for model. See below in script for exact formatting")
    parser.add_argument('--prefinetuned_name', type=str, default='',
                        help="Load name for model to start training with.")
    # debug flags
    parser.add_argument('--small_data', '-s', action='store_true',
                        help='Flag for using just a few datapoints for debugging purposes')
    parser.add_argument("--small_size", '-ss', default=100, type=int, help="")
    # experiment condition flags
    parser.add_argument("--condition_on_explanations", default=False, type=str2bool,
                        help="Whether or not to condition on explanations in input")
    parser.add_argument("--explanations_to_use", default='ground_truth', help="Which explanations to load with data.")
    parser.add_argument("--explanations_only", default=False, type=str2bool,
                        help="Include only answer choices and explanations (no x) as input")
    parser.add_argument("--labels_to_use", default='label',
                        help="Which labels to use with data. Intended for the use of simulating other models")
    parser.add_argument("--do_task", default=True, type=str2bool, help="Do QA")
    parser.add_argument("--do_explain", default=True, type=str2bool, help="Do LM")
    parser.add_argument("--multi_explanation", default=True, type=str2bool,
                        help="Generate an explanation for each answer choice")

    # control flow for script
    parser.add_argument("--test_file", default=None, type=str, help="If testing on a different test file.")

    parser.add_argument("--test_split", default='test', type=str, help="Which split to use for testing.")
    parser.add_argument("--do_train", default=True, type=str2bool, help="Whether to run training.")
    parser.add_argument("--non_explanation", default=False, type=str2bool, help="Mask only tokens not in explanation.")
    parser.add_argument("--save_agent", default=False, type=str2bool, help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, type=str2bool,
                        help="Whether to run final eval on dev and test sets.")
    parser.add_argument("--eval_on_train", default=False, action='store_true',
                        help="Whether to run eval on the train data.")
    parser.add_argument("--n_pos", default=1, type=int,
                        help='Number of positions to try inserting words to for a single instance.')
    parser.add_argument("--num_return_sequences", default=1, type=int,
                        help='Number of returned sampled sequences per instance.')
    parser.add_argument("--num_beams", default=2, type=int,
                        help='Number of beams to use for generation.')
    parser.add_argument("--max_words_gen", default=1, type=int,
                        help='Number of beams to use for generation.')
    parser.add_argument("--filling_loss_weight", default=0.5, type=float,
                        help='Weight for the in-filling loss.')
    parser.add_argument("--imitation_loss_weight", default=0.5, type=float,
                        help='Weight for the imitation loss')
    parser.add_argument("--adversary_loss_weight", default=0.5, type=float,
                        help='Weight for the imitation loss')
    parser.add_argument("--max_mask_tokens", default=3, type=int,
                        help='How many tokens to mask at most during training.')
    parser.add_argument("--weights_mask_n", default=[1, 1, 1], type=int, nargs='+',
                        help='Weight for choosing how many tokens to mask at most.')

    args = parser.parse_args()
    enforce_reproducibility(args.seed)
    args.hypothesis_only, args.random_label, args.stain, args.rand_threshold = False, False, False, False
    version = '1.0'  # if '1.0' in args.data_dir else '1.1'

    # data reading
    train_examples, dev_examples, test_examples, data_name, prep_function = load_data(version, args)
    best_adversarial_rate = 0
    best_model_weights = None
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    device = torch.device("cuda")
    collate_fn_test = partial(collate_snli_test,
                         tokenizer=tokenizer,
                         device=device,
                         n_pos=args.n_pos)
    train_collate_function = None
    if data_name == 'NLI':
        train_collate_function = collate_snli
    elif data_name == 'QA':
        train_collate_function = collate_qa
    elif data_name == 'COMVE':
        train_collate_function = collate_comve

    collate_fn_train = partial(train_collate_function,
                               tokenizer=tokenizer,
                               device=device,
                               max_mask_tokens=args.max_mask_tokens,
                               weights_mask_n=args.weights_mask_n,
                               non_explanation=args.non_explanation)
    collate_fn_student = partial(collate_explanation_student,
                                 tokenizer=tokenizer,
                                 device=device,
                                 data_name=data_name)

    if data_name == 'NLI':
        dev_examples = add_insertions(dev_examples, tokenizer)
        test_examples = add_insertions(test_examples, tokenizer)
    elif data_name == 'QA':
        dev_examples = add_insertions_qa(dev_examples, tokenizer)
        test_examples = add_insertions_qa(test_examples, tokenizer)
    elif data_name == 'COMVE':
        dev_examples = add_insertions_comve(dev_examples, tokenizer)
        test_examples = add_insertions_comve(test_examples, tokenizer)
    dev_dl = DataLoader(dev_examples, batch_size=args.dev_batch_size,
                             collate_fn=collate_fn_test, shuffle=False)
    train_dl = DataLoader(train_examples, batch_size=args.train_batch_size,
                            collate_fn=collate_fn_train, shuffle=True)

    # Load explanation and prediction models
    agent_insert = '2_agent_task_' if args.save_agent else ''
    agent_epoch = f'_epoch_{args.load_epoch}' if args.save_agent else ''
    save_name = f"{data_name}_{agent_insert}{args.task_pretrained_name}_{args.model_name}_seed{args.seed}{agent_epoch}"

    if args.small_data:
        save_name += '_DEBUG'

    model_path = os.path.join(args.save_dir, save_name + ".hdf5")
    prefinetuned_name = f"{data_name}_{args.task_pretrained_name}_{args.prefinetuned_name}"
    prefinetuned_path = os.path.join(args.save_dir, prefinetuned_name + ".hdf5") if args.prefinetuned_name != '' else None
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    if not os.path.exists(args.report_dir): os.makedirs(args.report_dir)

    if args.model_name.startswith('MT'):
        explanation_model = load_model(args, device, tokenizer, finetuned_path=prefinetuned_path)
    if args.model_name == 'ST.RA':
        # explanation model
        args.model_name = 'CLM.rationalize'
        args.prefinetuned_name = 'CLM.rationalize_seed21'
        prefinetuned_name = f"{data_name}_{args.task_pretrained_name}_{args.prefinetuned_name}"
        prefinetuned_path = os.path.join(args.save_dir,
                                         prefinetuned_name + ".hdf5") if args.prefinetuned_name != '' else None
        explanation_model = load_model(args, device, tokenizer, finetuned_path=prefinetuned_path)

        #### prediction model
        args.prefinetuned_name = 'ST.RA_seed21'
        args.model_name = 'ST.RA'
        prefinetuned_name = f"{data_name}_{args.task_pretrained_name}_{args.prefinetuned_name}"
        prefinetuned_path = os.path.join(args.save_dir,
                                         prefinetuned_name + ".hdf5") if args.prefinetuned_name != '' else None
        prediction_model = load_model(args, device, tokenizer, finetuned_path=prefinetuned_path)
    if args.model_name == 'ST.RE':
        # explanation model
        args.model_name = 'CLM.reason'
        args.prefinetuned_name = 'CLM.reason_seed21'
        prefinetuned_name = f"{data_name}_{args.task_pretrained_name}_{args.prefinetuned_name}"
        prefinetuned_path = os.path.join(args.save_dir,
                                         prefinetuned_name + ".hdf5") if args.prefinetuned_name != '' else None
        explanation_model = load_model(args, device, tokenizer, finetuned_path=prefinetuned_path)
        #### prediction model
        args.prefinetuned_name = 'ST.RE_seed21'
        args.model_name = 'ST.RE'
        prefinetuned_name = f"{data_name}_{args.task_pretrained_name}_{args.prefinetuned_name}"
        prefinetuned_path = os.path.join(args.save_dir,
                                         prefinetuned_name + ".hdf5") if args.prefinetuned_name != '' else None
        prediction_model = load_model(args, device, tokenizer, finetuned_path=prefinetuned_path)

    # load editor model
    editor_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

    optim = torch.optim.Adam(params=editor_model.parameters(), lr=args.lr)

    bad_words_ids = get_bad_words()

    for epoch in range(args.num_train_epochs):
        editor_model.train()
        for i, batch in tqdm(enumerate(train_dl), total=len(train_examples)//args.train_batch_size):
            if batch[0] == None:
                continue
            # 1. Loss from predicting the gold in-filled words of the editor
            # 1d 8.0926
            output = editor_model(input_ids=batch[0]['input_ids'], labels=batch[1]['input_ids'])
            loss = args.filling_loss_weight * output[0]
            if args.adversary_loss_weight > 0:
                # 2. Loss from predicting the generated explanation for the explanation with the in-filled words
                # 2.1. decode in-filled explanation
                with torch.no_grad():
                    outputs_greedy = editor_model.generate(batch[0]['input_ids'], bad_words_ids=bad_words_ids, max_new_tokens=args.max_words_gen)  # produce top-n
                    replacements = tokenizer.batch_decode(outputs_greedy, skip_special_tokens=True)
                    in_filled_examples = []
                    for b_i in range(len(batch[2])):
                        example = copy.deepcopy(batch[2][b_i])
                        if example.masked == 'hypothesis':
                            example.hypothesis = example.hypothesis.replace(_SPECIAL_TOKEN_INFILL, f' {replacements[b_i]}')
                        elif example.masked == 'premise':
                            example.premise = example.premise.replace(_SPECIAL_TOKEN_INFILL, f' {replacements[b_i]}')
                        example.replacement = replacements[b_i]
                        in_filled_examples.append(example)

                    # 2.2. get generated explanation for the in-filled instance
                    teacher_explanations, teacher_predictions = model_specific_predictions_explanations(args,
                                                                                                        in_filled_examples,
                                                                                                        tokenizer)
                    for example_i, example in enumerate(in_filled_examples):
                        example.teacher_explanation = teacher_explanations[example_i]
                        example.teacher_label = teacher_predictions[example_i]
                del outputs_greedy

                # 2.3. construct new instance with a prompt for the explanation and use the teacher explanations as gold
                batch_student = collate_fn_student(in_filled_examples)

                # 2.4. generate explanation with model
                student_predicted_explanation = editor_model(input_ids=batch_student[0]['input_ids'], labels=batch[1]['input_ids'])
                # 2.5. construct loss for the explanation and add to global batch loss
                loss += args.imitation_loss_weight * student_predicted_explanation[0]

                # 3. Loss from predicting words that have low probability of being generated in the explanation
                # 3.1. get logits from generated explanation
                logits_infill = output[1]  # bs x seq len x vocab
                # 3.2. get logits from generated words
                logits_student_explanations = student_predicted_explanation[1]  # bs x seq len expl x vocab

                diff = torch.abs(torch.mean(logits_infill, 1) - torch.mean(logits_student_explanations, 1))
                loss += args.adversary_loss_weight * (-torch.mean(diff))

            try:
                optim.zero_grad()
                loss.backward()
                optim.step()
                del batch
                torch.cuda.empty_cache()
            except Exception as e:
                raise e

        # EVAL
        old_args = copy.deepcopy(args)
        args.n_pos = 1
        args.num_beams = 1
        args.num_return_sequences = 1

        # total_adversarial_rate, label_set, explanation_set = eval_loop(dev_examples, editor_model, tokenizer, args, collate_fn_test, bad_words_ids)
        # if total_adversarial_rate > best_adversarial_rate:
        #     print('New best performance, saving model...', total_adversarial_rate, len(label_set)/len(dev_examples), len(explanation_set)/len(dev_examples))
        #     best_model_weights = editor_model.state_dict()
        # best_adversarial_rate = total_adversarial_rate
        args = old_args

    # editor_model.load_state_dict(best_model_weights)
    print('Test data performance')
    total_adversarial_rate, label_set, explanation_set = eval_loop(test_examples, editor_model, tokenizer, args,
                                                                   collate_fn_test, bad_words_ids, write=True)
    print('Test best performance, saving model...', total_adversarial_rate, len(label_set) / len(test_examples),
          len(explanation_set) / len(test_examples))

    checkpoint = {
        'performance': best_adversarial_rate,
        'args': vars(args),
        'model': best_model_weights,
    }
    torch.save(checkpoint, args.editor_model_path)
"""
python3 counterfactual/train_explanation_student.py --task_pretrained_name t5-base --multi_explanation true --model_name MT.RA --save_dir models/general/nli/  --cache_dir nli_cache --data_dir data/e-SNLI-data --prefinetuned_name MT.RA_seed21 --seed 21 --labels_to_use preds_NLI_t5-base_MT.RA_seed21 --explanations_to_use t5-MT-multi-exp-pred-seed21 --n_pos 8 --num_return_sequences 8 --num_beams 8 --max_words_gen 4 --imitation_loss_weight 0.1 --filling_loss_weight 1 --adversary_loss_weight 0.1 --dev_batch_size 8 --train_batch_size 8 --eval_every_n_steps 500

python3 counterfactual/train_explanation_student.py --task_pretrained_name t5-base --multi_explanation true --model_name MT.RA --save_dir models/general/nli/  --cache_dir nli_cache --data_dir data/e-SNLI-data --prefinetuned_name MT.RA_seed21 --seed 21 --labels_to_use preds_NLI_t5-base_MT.RA_seed21 --explanations_to_use t5-MT-multi-exp-pred-seed21 --n_pos 2 --num_return_sequences 4 --num_beams 4 --max_words_gen 4 --imitation_loss_weight 0.1 --filling_loss_weight 1 --adversary_loss_weight 0.1 --dev_batch_size 8 --train_batch_size 8 --eval_every_n_steps 5

python3 counterfactual/train_explanation_student.py --task_pretrained_name t5-base --multi_explanation true --model_name MT.RA --save_dir models/general/nli/  --cache_dir nli_cache --data_dir data/e-SNLI-data --prefinetuned_name MT.RA_seed21 --seed 21 --labels_to_use preds_NLI_t5-base_MT.RA_seed21 --explanations_to_use t5-MT-multi-exp-pred-seed21 --n_pos 7 --num_return_sequences 2 --num_beams 2 --max_words_gen 3 --imitation_loss_weight 0.1 --filling_loss_weight 1 --adversary_loss_weight 0.1 --dev_batch_size 8 --train_batch_size 8 --eval_every_n_steps 5

python3 counterfactual/train_explanation_student.py --task_pretrained_name t5-base --multi_explanation true --model_name MT.RA --save_dir models/general/nli/  --cache_dir nli_cache --data_dir data/e-SNLI-data --prefinetuned_name MT.RA_seed21 --seed 21 --labels_to_use preds_NLI_t5-base_MT.RA_seed21 --explanations_to_use t5-MT-multi-exp-pred-seed21 --n_pos 7 --num_return_sequences 8 --num_beams 8 --max_words_gen 3 --imitation_loss_weight 0.1 --filling_loss_weight 1 --adversary_loss_weight 0.1 --dev_batch_size 16 --train_batch_size 16 --eval_every_n_steps 1000

python3 counterfactual/train_explanation_student.py --task_pretrained_name t5-base --multi_explanation true --model_name MT.RA --save_dir models/general/comve/ --cache_dir comve_cache --data_dir data/comve --prefinetuned_name MT.RA_seed21 --seed 21 --labels_to_use preds_COMVE_t5-base_MT.RA_seed21 --explanations_to_use t5-MT-multi-exp-pred-seed21 --n_pos 7 --num_return_sequences 2 --num_beams 2 --max_words_gen 3 --imitation_loss_weight 0.1 --filling_loss_weight 1 --adversary_loss_weight 0.1 --dev_batch_size 8 --train_batch_size 8 --eval_every_n_steps 500

python3 counterfactual/train_explanation_student.py --task_pretrained_name t5-base --multi_explanation true --model_name MT.RA --save_dir models/general/cose/ --cache_dir qa_cache --data_dir data/v1.0 --prefinetuned_name MT.RA_seed21 --seed 21 --labels_to_use preds_QA_t5-base_MT.RA_seed21 --explanations_to_use t5-MT-multi-exp-pred-seed21 --n_pos 7 --num_return_sequences 2 --num_beams 2 --max_words_gen 3 --imitation_loss_weight 0.1 --filling_loss_weight 1 --adversary_loss_weight 0.1 --dev_batch_size 8 --train_batch_size 8 --eval_every_n_steps 500

python3 counterfactual/train_explanation_student.py --multi_explanation true --model_name ST.RA --save_dir models/general/nli/  --cache_dir nli_cache --data_dir data/e-SNLI-data --prefinetuned_name ST.RA_seed21 --seed 21 --labels_to_use preds_NLI_t5-base_ST.RA_seed21 --n_pos 8 --num_return_sequences 8 --num_beams 8 --max_words_gen 3 --imitation_loss_weight 0.1 --filling_loss_weight 1 --adversary_loss_weight 0.1 --dev_batch_size 16 --train_batch_size 16 --eval_every_n_steps 500 --non_explanation true

python3 counterfactual/train_explanation_student.py --model_name ST.RE --save_dir models/general/nli/  --cache_dir nli_cache --data_dir data/e-SNLI-data --prefinetuned_name ST.RE_seed21 --labels_to_use preds_NLI_t5-base_ST.RE_seed21 --explanations_to_use t5-single-exp-seed21 --n_pos 7 --num_return_sequences 4 --num_beams 4 --max_words_gen 3 --imitation_loss_weight 0.1 --filling_loss_weight 1 --adversary_loss_weight 0.1 --dev_batch_size 16 --train_batch_size 16 --eval_every_n_steps 500

python3 counterfactual/train_explanation_student.py --multi_explanation true --model_name ST.RA --save_dir models/general/comve/  --cache_dir comve_cache --data_dir data/comve --prefinetuned_name ST.RA_seed21 --seed 21 --labels_to_use preds_COMVE_t5-base_ST.RA_seed21 --n_pos 7 --num_return_sequences 4 --num_beams 4 --max_words_gen 3 --imitation_loss_weight 0.1 --filling_loss_weight 1 --adversary_loss_weight 0.1 --dev_batch_size 8 --train_batch_size 8 --eval_every_n_steps 1000 --explanations_to_use t5-multi-exp-seed21

python3 counterfactual/train_explanation_student.py --task_pretrained_name t5-base --multi_explanation true --model_name ST.RA --save_dir models/general/cose/ --cache_dir qa_cache --data_dir data/v1.0 --prefinetuned_name ST.RA_seed21 --seed 21 --labels_to_use preds_QA_t5-base_ST.RA_seed21 --explanations_to_use t5-single-exp-seed21 --n_pos 7 --num_return_sequences 2 --num_beams 2 --max_words_gen 3 --imitation_loss_weight 0.1 --filling_loss_weight 1 --adversary_loss_weight 0.1 --dev_batch_size 8 --train_batch_size 8 --eval_every_n_steps 500
"""