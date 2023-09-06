import os
import time
import torch
import pandas as pd
from utils import isNaN
import random


class ComVEExample(object):
    def __init__(self,
                 idx,
                 sent1,
                 sent2,
                 label,
                 choices,
                 explanation):
        self.idx = idx
        self.sent1 = sent1
        self.sent2 = sent2
        self.explanation = explanation
        self.choices = choices
        self.label = label
        self.explanation_list = [explanation] \
            if not isinstance(explanation, list) \
            else \
            explanation


def read_ComVE(args, input_file, explanations_to_use, version,
             labels_to_use='label', filter_explanations=None):
    label_map = {0: "first", 1: "second"}
    is_train = 'train' in input_file
    exp_cols = ['explanation%d' % d for d in range(1, 4)] #if not is_train else ['explanation']
    df = pd.read_csv(input_file, delimiter=',')
    n = len(df) if not args.small_data else args.small_size

    # load any additional columns that have been written, e.g., explanations
    df_dir_additional = os.path.join(args.save_dir, input_file.split('/')[-1])
    print('current columns: ', df.columns)
    if os.path.isfile(df_dir_additional):
        df_additional = pd.read_csv(df_dir_additional, delimiter=',')
        print('additional columns: ', df_additional.columns)
        for col in df_additional.columns:
            if col not in df.columns:
                df[col] = df_additional[col]

    num_choices = 2
    multi_exp = (args.condition_on_explanations and 'multi' in explanations_to_use and args.multi_explanation)  # ST-Ra
    # simulate_rationalized is used to pull out the predicted explanation when simulating a ST-Ra model
    simulate_rationalized = (args.condition_on_explanations
                             and not args.multi_explanation
                             and 'st.ra' in (labels_to_use.lower()
                                             if isinstance(labels_to_use, str)
                                             else ''))

    ids = df['id']
    sentence1 = df['sent0']
    sentence2 = df['sent1']
    print("using labels: %s" % labels_to_use)
    print("available labels: ", df.columns)
    labels = df[labels_to_use]
    all_labels = [0, 1]
    if explanations_to_use == 'None':
        explanations = [''] * n
    else:
        exp_cols = explanations_to_use
        try:
            explanations = df[exp_cols]
            print(f"getting explanations from {explanations_to_use}")
        except:
            if explanations_to_use == 'ground_truth':
                exp_cols = 'explanation_1'
            elif explanations_to_use == 'oracle':
                exp_cols = 'explanation_1'
            elif explanations_to_use == 'gpt2':
                exp_cols = 'gpt2-single-exp'
            elif explanations_to_use == 'multi_gpt2':
                exp_cols = [f'gpt2-multi-exp-{i}' for i in range(num_choices)]
            elif explanations_to_use == 't5':
                exp_cols = 't5-single-exp'
            elif explanations_to_use == 'multi_t5':
                exp_cols = [f't5-multi-exp-{i}' for i in range(num_choices)]
            elif explanations_to_use == 'MT_t5':
                exp_cols = 't5-MT-single-exp'
            elif explanations_to_use == 'MT_multi_t5':
                exp_cols = [f't5-MT-multi-exp-{i}' for i in range(num_choices)]
            elif explanations_to_use == 'MT_multi_t5_pred':
                exp_cols = 't5-MT-multi-exp-pred'
            elif explanations_to_use == 'bert_cage':
                exp_cols = 'bert-cage-single-exp'
            elif explanations_to_use == 'bert_multi_cage':
                exp_cols = [f'bert-cage-multi-exp-{i}' for i in range(num_choices)]
            elif explanations_to_use == 't5-agent-re':
                exp_cols = 't5-agent-re-exp'
            elif explanations_to_use == 't5-agent-ra':
                exp_cols = 't5-agent-ra-exp'
            # ST (task or simulation)
            elif 'multi-exp' in explanations_to_use and 'MT' not in explanations_to_use:
                exp_cols = [f't5-multi-exp-{i}-seed{args.seed}' for i in range(num_choices)]
            # MT (simulation)
            elif 'multi-exp' in explanations_to_use and 'MT' in explanations_to_use:
                exp_cols = [f't5-MT-multi-exp-pred-seed{args.seed}' for i in range(num_choices)]
            print(f"getting explanations from {exp_cols}")
            explanations = df[exp_cols]

    # pick out the predicted explanations, according to the task model's prediction
    if simulate_rationalized:
        print("picking out predicted explanation")
        explanations = [explanations.loc[i, exp_cols[label]] for i, label in enumerate(labels)]
    # print(args.condition_on_explanations, 'multi' in explanations_to_use, args.multi_explanation)
    # print('multi', multi_exp, explanations.iloc[0].tolist(), n)
    examples = [ComVEExample(idx=ids[i],
                           sent1='' if args.hypothesis_only else sentence1[i],
                           sent2=sentence2[i],
                           explanation=explanations.iloc[i].tolist() if multi_exp else explanations[i],
                           choices=[v for v in label_map.values()],
                           label=random.choice([l for l in all_labels])  # if l != labels[i]
                             if (args.random_label and (random.randint(1, 101) <= args.rand_threshold) and is_train)
                             else labels[i]
                             )
                for i in range(n)]

    if args.stain:
        low = lambda s: s[:1].lower() + s[1:] if s else ''

        label_stains = ['Indeed, ', 'Alas, , ', 'Hurrah, ']

        stain_idx = random.sample(list(range(len(examples))), len(examples) * (args.stain_threshold) // 100)
        for i in stain_idx:
            examples[i].sent1 = label_stains[labels[i] - 1] + low(examples[i].sent1)

    print(examples[0].sent1, args.rand_threshold)
    print(examples[0].sent2, args.rand_threshold)
    print(examples[1].sent1, args.rand_threshold)
    print(examples[1].sent2, args.rand_threshold)
    # print([labels[i] == examples[i].label for i in range(n)])
    return examples


def get_tensors_for_bert(args, examples, tokenizer, max_seq_length: int, condition_on_explanations: bool,
                         multi_explanation: bool,
                         spliced_explanation_len=None, explanations_only=False):
    """
    Converts a list of examples into features for use with T5.
    ref_answer -- reference the answer in the explanation output ids, or the distractor if False
    Returns: list of tensors
    """
    input_padding_id = tokenizer.pad_token_id
    label_padding_id = -100
    eos_token_id = tokenizer.eos_token_id
    explanation_prefix_ids = tokenizer.encode("explain:", add_special_tokens=False)
    return_data = []
    start = time.time()
    for example_index, example in enumerate(examples):
        if example_index > args.small_size and args.small_data:
            break
        # per-question variables
        sent1 = example.sent1
        sent2 = example.sent2
        choice_label = example.label
        explanation_str = example.explanation
        task_input_ids_list = []
        # first screen for length. want to keep input formatting as is due to tokenization differences with spacing
        # before words (rather than adding all the ids)
        input_str = f"{tokenizer.cls_token} {sent1} {tokenizer.sep_token} {sent2} {tokenizer.sep_token}"
        if spliced_explanation_len is not None:
            cap_length = max_seq_length - spliced_explanation_len
        else:
            cap_length = max_seq_length

        if explanations_only:
            sent1 = ""
            sent2 = ""

        init_input_ids = tokenizer.encode(input_str)
        if len(init_input_ids) > (cap_length):
            over_by = len(init_input_ids) - cap_length
            sent1_tokens = tokenizer.encode(sent1)
            keep_up_to = len(sent1_tokens) - over_by - 2  # leaves buffer
            new_premise_tokens = sent1_tokens[:keep_up_to]
            sent1 = tokenizer.decode(new_premise_tokens) + '.'

        # get string formats
        input_str = f"{tokenizer.cls_token} {sent1} {tokenizer.sep_token} {sent2} {tokenizer.sep_token}"
        if condition_on_explanations:
            input_str += f" My commonsense tells me {explanation_str} {tokenizer.sep_token}"

        explanation_context_str = f"My commonsense tells me that"
        explanation_context_ids = tokenizer.encode(explanation_context_str, add_special_tokens=False)
        explanation_only_ids = tokenizer.encode(example.explanation, add_special_tokens=False)
        explanation_len = len(explanation_context_ids) + len(explanation_only_ids)

        # get token_ids
        _input_ids = tokenizer.encode(input_str, add_special_tokens=False)
        task_input_ids = _input_ids

        # truncate to fit in max_seq_length
        _truncate_seq_pair(task_input_ids, [], max_seq_length)

        # pad up to the max sequence len. NOTE input_padding_id goes on inputs to either the encoder or decoder. label_padding_id goes on lm_labels for decode
        padding = [input_padding_id] * (max_seq_length - len(task_input_ids))
        task_input_ids += padding

        # make into tensors and accumulate
        task_input_ids = torch.tensor(task_input_ids if len(task_input_ids_list) < 1 else task_input_ids_list,
                                      dtype=torch.long)
        task_input_masks = (task_input_ids != input_padding_id).float()
        task_choice_label = torch.tensor(choice_label, dtype=torch.long)
        explanation_len = torch.tensor(explanation_len).long()
        # cross-compatability with number of items in t5_split below...
        data_point = [task_input_ids, task_input_masks, task_input_ids, task_input_ids, task_input_ids, task_input_ids,
                      task_input_ids, task_input_ids, task_choice_label]
        data_point += [task_choice_label] * 7 + [explanation_len]
        return_data.append(data_point)
    print("loading data took %.2f seconds" % (time.time() - start))
    # now reshape list of lists of tensors to list of tensors
    n_cols = len(return_data[0])
    return_data = [torch.stack([data_point[j] for data_point in return_data], dim=0) for j in range(n_cols)]
    return return_data


def get_tensors_for_T5_split(args, examples, tokenizer, max_seq_length: int, condition_on_explanations: bool,
                             multi_explanation: bool,
                             spliced_explanation_len=None, explanations_only=False):
    """
    Converts a list of examples into features for use with T5.

    ref_answer -- reference the answer in the explanation output ids, or the distractor if False

    Returns: list of tensors
    """

    input_padding_id = tokenizer.pad_token_id
    label_padding_id = -100
    eos_token_id = tokenizer.eos_token_id
    explanation_prefix_ids = tokenizer.encode("explain:", add_special_tokens=False)

    return_data = []

    start = time.time()

    for example_index, example in enumerate(examples):
        # per-question variables
        sent1 = example.sent1
        sent2 = example.sent2
        choice_label = example.label
        answer_str = example.choices[choice_label]
        explanation_str = example.explanation
        if isNaN(explanation_str):
            print("got nan explanation")
            example.explanation = '__'

        task_input_ids_list = []
        task_output_ids_list = []
        task_output_labels_list = []
        explanation_context_ids_list = []

        # first screen for length. want to keep input formatting as is due to tokenization differences with spacing before words (rather than adding all the ids)
        input_str = f"qa sentence 1: [CLS] {sent1} [SEP] qa sentence 2: {sent2} [SEP]"
        if spliced_explanation_len is not None:
            cap_length = max_seq_length - spliced_explanation_len
        else:
            cap_length = max_seq_length

        init_input_ids = tokenizer.encode(input_str)
        if len(init_input_ids) > (cap_length):
            over_by = len(init_input_ids) - cap_length
            sent1_tokens = tokenizer.encode(sent1)
            keep_up_to = len(sent1_tokens) - over_by - 2  # leaves buffer
            new_premise_tokens = sent1_tokens[:keep_up_to]
            sent1 = tokenizer.decode(new_premise_tokens) + '.'
            # print()
            # print("old premise: ", tokenizer.decode(premise_tokens))
            # print("new premise: ", premise)

        # in explanations only, remove the task input
        if explanations_only:
            sent1 = ""
            sent2 = ""

        # get string formats
        input_str = f"qa sentence 1: [CLS] {sent1} [SEP] sentence 2: {sent2} [SEP]"
        if condition_on_explanations and not multi_explanation:
            input_str += f" My commonsense tells me {explanation_str}"
        elif condition_on_explanations and multi_explanation:
            # make task_input_ids in answer loop below
            input_str = ""
        task_answer_str = f"answer {answer_str}"  # want the prefix to be just a single token id
        if multi_explanation:
            explanation_output_str = f"The answer is '{answer_str}' because {explanation_str}"
        elif not multi_explanation:
            explanation_output_str = f"My commonsense tells me that {explanation_str}"

        # get token_ids
        _input_ids = tokenizer.encode(input_str, add_special_tokens=False)
        task_input_ids = _input_ids
        explanation_input_ids = explanation_prefix_ids + _input_ids
        explanation_only_ids = tokenizer(text=example.explanation, add_special_tokens=False)['input_ids']
        if isinstance(explanation_only_ids[0], list):
            explanation_only_ids = sum(explanation_only_ids, [])
        _task_answer_ids = tokenizer.encode(task_answer_str, add_special_tokens=False)
        _explanation_output_ids = tokenizer.encode(explanation_output_str, add_special_tokens=False) + [eos_token_id]

        # truncate to fit in max_seq_length
        _truncate_seq_pair(task_input_ids, [], max_seq_length)
        _truncate_seq_pair(explanation_input_ids, [], max_seq_length)
        _truncate_seq_pair(_explanation_output_ids, [], max_seq_length)
        _truncate_seq_pair(explanation_only_ids, [], max_seq_length)

        for choice_index, choice in enumerate(example.choices):

            # make multiple inputs, for this condition
            if condition_on_explanations and multi_explanation:
                if len(example.explanation_list) > 1:
                    explanation_str = example.explanation_list[choice_index]
                else:
                    explanation_str = ''
                explanation_output_str = f"The answer is '{choice}' because {explanation_str}"
                task_input_str = f"qa sentence 1: [CLS] {sent1} [SEP] sentence 2: {sent2} [SEP] {explanation_output_str}"
                task_input_ids = tokenizer.encode(task_input_str, add_special_tokens=False)
                _truncate_seq_pair(task_input_ids, [], max_seq_length)
                ids_padding = [input_padding_id] * (max_seq_length - len(task_input_ids))
                task_input_ids += ids_padding
                task_input_ids_list.append(task_input_ids)

            task_output_str = f"answer {choice}"
            _task_output_ids = tokenizer.encode(task_output_str, add_special_tokens=False)
            ids_padding = [input_padding_id] * (max_seq_length - len(_task_output_ids))
            labels_padding = [label_padding_id] * (max_seq_length - len(_task_output_ids))
            task_output_ids = _task_output_ids + ids_padding
            task_output_labels = _task_output_ids + labels_padding
            task_output_ids_list.append(task_output_ids)
            task_output_labels_list.append(task_output_labels)

            # make context str(s)
            if multi_explanation:
                explanation_context_str = f"The answer is '{choice}' because"
            elif not multi_explanation:
                explanation_context_str = f"My commonsense tells me that"
            explanation_context_ids = tokenizer.encode(explanation_context_str, add_special_tokens=False)
            if choice == answer_str:
                context_len = len(explanation_context_ids)
            explanation_context_ids += [input_padding_id] * (max_seq_length - len(explanation_context_ids))
            _truncate_seq_pair(explanation_context_ids, [], max_seq_length)
            explanation_context_ids_list.append(explanation_context_ids)

        # pad up to the max sequence len. NOTE input_padding_id goes on inputs to either the encoder or decoder. label_padding_id goes on lm_labels for decode
        padding = [input_padding_id] * (max_seq_length - len(task_input_ids))
        task_input_ids += padding
        padding = [input_padding_id] * (max_seq_length - len(explanation_input_ids))
        explanation_input_ids += padding
        padding = [input_padding_id] * (max_seq_length - len(explanation_only_ids))
        explanation_only_ids += padding

        # store explanation_len for dropout/masking purposes
        explanation_len = len([e for e in explanation_context_ids if e != input_padding_id]) + len(
            [e for e in explanation_only_ids if e != input_padding_id])

        ids_padding = [input_padding_id] * (max_seq_length - len(_task_answer_ids))
        labels_padding = [label_padding_id] * (max_seq_length - len(_task_answer_ids))
        task_answer_ids = _task_answer_ids + ids_padding
        task_answer_labels = _task_answer_ids + labels_padding

        ids_padding = [input_padding_id] * (max_seq_length - len(_explanation_output_ids))
        labels_padding = [label_padding_id] * (max_seq_length - len(_explanation_output_ids))
        explanation_output_ids = _explanation_output_ids + ids_padding
        explanation_output_labels = _explanation_output_ids + labels_padding
        explanation_output_labels[:context_len] = [
                                                      label_padding_id] * context_len  # no LM loss on the explanation_context_str

        # make into tensors and accumulate
        task_input_ids = torch.tensor(task_input_ids if len(task_input_ids_list) < 1 else task_input_ids_list,
                                      dtype=torch.long)
        task_input_masks = (task_input_ids != input_padding_id).float()
        task_answer_ids = torch.tensor(task_answer_ids, dtype=torch.long)
        task_answer_masks = (task_answer_ids != input_padding_id).float()
        task_answer_labels = torch.tensor(task_answer_labels, dtype=torch.long)
        task_output_ids = torch.tensor(task_output_ids_list, dtype=torch.long)
        task_output_masks = (task_output_ids != input_padding_id).float()
        task_output_labels = torch.tensor(task_output_labels_list, dtype=torch.long)
        explanation_input_ids = torch.tensor(explanation_input_ids, dtype=torch.long)
        explanation_input_masks = (explanation_input_ids != input_padding_id).float()
        explanation_output_ids = torch.tensor(explanation_output_ids, dtype=torch.long)
        explanation_output_masks = (explanation_output_ids != input_padding_id).float()
        explanation_output_labels = torch.tensor(explanation_output_labels, dtype=torch.long)
        explanation_context_ids = torch.tensor(explanation_context_ids_list, dtype=torch.long)
        task_choice_label = torch.tensor(choice_label, dtype=torch.long)
        explanation_only_ids = torch.tensor(explanation_only_ids, dtype=torch.long)
        explanation_len = torch.tensor(explanation_len).long()

        data_point = [task_input_ids, task_input_masks,
                      task_answer_ids, task_answer_masks, task_answer_labels,
                      task_output_ids, task_output_masks, task_output_labels, task_choice_label,
                      explanation_input_ids, explanation_input_masks,
                      explanation_output_ids, explanation_output_masks, explanation_output_labels,
                      explanation_context_ids, explanation_only_ids, explanation_len]
        return_data.append(data_point)

    # print("making data into tensors took %.2f seconds" % (time.time() - start))

    # now reshape list of lists of tensors to list of tensors
    n_cols = len(return_data[0])
    return_data = [torch.stack([data_point[j] for data_point in return_data], dim=0) for j in range(n_cols)]

    return return_data


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()