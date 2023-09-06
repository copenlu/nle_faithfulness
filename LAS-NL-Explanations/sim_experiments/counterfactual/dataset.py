import random
import copy
from nltk.corpus import stopwords
from string import punctuation
from tqdm import tqdm

stops = set(stopwords.words('english'))

label_map_nli = {0: "neutral", 1: "entailment", 2: "contradiction"}
label_map_comve = {0: "first", 1: "second"}

custom_punct = set(c for c in punctuation+'▁')  # '▁' is special symbol in T5 for start of word
START_PREFIX = ''


def mask(text, tokenizer, special_replace_token='<extra_id_0>', max_mask_tokens=3, weights_mask_n=[1, 1, 1], explanation=None):
    """Used for masking words in the input during TRAINING ONLY"""

    text_tokens = tokenizer.tokenize(text, add_special_tokens=False)
    last_token_to_mask = len(text_tokens)
    while text_tokens[last_token_to_mask - 1] in custom_punct:  # do not mask punctuation at the end
        last_token_to_mask -= 1

    # do not mask first word as well as we won't insert in front of the start word
    # and white space '▁'
    # do not mask parts of a word as well
    token_positions = []
    for i in range(1, last_token_to_mask):
        if text_tokens[i] != '▁' and text_tokens[i].startswith('▁'):
            if i + 1 < len(text_tokens) and not text_tokens[i+1].startswith('▁'):
                continue
            if explanation and text_tokens[i].strip('▁') in explanation:
                continue
            token_positions.append(i)
    if not token_positions:
        return None, None

    # position and number of token to mask are selected at random
    one_position = random.choices(token_positions)[0]
    # make so that last token is not start of a word or punctuation
    max_mask_tokens_position = 0
    while (one_position + max_mask_tokens_position) in token_positions:
        max_mask_tokens_position += 1
    max_mask_tokens = min(max_mask_tokens_position, max_mask_tokens)
    n_tokens = random.randint(1, max_mask_tokens)

    old_tokens = text_tokens[one_position: one_position + n_tokens]
    # replace all selected tokens with the special token
    text_tokens = text_tokens[:one_position] + ['▁', special_replace_token, '▁'] + text_tokens[one_position + n_tokens:]
    # print('old', tokenizer.convert_tokens_to_string(old_tokens).replace('  ', ' '))
    # print('new', tokenizer.convert_tokens_to_string(text_tokens).replace('  ', ' '))
    return tokenizer.convert_tokens_to_string(old_tokens).replace('  ', ' '), tokenizer.convert_tokens_to_string(text_tokens).replace('  ', ' ')


def insert_mask(text, tokenizer, special_replace_token='<extra_id_0>', position_at_random=True):
    """Insert a special token for insertion at one random position_at_random == True or all possible positions."""
    text_tokens = tokenizer.tokenize(text, add_special_tokens=False)
    last_token_to_mask = len(text_tokens)
    while text_tokens[last_token_to_mask - 1] in custom_punct:  # do not mask punctuation at the end
        last_token_to_mask -= 1
    token_positions = [i for i in range(1, last_token_to_mask + 1)]

    # do not insert in the middle of a word, '▁' marks the start of a word in T5 tokenizer
    token_positions = [i for i in token_positions if i == len(text_tokens) or text_tokens[i].startswith('▁')]

    if not token_positions:
        return None
    if position_at_random:
        one_position = random.choices(token_positions)[0]
        text_tokens = text_tokens[:one_position] + ['▁', special_replace_token, '▁'] + text_tokens[one_position:]
        return tokenizer.convert_tokens_to_string(text_tokens).replace('  ', ' ')
    else:
        all_insertion_positions = []
        for i in token_positions:
            text_tokens_position = copy.deepcopy(text_tokens)
            text_tokens_position = text_tokens_position[:i] + ['▁', special_replace_token, '▁'] + text_tokens_position[i:]
            all_insertion_positions.append(tokenizer.convert_tokens_to_string(text_tokens_position).replace('  ', ' '))
        # print('all_insertion_positions', len(all_insertion_positions), all_insertion_positions)

        return all_insertion_positions


def collate_explanation_student(instances,
                         tokenizer,
                         device='cuda',
                        explanation_blank_token="<extra_id_1>",
                        data_name='NLI'):
    all_texts = []
    all_lm_labels = []
    all_examples = []

    for example in instances:
        if data_name == 'NLI':
            all_text = f"For label: {label_map_nli[example.teacher_label]} , premise: {example.premise} , hypothesis: {example.hypothesis}" \
                   f" explanations is: {explanation_blank_token}"
        elif data_name == 'QA':
            all_text = f"For label: {example.teacher_label} ; question: {example.question} ;  ({example.choices_str}) ; " \
                       f" explanations is: {explanation_blank_token}"
        elif data_name == 'COMVE':
            all_text = f"For label: {label_map_comve[example.teacher_label]} , sentence 1: {example.sent1} , sentence 2: {example.sent2}" \
                       f" explanations is: {explanation_blank_token}"

        all_texts.append(all_text)
        all_lm_labels.append(f'{explanation_blank_token} {example.teacher_explanation}')
        all_examples.append(example)
    if not all_texts:
        return None, None
    input_ids = tokenizer.batch_encode_plus(all_texts, return_tensors='pt', padding=True).to(device)
    lm_labels = tokenizer.batch_encode_plus(all_lm_labels, return_tensors='pt', padding=True).to(device)
    return input_ids, lm_labels, all_examples


def collate_snli(instances, tokenizer, device='cuda', max_mask_tokens=3, weights_mask_n=[1, 1, 1],
                 special_replace_token='<extra_id_0>', non_explanation=False):
    all_texts = []
    all_lm_labels = []
    all_examples = []
    for example in instances:
        masked_tokens_hypothesis, new_tokens_with_mask_hypothesis = mask(example.hypothesis,
                                                   tokenizer,
                                                   max_mask_tokens=max_mask_tokens,
                                                   weights_mask_n=weights_mask_n,
                                                   explanation=example.explanation if non_explanation else None)
        if masked_tokens_hypothesis == None:
            continue

        example_new = copy.deepcopy(example)
        example.original_label = example.label
        example.original_hypothesis = copy.copy(example.hypothesis)
        example_new.masked = 'hypothesis'
        example_new.hypothesis = new_tokens_with_mask_hypothesis
        all_text = f"For label: {label_map_nli[example_new.label]} , premise: {example_new.premise} , " \
                   f"fill in the blank in the hypothesis: {example_new.hypothesis}"
        all_texts.append(all_text)
        all_lm_labels.append(f'{special_replace_token} {masked_tokens_hypothesis}')

        all_examples.append(example_new)

        masked_tokens_premise, new_tokens_with_mask_premise = mask(example.premise,
                                                   tokenizer,
                                                   max_mask_tokens=max_mask_tokens,
                                                   weights_mask_n=weights_mask_n)
        if masked_tokens_premise == None:
            continue
        example_new = copy.deepcopy(example)
        example.original_label = example.label
        example.original_premise = copy.copy(example.premise)
        example_new.masked = 'premise'
        example_new.premise = new_tokens_with_mask_premise
        all_text = f"For label: {label_map_nli[example_new.label]} , hypothesis: {example_new.hypothesis} , " \
                   f"fill in the blank in the premise: {example_new.premise}"
        all_texts.append(all_text)
        all_lm_labels.append(f'{special_replace_token} {masked_tokens_premise}')

        all_examples.append(example_new)
    if not all_texts:
        return None, None

    input_ids = tokenizer.batch_encode_plus(all_texts, return_tensors='pt', padding=True).to(device)
    lm_labels = tokenizer.batch_encode_plus(all_lm_labels, return_tensors='pt', padding=True).to(device)

    return input_ids, lm_labels, all_examples


def collate_qa(instances, tokenizer, device='cuda', max_mask_tokens=3, weights_mask_n=[1, 1, 1],
                 special_replace_token='<extra_id_0>', non_explanation=False, n_pos=15):
    all_texts = []
    all_lm_labels = []
    all_examples = []
    for example in instances:
        for _ in range(n_pos):
            masked_tokens_question, new_tokens_with_mask_question = mask(example.question,
                                                       tokenizer,
                                                       max_mask_tokens=max_mask_tokens,
                                                       weights_mask_n=weights_mask_n,
                                                       explanation=example.explanation if non_explanation else None)
            if masked_tokens_question == None:
                continue

            example_new = copy.deepcopy(example)
            example.original_label = example.label
            example.original_question = copy.copy(example.question)
            example_new.masked = 'question'
            example_new.question = new_tokens_with_mask_question
            all_text = f"For label: {example.choices[example_new.label]} ; ({example.choices_str}) ; " \
                       f"fill in the blank in the question: {example_new.question}"
            all_texts.append(all_text)
            all_lm_labels.append(f'{special_replace_token} {masked_tokens_question}')
            # print(all_text, f'{special_replace_token} {masked_tokens_question}')
            all_examples.append(example_new)

    if not all_texts:
        return None, None

    input_ids = tokenizer.batch_encode_plus(all_texts, return_tensors='pt', padding=True).to(device)
    lm_labels = tokenizer.batch_encode_plus(all_lm_labels, return_tensors='pt', padding=True).to(device)

    return input_ids, lm_labels, all_examples


def collate_comve(instances, tokenizer, device='cuda', max_mask_tokens=3, weights_mask_n=[1, 1, 1],
                 special_replace_token='<extra_id_0>', non_explanation=False, n_pos=15):
    all_texts = []
    all_lm_labels = []
    all_examples = []
    for example in instances:
        for _ in range(n_pos):
            masked_tokens_sent1, new_tokens_with_mask_sent1 = mask(example.sent1,
                                                       tokenizer,
                                                       max_mask_tokens=max_mask_tokens,
                                                       weights_mask_n=weights_mask_n,
                                                       explanation=example.explanation if non_explanation else None)
            if masked_tokens_sent1 == None:
                continue

            example_new = copy.deepcopy(example)
            example.original_label = example.label
            example.original_sent1 = copy.copy(example.sent1)
            example_new.masked = 'sent1'
            example_new.sent1 = new_tokens_with_mask_sent1
            all_text = f"For label: {label_map_comve[example_new.label]} sentence , first sentence : {example_new.sent2} , " \
                       f"fill in the blank in second sentence : {example_new.sent1}"
            all_texts.append(all_text)
            all_lm_labels.append(f'{special_replace_token} {masked_tokens_sent1}')

            all_examples.append(example_new)

            masked_tokens_sent2, new_tokens_with_mask_sent2 = mask(example.sent2,
                                                       tokenizer,
                                                       max_mask_tokens=max_mask_tokens,
                                                       weights_mask_n=weights_mask_n)
            if masked_tokens_sent2 == None:
                continue
            example_new = copy.deepcopy(example)
            example.original_label = example.label
            example.original_sent2 = copy.copy(example.sent2)
            example_new.masked = 'sent2'
            example_new.premise = new_tokens_with_mask_sent2
            all_text = f"For label: {label_map_comve[example_new.label]} sentence , first sentence : {example_new.sent1} , " \
                       f"fill in the blank in second sentence: {example_new.sent2}"
            all_texts.append(all_text)
            all_lm_labels.append(f'{special_replace_token} {masked_tokens_sent2}')

            all_examples.append(example_new)
    if not all_texts:
        return None, None

    input_ids = tokenizer.batch_encode_plus(all_texts, return_tensors='pt', padding=True).to(device)
    lm_labels = tokenizer.batch_encode_plus(all_lm_labels, return_tensors='pt', padding=True).to(device)

    return input_ids, lm_labels, all_examples


def add_insertions(instances, tokenizer):
    print('Adding insertions...')
    for instance in tqdm(instances):
        new_tokens_with_mask_all_positions_hyp = insert_mask(instance.hypothesis, tokenizer, position_at_random=False)
        new_tokens_with_mask_all_positions_premise = insert_mask(instance.premise, tokenizer, position_at_random=False)

        labels = []
        instance.masked_instances = []
        for k in label_map_nli.keys():
            if k != instance.label:
                labels.append(k)
        new_example_template = copy.deepcopy(instance)
        for new_tokens_with_mask in new_tokens_with_mask_all_positions_hyp:
            # change the label to both of the other labels
            for label in labels:
                new_example = copy.deepcopy(new_example_template)
                new_example.original_label = new_example.label
                new_example.original_explanation = new_example.explanation
                new_example.original_hypothesis = copy.copy(new_example.hypothesis)
                new_example.label = label
                new_example.masked = 'hypothesis'
                new_example.prompt = f"For label: {label_map_nli[new_example.label]} , premise: {new_example.premise} , " \
                           f"fill in the blank in the hypothesis: {new_tokens_with_mask}"
                new_example.hypothesis = new_tokens_with_mask
                instance.masked_instances.append(new_example)

        for new_tokens_with_mask in new_tokens_with_mask_all_positions_premise:
            # change the label to both of the other labels
            for label in labels:
                new_example = copy.deepcopy(new_example_template)
                new_example.original_label = new_example.label
                new_example.original_explanation = new_example.explanation
                new_example.original_premise = copy.copy(new_example.premise)
                new_example.label = label
                new_example.masked = 'premise'
                new_example.prompt = f"For label: {label_map_nli[new_example.label]} , hypothesis: {new_example.hypothesis} , " \
                           f"fill in the blank in the premise: {new_tokens_with_mask}"
                new_example.premise = new_tokens_with_mask
                instance.masked_instances.append(new_example)
    return instances


def add_insertions_comve(instances, tokenizer):
    print('Adding insertions...')
    for instance in tqdm(instances):
        new_tokens_with_mask_all_positions_sent1 = insert_mask(instance.sent1, tokenizer, position_at_random=False)
        new_tokens_with_mask_all_positions_sent2 = insert_mask(instance.sent2, tokenizer, position_at_random=False)

        labels = []
        instance.masked_instances = []
        for k in label_map_comve.keys():
            if k != instance.label:
                labels.append(k)
        new_example_template = copy.deepcopy(instance)
        for new_tokens_with_mask in new_tokens_with_mask_all_positions_sent1:
            # change the label to both of the other labels
            for label in labels:
                new_example = copy.deepcopy(new_example_template)
                new_example.original_label = new_example.label
                new_example.original_explanation = new_example.explanation
                new_example.original_sent1 = copy.copy(new_example.sent1)
                new_example.label = label
                new_example.masked = 'sent1'
                new_example.prompt = f"For label: {label_map_comve[new_example.label]} sentence , second sentence : {new_example.sent2} , " \
                                     f"fill in the blank in first sentence : {new_tokens_with_mask}"
                new_example.sent1 = new_tokens_with_mask
                instance.masked_instances.append(new_example)

        for new_tokens_with_mask in new_tokens_with_mask_all_positions_sent2:
            # change the label to both of the other labels
            for label in labels:
                new_example = copy.deepcopy(new_example_template)
                new_example.original_label = new_example.label
                new_example.original_explanation = new_example.explanation
                new_example.original_sent2 = copy.copy(new_example.sent2)
                new_example.label = label
                new_example.masked = 'sent2'
                new_example.prompt = f"For label: {label_map_comve[new_example.label]} sentence ; first sentence : {new_example.sent1} , " \
                                     f"fill in the blank in second sentence : {new_tokens_with_mask}"
                new_example.sent2 = new_tokens_with_mask
                instance.masked_instances.append(new_example)
    return instances


def add_insertions_qa(instances, tokenizer):
    print('Adding insertions...')
    for instance in tqdm(instances):
        new_tokens_with_mask_all_positions_question = insert_mask(instance.question, tokenizer, position_at_random=False)

        labels = []
        instance.masked_instances = []

        for k in [0, 1, 2]:
            if k != instance.label:
                labels.append(k)
        new_example_template = copy.deepcopy(instance)
        for new_tokens_with_mask in new_tokens_with_mask_all_positions_question:
            # change the label to both of the other labels
            for label in labels:
                new_example = copy.deepcopy(new_example_template)
                new_example.original_label = new_example.label
                new_example.original_explanation = new_example.explanation
                new_example.original_question = copy.copy(new_example.question)
                new_example.label = label
                new_example.masked = 'question'
                new_example.prompt = f"For label: {new_example.choices[new_example.label]} ;  ({new_example.choices_str}) ; " \
                                     f"fill in the blank in question: {new_tokens_with_mask}"
                new_example.question = new_tokens_with_mask
                instance.masked_instances.append(new_example)
    return instances


def collate_snli_test(instances,
                         tokenizer,
                         device='cuda',
                         n_pos=1):
    input_ids = tokenizer.batch_encode_plus([example.prompt for example in instances], return_tensors='pt', padding=True).to(device)
    return input_ids, instances