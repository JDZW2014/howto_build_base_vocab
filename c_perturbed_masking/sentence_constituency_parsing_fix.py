# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2021-03-24
Description :
auther : wcy
"""
# import modules
import torch
import pickle
import nltk
from tqdm import tqdm
import numpy as np
from transformers import BertModel, BertTokenizer
from c_perturbed_masking.constituency.subword_script import match_tokenized_to_untokenized
from c_perturbed_masking.constituency.decoder import mart, right_branching, left_branching
import copy


__all__ = []


# define function
def get_all_subword_id(mapping, idx):
    current_id = mapping[idx]
    id_for_all_subwords = [tmp_id for tmp_id, v in enumerate(mapping) if v == current_id]
    return id_for_all_subwords


def get_con_matrix(sentence_list, layers, metric, out_file, model, tokenizer, use_cuda=False):
    # corpus = Corpus(args.dataset, args.data_split)

    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    model.eval()

    # LAYER = int(args.layers)
    LAYER = layers
    LAYER += 1  # also consider embedding layer
    out = [[] for i in range(LAYER)]

    for sentence in tqdm(sentence_list):
        tree2list, nltk_tree = [], []
        sents = nltk.word_tokenize(sentence)
        tokenized_text = tokenizer.tokenize(sentence)
        tokenized_text.insert(0, '[CLS]')
        tokenized_text.append('[SEP]')
        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        mapping = match_tokenized_to_untokenized(tokenized_text, sents) # [-1, 0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 5, 6, 6, 7, -1]

        all_word_idx = set(mapping)

        all_word_idx.remove(-1)
        all_word_idx = sorted(list(all_word_idx)) # [0, 1, 2, 3, 4, 5, 6, 7]

        masked_indexed_tokens_list = []

        # todo 改成可以输出13层结果的逻辑
        init_matrix = np.zeros((len(all_word_idx), len(all_word_idx)))
        for i, word_idx in enumerate(all_word_idx):

            # 1. Generate mask indices
            masked_indexed_tokens = copy.deepcopy(indexed_tokens)
            for mapping_idx, mapping_value in enumerate(mapping):
                if word_idx == mapping_value:
                    masked_indexed_tokens[mapping_idx] = mask_id

            masked_indexed_tokens_list.append(copy.deepcopy(masked_indexed_tokens))

            for second_word_idx in all_word_idx:
                if second_word_idx == word_idx:
                    continue
                masked_masked_indexed_tokens = copy.deepcopy(masked_indexed_tokens)
                for mapping_idx, mapping_value in enumerate(mapping):
                    if second_word_idx == mapping_value:
                        masked_masked_indexed_tokens[mapping_idx] = mask_id
                masked_indexed_tokens_list.append(masked_masked_indexed_tokens)

            # 2. Convert one batch to PyTorch tensors
            tokens_tensor = torch.tensor(masked_indexed_tokens_list)
            segments_tensor = torch.tensor([[0 for _ in one_sent] for one_sent in masked_indexed_tokens_list])

            if use_cuda:
                tokens_tensor = tokens_tensor.to('cuda')
                segments_tensor = segments_tensor.to('cuda')
                model.to('cuda')

            # 3. get all hidden states for one batch
            with torch.no_grad():
                model_outputs = model(tokens_tensor, segments_tensor)
                all_layers = model_outputs[-1]  # 12 layers + embedding layer

            # 4. get hidden states for word_i in one batch
            # todo 改成可以输出12层结果的逻辑
            layer = all_layers[-1]

            all_layers_matrix_as_list = []
            for k, layer in enumerate(all_layers):
                # layer shape ->  torch.Size([16, 16, 768])
                if use_cuda:
                    hidden_states_for_token_i = layer[:, i, :].cpu().numpy()
                else:
                    # hidden_states_for_token_i 取的是 [batch_size, dim] ?? 为什么不是
                    hidden_states_for_token_i = layer[:, i, :].numpy()
                all_layers_matrix_as_list[k].append(hidden_states_for_token_i)

        for k, one_layer_matrix in enumerate(all_layers_matrix_as_list):
            init_matrix = np.zeros((len(tokenized_text), len(tokenized_text)))
            for i, hidden_states in enumerate(one_layer_matrix):
                base_state = hidden_states[i]
                for j, state in enumerate(hidden_states):
                    if metric == 'dist':
                        init_matrix[i][j] = np.linalg.norm(base_state - state)
                    if metric == 'cos':
                        init_matrix[i][j] = np.dot(base_state, state) / (
                                np.linalg.norm(base_state) * np.linalg.norm(state))
            out[k].append((sents, tokenized_text, init_matrix, tree2list, nltk_tree))

    if out_file:
        for k, one_layer_out in enumerate(out):
            k_output = "{}_{}.pkl".format(out_file, str(k))
            with open(k_output, 'wb') as wf:
                pickle.dump(out[k], wf)
                wf.close()

    return out


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def decoding(decoder, subword, matrix=None, results=None):
    trees = []
    new_results = []
    assert matrix or results
    if matrix:
        with open(matrix, 'rb') as f:
            results = pickle.load(f)

    for (sen, tokenized_text, init_matrix, _, _) in results:
        mapping = match_tokenized_to_untokenized(tokenized_text, sen)
        # merge subwords in one row
        merge_column_matrix = []
        for i, line in enumerate(init_matrix):
            new_row = []
            buf = []
            for j in range(0, len(line) - 1):
                buf.append(line[j])
                if mapping[j] != mapping[j + 1]:
                    new_row.append(buf[0])
                    buf = []
            merge_column_matrix.append(new_row)

        # merge subwords in multi rows
        # transpose the matrix so we can work with row instead of multiple rows
        merge_column_matrix = np.array(merge_column_matrix).transpose()
        merge_column_matrix = merge_column_matrix.tolist()
        final_matrix = []
        for i, line in enumerate(merge_column_matrix):
            new_row = []
            buf = []
            for j in range(0, len(line) - 1):
                buf.append(line[j])
                if mapping[j] != mapping[j + 1]:
                    if subword == 'max':
                        new_row.append(max(buf))
                    elif subword == 'avg':
                        new_row.append((sum(buf) / len(buf)))
                    elif subword == 'first':
                        new_row.append(buf[0])
                    buf = []
            final_matrix.append(new_row)

        # transpose to the original matrix
        final_matrix = np.array(final_matrix).transpose()

        # filter some empty matrix (only one word)
        if final_matrix.shape[0] == 0:
            print(final_matrix.shape)
            continue
        assert final_matrix.shape[0] == final_matrix.shape[1]

        new_results.append((sen, tokenized_text, init_matrix))
        final_matrix = final_matrix[1:, 1:]

        final_matrix = softmax(final_matrix)

        np.fill_diagonal(final_matrix, 0.)

        final_matrix = 1. - final_matrix
        np.fill_diagonal(final_matrix, 0.)

        if decoder == 'mart':
            parse_tree = mart(final_matrix, sen)
            trees.append(parse_tree)

        if decoder == 'right_branching':
            trees.append(right_branching(sen))

        if decoder == 'left_branching':
            trees.append(left_branching(sen))

    return trees, new_results


# define test
def test1():
    pretrained_model_path = "/howto_build_base_vocab/model/bert-base-indonesian-522M"
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_path,
                                              do_lower_case=True)
    model = BertModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_path,
                                      output_hidden_states=True)
    # get_con_matrix(args, model, tokenizer)
    corpus = ["bajuyuli jilbab anak syari gws* series",
              "bajuyuli gamis anak pita peach mint",
              "bajuyuli gamis anak balotelli pita pink hahah hello"]

    corpus = ["bajuyuli gamis anak balotelli pita pink hahah hello"]
    get_con_matrix(sentence_list=corpus, layers=12, metric="constituency", out_file="temp",
                   model=model, tokenizer=tokenizer, use_cuda=False)
    # with open('./results/WSJ23/bert-base-uncased-False-dist-12.pkl', 'rb') as f:
    #     results = pickle.load(f)


def test2():
    trees, results = decoding(matrix="temp_11.pkl", decoder='mart', subword='avg')


if __name__ == '__main__':
    test1()
    # test2()
    pass


