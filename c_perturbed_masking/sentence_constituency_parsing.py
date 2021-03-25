# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2021-03-24
Description :
auther : wcy
"""
# import modules
import os
from transformers import BertModel, BertTokenizer
import torch
import pickle
import argparse
from tqdm import tqdm
import numpy as np
from c_perturbed_masking.constituency.subword_script import match_tokenized_to_untokenized
from c_perturbed_masking.constituency.data_ptb import Corpus

__all__ = []


# define function
def get_all_subword_id(mapping, idx):
    current_id = mapping[idx]
    id_for_all_subwords = [tmp_id for tmp_id, v in enumerate(mapping) if v == current_id]
    return id_for_all_subwords


def get_con_matrix(corpus, layers, metric, out_file, model, tokenizer, use_cuda=False):
    # corpus = Corpus(args.dataset, args.data_split)

    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    model.eval()

    # LAYER = int(args.layers)
    LAYER = layers
    LAYER += 1  # also consider embedding layer
    out = [[] for i in range(LAYER)]
    for sents, tree2list, nltk_tree in tqdm(zip(corpus.sens, corpus.trees, corpus.nltk_trees), total=len(corpus.sens)):

        # sents = ['rudolph', 'agnew', 'N', 'years', 'old', 'and', 'former', 'chairman', 'of', 'consolidated', 'gold', 'fields', 'plc', 'was', 'named', 'a', 'nonexecutive', 'director', 'of', 'this', 'british', 'industrial', 'conglomerate']
        # tree2list = [[['Rudolph', 'Agnew'], [[['55', 'years'], 'old'], 'and', [['former', 'chairman'], ['of', ['Consolidated', 'Gold', 'Fields', 'PLC']]]]], ['was', ['named', [['a', 'nonexecutive', 'director'], ['of', ['this', 'British', 'industrial', 'conglomerate']]]]]]
        # nltk_tree = Tree('S', [Tree('NP-SBJ-1', [Tree('NP', [Tree('NNP', ['Rudolph']), Tree('NNP', ['Agnew'])]), Tree(',', [',']), Tree('UCP', [Tree('ADJP', [Tree('NP', [Tree('CD', ['55']), Tree('NNS', ['years'])]), Tree('JJ', ['old'])]), Tree('CC', ['and']), Tree('NP', [Tree('NP', [Tree('JJ', ['former']), Tree('NN', ['chairman'])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('NNP', ['Consolidated']), Tree('NNP', ['Gold']), Tree('NNP', ['Fields']), Tree('NNP', ['PLC'])])])])]), Tree(',', [','])]), Tree('VP', [Tree('VBD', ['was']), Tree('VP', [Tree('VBN', ['named']), Tree('S', [Tree('NP-SBJ', [Tree('-NONE-', ['*-1'])]), Tree('NP-PRD', [Tree('NP', [Tree('DT', ['a']), Tree('JJ', ['nonexecutive']), Tree('NN', ['director'])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('DT', ['this']), Tree('JJ', ['British']), Tree('JJ', ['industrial']), Tree('NN', ['conglomerate'])])])])])])]), Tree('.', ['.'])])
        """
            (S
              (NP-SBJ-1
                (NP (NNP Rudolph) (NNP Agnew))
                (, ,)
                (UCP
                  (ADJP (NP (CD 55) (NNS years)) (JJ old))
                  (CC and)
                  (NP
                    (NP (JJ former) (NN chairman))
                    (PP
                      (IN of)
                      (NP (NNP Consolidated) (NNP Gold) (NNP Fields) (NNP PLC)))))
                (, ,))
              (VP
                (VBD was)
                (VP
                  (VBN named)
                  (S
                    (NP-SBJ (-NONE- *-1))
                    (NP-PRD
                      (NP (DT a) (JJ nonexecutive) (NN director))
                      (PP
                        (IN of)
                        (NP
                          (DT this)
                          (JJ British)
                          (JJ industrial)
                          (NN conglomerate)))))))
              (. .))
        """


        sentence = ' '.join(sents)
        tokenized_text = tokenizer.tokenize(sentence)
        tokenized_text.insert(0, '[CLS]')
        tokenized_text.append('[SEP]')
        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        mapping = match_tokenized_to_untokenized(tokenized_text, sents)

        # 1. Generate mask indices
        all_layers_matrix_as_list = [[] for i in range(LAYER)]
        for i in range(0, len(tokenized_text)):
            id_for_all_i_tokens = get_all_subword_id(mapping, i)
            tmp_indexed_tokens = list(indexed_tokens)
            for tmp_id in id_for_all_i_tokens:
                if mapping[tmp_id] != -1:  # both CLS and SEP use -1 as id e.g., [-1, 0, 1, 2, ..., -1]
                    tmp_indexed_tokens[tmp_id] = mask_id
            one_batch = [list(tmp_indexed_tokens) for _ in range(0, len(tokenized_text))]
            for j in range(0, len(tokenized_text)):
                id_for_all_j_tokens = get_all_subword_id(mapping, j)
                for tmp_id in id_for_all_j_tokens:
                    if mapping[tmp_id] != -1:
                        one_batch[j][tmp_id] = mask_id

            # 2. Convert one batch to PyTorch tensors
            tokens_tensor = torch.tensor(one_batch)
            segments_tensor = torch.tensor([[0 for _ in one_sent] for one_sent in one_batch])
            if use_cuda:
                tokens_tensor = tokens_tensor.to('cuda')
                segments_tensor = segments_tensor.to('cuda')
                model.to('cuda')

            # 3. get all hidden states for one batch
            with torch.no_grad():
                model_outputs = model(tokens_tensor, segments_tensor)
                all_layers = model_outputs[-1]  # 12 layers + embedding layer

            # 4. get hidden states for word_i in one batch
            for k, layer in enumerate(all_layers):
                if use_cuda:
                    hidden_states_for_token_i = layer[:, i, :].cpu().numpy()
                else:
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

    for k, one_layer_out in enumerate(out):
        k_output = "{}_{}.pkl".format(out_file, str(k))
        with open(k_output, 'wb') as wf:
            pickle.dump(out[k], wf)
            wf.close()


if __name__ == '__main__':

    pretrained_model_path = "/Users/wcy/2_myself_learn/nlp/howto_build_base_vocab/model/bert-base-indonesian-522M"

    # parser = argparse.ArgumentParser()
    # # Model args
    # parser.add_argument("--model_type", default='bert', type=str)
    # parser.add_argument('--layers', default='12')
    #
    # # Data args
    # parser.add_argument('--data_split', default='WSJ23')
    # parser.add_argument('--dataset', default='constituency/data/WSJ/')
    # parser.add_argument('--output_dir', default='./results/')
    #
    # # Matrix args
    # parser.add_argument('--metric', default='dist')
    # parser.add_argument('--probe', default='constituency', help="dependency, constituency, discourse")
    #
    # # Cuda
    # parser.add_argument('--cuda', action='store_true')
    #
    # args = parser.parse_args()
    #
    # args.output_dir = args.output_dir + args.probe + '/'
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    # data_split = args.dataset.split('/')[-1].split('.')[0]
    # args.output_file = args.output_dir + '/{}-{}-{}-{}.pkl'
    #
    # print(args)


    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path="model_path_dir", output_hidden_states=True)
    model = BertModel.from_pretrained(pretrained_model_name_or_path="model_path_dir", do_lower_case=True)
    # get_con_matrix(args, model, tokenizer)
    corpus = []
    get_con_matrix(corpus=corpus, layers=12, metric="constituency", out_file="temp",
                   model=model, tokenizer=tokenizer, use_cuda=False)
    # with open('./results/WSJ23/bert-base-uncased-False-dist-12.pkl', 'rb') as f:
    #     results = pickle.load(f)