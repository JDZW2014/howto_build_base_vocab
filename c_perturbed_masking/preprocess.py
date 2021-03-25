import os
import argparse
from c_perturbed_masking.utils import ConllUDataset
from transformers import BertModel, BertTokenizer
from c_perturbed_masking.dependency import get_dep_matrix
from c_perturbed_masking.constituency import get_con_matrix
from c_perturbed_masking.discourse import get_dis_matrix


# model config
MODEL_CLASSES = {
    'bert': (BertModel, BertTokenizer, 'bert-base-uncased'),
}


# define function
def processing(model_type, layers, data_split, dataset, output_dir, metric, cuda, probe, result_type):

    model_class, tokenizer_class, pretrained_weights = MODEL_CLASSES[model_type]

    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)

    if result_type == 'dependency':
        dataset = ConllUDataset(args.dataset)
        get_dep_matrix(args, model, tokenizer, dataset)
    elif result_type == 'constituency':
        get_con_matrix(args, model, tokenizer)
    elif result_type == 'discourse':
        get_dis_matrix(args, model, tokenizer)


if __name__ == '__main__':
    MODEL_CLASSES = {
        'bert': (BertModel, BertTokenizer, 'bert-base-uncased'),
    }
    parser = argparse.ArgumentParser()

    # Model args
    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument('--layers', default='12')

    # Data args
    parser.add_argument('--data_split', default='SciDTB', choices=['PUD', 'WSJ10', 'WSJ23', 'WSJ10', 'SciDTB'])
    parser.add_argument('--dataset', default='./discourse/SciDTB/test/gold/', required=True)
    parser.add_argument('--output_dir', default='./results/')

    parser.add_argument('--metric', default='dist', help='metrics for impact calculation, support [dist, cos] so far')
    parser.add_argument('--cuda', action='store_true', help='invoke to use gpu')

    parser.add_argument('--probe', required=True, choices=['dependency', 'constituency', 'discourse'])

    args = parser.parse_args()

    model_class, tokenizer_class, pretrained_weights = MODEL_CLASSES[args.model_type]

    args.output_dir = args.output_dir + args.probe + '/'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_file = args.output_dir + '/{}-{}-{}-{}.pkl'

    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)

    print(args)

    if args.probe == 'dependency':
        dataset = ConllUDataset(args.dataset)
        get_dep_matrix(args, model, tokenizer, dataset)
    elif args.probe == 'constituency':
        get_con_matrix(args, model, tokenizer)
    elif args.probe == 'discourse':
        get_dis_matrix(args, model, tokenizer)



