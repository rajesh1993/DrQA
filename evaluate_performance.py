import json
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='data/MovieQA_benchmark/data/qa.json',
#                     help='MovieQA-like dataset to evaluate on')
parser.add_argument('--preds', type=str, default='qa-multitask-specific',
                    help='Predictions filename prefix for json format file')
parser.add_argument('--only-val', action='store_true',
                    help='Do not include validation qns in evaluation')
parser.add_argument('--only-train', action='store_true',
                    help='Do not include training qns in evaluation')
args = parser.parse_args()

if args.only_train and args.only_val:
    print('Need at least one dataset for evaluation. Including both datasets again.')
    args.only_train = False
    args.only_val = False

filenames = [args.preds+'.preds']#, args.preds+'-3200.preds', args.preds+'-6400.preds', args.preds+'-9600.preds']

correct = 0
total = 0


for f_name in filenames:
    predictions = json.load(open(f_name, 'r'))
    for qid, infodict in predictions.items():
    # for i, qid, correct_index, selected_index in predictions.loc[:, ['qid', 'correct_index', 'selected_index']].itertuples():
        if ('test' in qid) or (('val' in qid) and args.only_train) or (('train' in qid) and args.only_val):
            continue
        selected_index = int(infodict['selected_index'])
        correct_index = int(infodict['correct_index'])
        if int(selected_index) == int(correct_index):
            correct += 1
        total += 1

print('Accuracy = {}'.format(correct*100/total))
