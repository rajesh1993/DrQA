import pandas
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data/MovieQA_benchmark/data/qa.json',
                    help='MovieQA-like dataset to evaluate on')
parser.add_argument('--predictions', type=str, default='qa-multitask.preds',
                    help='Predictions file in json format')
parser.add_argument('--only-val', action='store_true',
                    help='Do not include validation qns in evaluation')
parser.add_argument('--only-train', action='store_true',
                    help='Do not include training qns in evaluation')
args = parser.parse_args()

if args.only_train and args.only_val:
    print('Need at least one dataset for evaluation. Including both datasets again.')
    args.only_train = False
    args.only_val = False

predictions = pandas.read_json(open(args.predictions, 'r'))
qa_dataset = pandas.read_json(open(args.dataset, 'r'))

correct = 0
total = 0

for i, qid, correct_index in qa_dataset.loc[:, ['qid', 'correct_index']].itertuples():
    if ('test' in qid) or (('val' in qid) and args.only_train) or (('train' in qid) and args.only_val):
        continue
    if int(predictions[qid]['index']) == int(correct_index):
        correct += 1
    total += 1

print('Accuracy = {}'.format(correct*100/total))
