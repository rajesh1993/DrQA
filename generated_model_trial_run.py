#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to make and save model predictions on an input dataset."""

import os
import time
import torch
import argparse
import logging
import json
import pandas
import numpy as np

from tqdm import tqdm
from drqa.reader import Predictor

import spacy
# import wmd

# from spacy.lemmatizer import Lemmatizer
# from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES, LOOKUP
# lemmatizer = Lemmatizer(index=LEMMA_INDEX, exceptions=LEMMA_EXC, rules=LEMMA_RULES, lookup=LOOKUP)
# nlp = spacy.load('en_vectors_web_lg', create_pipeline=wmd.WMD.create_spacy_pipeline)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data/MovieQA_benchmark/data/qa.json',
                    help='SQuAD-like dataset to evaluate on')
parser.add_argument('--model', type=str, default='models/20180503-52e34880.mdl',
                    help='Path to model to use')
parser.add_argument('--embedding-file', type=str, default=None,
                    help=('Expand dictionary to use all pretrained '
                          'embeddings in this file.'))
parser.add_argument('--out-dir', type=str, default='/home/sai/NLP/Project/DrQA',
                    help=('Directory to write prediction file to '
                          '(<dataset>-<model>.preds)'))
parser.add_argument('--tokenizer', type=str, default='spacy',
                    help=("String option specifying tokenizer type to use "
                          "(e.g. 'corenlp')"))
parser.add_argument('--num-workers', type=int, default=None,
                    help='Number of CPU processes (for tokenizing, etc)')
parser.add_argument('--no-cuda', action='store_true',
                    help='Use CPU only')
parser.add_argument('--gpu', type=int, default=-1,
                    help='Specify GPU device id to use')
parser.add_argument('--batch-size', type=int, default=8,
                    help='Example batching size')
parser.add_argument('--top-n', type=int, default=1,
                    help='Store top N predicted spans per example')
parser.add_argument('--official', action='store_true', default=True,
                    help='Only store single top span instead of top N list')
parser.add_argument('--start-point', type=int, default=0, 
                    help='Question Number in dataset to start with')
args = parser.parse_args()
print('args.official = {}'.format(args.official))
t0 = time.time()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')
# torch.cuda.set_device(-1)

# predictor = Predictor(
#     model=args.model,
#     tokenizer=args.tokenizer,
#     embedding_file=args.embedding_file,
#     num_workers=args.num_workers,
# )
predictor = Predictor(
    model=args.model,
    tokenizer=args.tokenizer,
    embedding_file=args.embedding_file,
    num_workers=args.num_workers,
)

if args.cuda:
    predictor.cuda()
# predictor.cuda()

# ------------------------------------------------------------------------------
# Read in dataset and make predictions.
# ------------------------------------------------------------------------------

f_qa_json = open(args.dataset, 'r')
qn_set_df = pandas.read_json(f_qa_json)
f_qa_json.close()

f_movies_json = open('data/MovieQA_benchmark/data/movies.json', 'r')
mv_set_df = pandas.read_json(f_movies_json)
f_movies_json.close()

results = {}
num_batches = 0

start = args.start_point
print('Total number of qns in dataset: {}'.format(qn_set_df.shape[0]))
print('Total number of estimated batches: {}'.format(qn_set_df.shape[0]/args.batch_size))

for q in range(start, min(start+3200, qn_set_df.shape[0]), args.batch_size):
# for q in range(0, args.batch_size, args.batch_size):
    num_batches += 1
    # if num_batches == 27:
    #     continue
    print('Batch Number: {}'.format(num_batches))
    examples = []
    qids = []
    example_candidates = []
    example_correct_index = []
    for i, qid, question, imdb_key, candidates, correct_index in \
            qn_set_df.loc[q:q+args.batch_size-1, ['qid', 'question', 'imdb_key',
                                                'answers', 'correct_index']].itertuples():
        # print(mv_set_df.loc[mv_set_df['imdb_key'] == imdb_key]['text'].any())
        if 'test' in qid:
            break
        plot_location = mv_set_df.loc[mv_set_df['imdb_key'] == imdb_key]['text'].any()['plot']
        if plot_location is not None:
            plot_file = open(os.path.join(os.getcwd(), 'data/MovieQA_benchmark/'+plot_location), 'r')
            context = plot_file.read(16384)  # Hard-coded. Don't expect plots larger than 16 KB. Increase if needed.
            plot_file.close()
            qids.append(qid)
            example_candidates.append(candidates)
            example_correct_index.append(correct_index)
            examples.append((context, question, candidates))
    # print('Finished creating inputs, Going to start predictions now')
    # print('examples size: {}'.format(len(examples)))
    for i in tqdm(range(0, len(examples), args.batch_size)):
        predictions = predictor.predict_batch(
            examples[i:i + args.batch_size], top_n=args.top_n
        )
        for j in range(len(predictions)):
            # Official eval expects just a qid --> span
            if args.official:
                # results[qids[i + j]] = predictions[j][0][0]
                answer = predictions[j][0][0][0]
                best_candidate = predictions[j][1]
                results[qids[i + j]] = {"span": answer,
                                        "selected_candidate": example_candidates[i + j][int(best_candidate)],
                                        "selected_index": float(best_candidate),
                                        "correct_candidate": example_candidates[i + j][int(example_correct_index[i + j])],
                                        "correct_index": float(example_correct_index[i + j])
                                        }
                # print('Answer: {}'.format(answer))
                # print('Candidate: {}'.format(results[qids[i + j]]))

            # Otherwise we store top N and scores for debugging.
            # else:
                # results[qids[i + j]] = [(p[0], float(p[1])) for p in predictions[j]]
                # ans_preds = [(p[0], float(p[1])) for p in predictions[j]]
                # lem_answers = [' '.join([lemmatizer.lookup(w.strip('.,!-;:()'))
                #                          for w in ans[0].split()]) for ans in ans_preds]
                # candidates = example_candidates[i + j]
                # lem_candidates = [' '.join([lemmatizer.lookup(w.strip('.,!-;:()'))
                #                             for w in c.split()]) for c in candidates]
                # ans_embs = [nlp(ans) for ans in lem_answers]
                # cand_embs = [nlp(c) for c in lem_candidates]
                # best_candidate = [np.argmax([pred_ans_emb.similarity(c) for c in cand_embs])
                #                   for pred_ans_emb in ans_embs]
                # results[qids[i + j]] = [{"span": ans_preds},
                #                         {"candidate": [(candidates[int(best_candidate[k])],
                #                                         float(ans_embs[k].similarity(cand_embs[int(best_candidate[k])])),
                #                                         int(best_candidate[k]))
                #                                        for k in range(len(best_candidate))]}
                #                         ]
                # print('Answers: {}'.format(ans_preds))
                # print('Candidates: {}\n'.format(results[qids[i + j]]))
print('Finished predictions')

model = os.path.splitext(os.path.basename(args.model or 'default'))[0]
basename = os.path.splitext(os.path.basename(args.dataset))[0]
outfile = os.path.join(args.out_dir, basename + '-' + model + '-' + str(args.start_point) + '.preds')
# outfile = 'trial_predictions.preds'
logger.info('Writing results to %s' % outfile)
with open(outfile, 'w') as f:
    if args.official:
        json.dump(results, f, indent='\t')
    else:
        json.dump(sorted(results.items()), f, indent='\t')

logger.info('Total time: %.2f' % (time.time() - t0))
