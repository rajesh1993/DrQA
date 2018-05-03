#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Functions for putting examples into torch format."""

from collections import Counter
import torch


def vectorize(ex, model, single_answer=False):
    """Torchify a single example."""
    args = model.args
    word_dict = model.word_dict
    feature_dict = model.feature_dict

    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])
    # candidates = [torch.LongTensor([word_dict[w] for w in entry]) for entry in ex['candidates']]
    candidate = torch.LongTensor([word_dict[w] for w in ex['candidate']])
    # c_num = len(ex['candidates'])

    # Create extra features vector
    if len(feature_dict) > 0:
        features = torch.zeros(len(ex['document']), len(feature_dict))
    else:
        features = None

    # f_{exact_match}
    if args.use_in_question:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0
            if q_lemma and ex['lemma'][i] in q_lemma:
                features[i][feature_dict['in_question_lemma']] = 1.0

    if args.use_in_candidate:
        c_words_cased = {w for w in ex['candidate']}
        c_words_uncased = {w.lower() for w in ex['candidate']}
        c_lemma = {w for w in ex['clemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in c_words_cased:
                features[i][feature_dict['in_candidate']] = 1.0
            if ex['document'][i].lower() in c_words_uncased:
                features[i][feature_dict['in_candidate_uncased']] = 1.0
            if c_lemma and ex['lemma'][i] in c_lemma:
                features[i][feature_dict['in_candidate_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    # Maybe return without target
    if 'answers' not in ex:
        return document, features, question, candidate, ex['id']

    # ...or with target(s) (might still be empty if answers is empty)
    if single_answer:
        assert(len(ex['answers']) > 0)
        start = torch.LongTensor(1).fill_(ex['answers'][0][0])
        end = torch.LongTensor(1).fill_(ex['answers'][0][1])
        # candidate_labels = [torch.LongTensor(1).fill_(ex['clabel'][k]) for k in range(c_num)]
    else:
        start = [a[0] for a in ex['answers']]
        end = [a[1] for a in ex['answers']]
        # candidate_labels = [a for a in ex['clabel']]

    # Candidate Label is 0 or 1 based on what candidate is passed in this qa entry
    # candidate_labels = ex['clabel']

    return document, features, question, candidate, start, end, ex['id']


def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    NUM_INPUTS = 4
    NUM_TARGETS = 2
    NUM_EXTRA = 1

    # EXTRA
    ids = [ex[-1] for ex in batch]

    # INPUTS
    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]
    candidates = [ex[3] for ex in batch]
    # c_num = len(candidates[0])

    # Batch documents and features (5 copies, but candidate number is first index for x1, not batch_idx)
    max_length = max([d.size(0) for d in docs])
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    if features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(docs), max_length, features[0].size(1))
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        if x1_f is not None:
            x1_f[i, :d.size(0)].copy_(features[i])
    # for k in range(c_num):
    #     x1_k = torch.LongTensor(len(docs), max_length).zero_()
    #     x1_mask_k = torch.ByteTensor(len(docs), max_length).fill_(1)
    #     if features[0] is None:
    #         x1_f_k = None
    #     else:
    #         x1_f_k = torch.zeros(len(docs), max_length, features[0][k].size(1))
    #     for i, d in enumerate(docs):
    #         x1_k[i, :d.size(0)].copy_(d)
    #         x1_mask_k[i, :d.size(0)].fill_(0)
    #         if x1_f_k is not None:
    #             x1_f_k[i, :d.size(0)].copy_(features[i][k])
    #     x1.append(x1_k)
    #     x1_mask.append(x1_mask_k)
    #     x1_f.append(x1_f_k)

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)

    # Batch candidates (5 copies, and candidate number is first index, not batch_idx)
    max_length = max([c.size(0) for c in candidates])
    x3 = torch.LongTensor(len(candidates), max_length).zero_()
    x3_mask = torch.ByteTensor(len(candidates), max_length).fill_(1)
    for i, c in enumerate(candidates):
        x3[i, :c.size(0)].copy_(c)
        x3_mask[i, :c.size(0)].fill_(0)
    # for k in range(c_num):
    #     x3_k = torch.LongTensor(len(candidates), max_length).zero_()
    #     x3_mask_k = torch.ByteTensor(len(candidates), max_length).fill_(1)
    #     for i, c in enumerate(candidates):
    #         x3_k[i, :c[k].size(0)].copy_(c[k])
    #         x3_mask_k[i, :c[k].size(0)].fill_(0)
    #     x3.append(x3_k)
    #     x3_mask.append(x3_mask_k)

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, x1_f, x1_mask, x2, x2_mask, x3, x3_mask, ids

    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        # ...Otherwise add targets
        if torch.is_tensor(batch[0][3]):
            y_s = torch.cat([ex[4] for ex in batch])
            y_e = torch.cat([ex[5] for ex in batch])
            # c_l = torch.cat([ex[6] for ex in batch])
            # for u in range(c_num):
            #     c_l_k = torch.cat([ex[6][u] for ex in batch])
            #     c_l.append(c_l_k)
        else:
            y_s = [ex[4] for ex in batch]
            y_e = [ex[5] for ex in batch]
            # c_l = [ex[6] for ex in batch]
            # for u in range(c_num):
            #     c_l_k = [ex[6][u] for ex in batch]
            #     c_l.append(c_l_k)
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return x1, x1_f, x1_mask, x2, x2_mask, x3, x3_mask, y_s, y_e, ids
