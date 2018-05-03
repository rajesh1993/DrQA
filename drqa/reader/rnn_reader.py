#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the RNN based DrQA reader."""

import torch
import torch.nn as nn
from . import layers


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class RnnDocReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(RnnDocReader, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        # Projection for attention weighted question
        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)

        # Projection for attention weighted candidate
        if args.use_qemb:
            self.cemb_match = layers.SeqAttnMatch(args.embedding_dim)

        # Input size to RNN: word emb + question emb + manual features
        # # # We want to add candidate emb to RNN.
        doc_input_size = args.embedding_dim + args.num_features
        if args.use_qemb:
            doc_input_size += args.embedding_dim

        if args.use_qemb:
            doc_input_size += args.embedding_dim

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.question_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # RNN candidate encoder
        # self.candidate_rnn = layers.StackedBRNN(
        #     input_size=args.embedding_dim,
        #     hidden_size=args.hidden_size,
        #     num_layers=args.question_layers,
        #     dropout_rate=args.dropout_rnn,
        #     dropout_output=args.dropout_rnn_output,
        #     concat_layers=args.concat_rnn_layers,
        #     rnn_type=self.RNN_TYPES[args.rnn_type],
        #     padding=args.rnn_padding,
        # )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        candidate_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers
            candidate_hidden_size *= args.question_layers

        # Question merging
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Candidate merging
        # if args.question_merge not in ['avg', 'self_attn']:
        #     raise NotImplementedError('candidate merge_mode = %s' % args.merge_mode)
        # if args.question_merge == 'self_attn':
        #     self.cand_self_attn = layers.LinearSeqAttn(candidate_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )

        # Bilinear attention for span start/end based on candidate instead of question
        # self.cand_start_attn = layers.BilinearSeqAttn(
        #     doc_hidden_size,
        #     candidate_hidden_size,
        #     normalize=normalize,
        # )
        # self.cand_end_attn = layers.BilinearSeqAttn(
        #     doc_hidden_size,
        #     candidate_hidden_size,
        #     normalize=normalize,
        # )

    def forward(self, x1, x1_f, x1_mask, x2, x2_mask, x3, x3_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        x3 = candidate word indices            [batch * len_c]
        x3_mask = candidate padding mask       [batch * len_c]
        """
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)
        x3_emb = self.embedding(x3)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x3_emb = nn.functional.dropout(x3_emb, p=self.args.dropout_emb,
                                           training=self.training)

        # Form document encoding inputs
        drnn_input = [x1_emb]

        # Add attention-weighted question representation
        if self.args.use_qemb:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input.append(x2_weighted_emb)

        if self.args.use_qemb:
            x3_weighted_emb = self.cemb_match(x1_emb, x3_emb, x3_mask)
            drnn_input.append(x3_weighted_emb)

        # Add manual features
        if self.args.num_features > 0:
            drnn_input.append(x1_f)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2), x1_mask)

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        # Encode candidate with RNN + merge hiddens
        # candidate_hiddens = self.candidate_rnn(x3_emb, x3_mask)
        # if self.args.question_merge == 'avg':
        #     c_merge_weights = layers.uniform_weights(candidate_hiddens, x3_mask)
        # elif self.args.question_merge == 'self_attn':
        #     c_merge_weights = self.cand_self_attn(candidate_hiddens, x3_mask)
        # candidate_hidden = layers.weighted_avg(candidate_hiddens, c_merge_weights)

        # Predict start and end positions
        start_scores_qstn = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        end_scores_qstn = self.end_attn(doc_hiddens, question_hidden, x1_mask)

        start_scores = start_scores_qstn
        end_scores = end_scores_qstn

        # print('\nShape of x1:')
        # print(x1.size())
        #
        # print('\nStart Score:')
        # print(start_scores)
        #
        # print('\nEnd Score:')
        # print(end_scores)
        # print('')

        # start_scores_cand = self.cand_start_attn(doc_hiddens, candidate_hidden, x1_mask)
        # end_scores_cand = self.cand_end_attn(doc_hiddens, candidate_hidden, x1_mask)

        return start_scores, end_scores
