import argparse
import json
import os
import sys
import time

from drqa.reader import utils, vector, config, data

from multiprocessing import Pool
from multiprocessing.util import Finalize
from functools import partial
from drqa import tokenizers


def init(tokenizer_class, options):
    global TOK
    TOK = tokenizer_class(**options)
    Finalize(TOK, TOK.shutdown, exitpriority=100)

def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        'words': tokens.words(),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(),
    }
    return output


def load_data(path):

    # Load movie data from the MovieQA dataset
    mv_path = os.path.join(path, 'movies.json')
    mv_file = open(mv_path, 'r')
    mv_data = json.load(mv_file)
    mv_dict = {}
    for movie in mv_data:
        plot = os.path.join('./data/datasets/MovieQA_benchmark', movie['text']['plot'])
        story = open(plot, 'r')
        mv_dict[movie['imdb_key']] = {'plot': story.read()}

    # Load the question data from the dataset
    qa_path = os.path.join(path, 'qa.json')
    qa_file = open(qa_path, 'r')
    qa_data = json.load(qa_file)
    qa_dict = {}
    for qa in qa_data:
        if qa['imdb_key'] in qa_dict:
            qa_dict[qa['imdb_key']].append(qa)
        else:
            qa_dict[qa['imdb_key']] = [qa]

    # Load the train/dev/test splits
    splits_path = os.path.join(path, 'splits.json')
    splits_file = open(splits_path, 'r')
    splits_data = json.load(splits_file)

    # Load the output in the below format
    output = {'qids': [], 'questions': [], 'answers': [],
              'contexts': [], 'qid2cid': []}
    for movie in mv_dict:
        if movie in splits_data['train']:
            output['contexts'].append(mv_dict[movie]['plot'])
            for qa in qa_dict[movie]:
                output['questions'].append(qa['question'])
                output['qids'].append(qa['qid'])
                output['qid2cid'].append(len(output['contexts']) - 1)
                if 'answers' in qa:
                    output['answers'].append(qa['answers'])
    return output

# def find_answer(offsets, begin_offset, end_offset):
#     """Match token offsets with the char begin/end offsets of the answer."""
#     start = [i for i, tok in enumerate(offsets) if tok[0] == begin_offset]
#     end = [i for i, tok in enumerate(offsets) if tok[1] == end_offset]
#     assert(len(start) <= 1)
#     assert(len(end) <= 1)
#     if len(start) == 1 and len(end) == 1:
#         return start[0], end[0]

def process_dataset(data, tokenizer, workers=None):
    """Iterate processing (tokenize, parse, etc) dataset multithreaded."""
    tokenizer_class = tokenizers.get_class(tokenizer)
    make_pool = partial(Pool, workers, initializer=init)
    workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}}))
    q_tokens = workers.map(tokenize, data['questions'])
    workers.close()
    workers.join()

    workers = make_pool(
        initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'}})
    )
    c_tokens = workers.map(tokenize, data['contexts'])
    workers.close()
    workers.join()
    ans_tokens = []
    for idx in range(len(data['answers'])):
        workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}}))
        curr_ans = workers.map(tokenize, data['answers'][idx])
        workers.close()
        workers.join()
        ans_tokens.append(curr_ans)

    for idx in range(len(data['qids'])):
        question = q_tokens[idx]['words']
        qlemma = q_tokens[idx]['lemma']
        document = c_tokens[data['qid2cid'][idx]]['words']
        offsets = c_tokens[data['qid2cid'][idx]]['offsets']
        lemma = c_tokens[data['qid2cid'][idx]]['lemma']
        pos = c_tokens[data['qid2cid'][idx]]['pos']
        ner = c_tokens[data['qid2cid'][idx]]['ner']
        # ans_tokens = []
        # print(offsets)
        # if len(data['answers']) > 0:
        #     for ans in data['answers'][idx]:
                # found = find_answer(offsets,
                #                     ans['answer_start'],
                #                     ans['answer_start'] + len(ans['text']))
                # if found:
                #     ans_tokens.append(found)
        yield {
            'id': data['qids'][idx],
            'question': question,
            'document': document,
            'offsets': offsets,
            'answers': ans_tokens,
            'qlemma': qlemma,
            'lemma': lemma,
            'pos': pos,
            'ner': ner,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to MovieQA data directory')
    parser.add_argument('out_dir', type=str, help='Path to output file dir')
    parser.add_argument('--split', type=str, help='Filename for train/dev split',
                        default='movieQA')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--tokenizer', type=str, default='corenlp')
    args = parser.parse_args()

    t0 = time.time()

    # data_path = os.path.join(os.getcwd(), 'data/datasets/MovieQA_benchmark/data')
    print('Loading dataset from %s' %args.data_dir , file=sys.stderr)
    dataset = load_data(args.data_dir)
    out_file = os.path.join(
        args.out_dir, '%s-processed-%s.txt' % (args.split, args.tokenizer)
    )
    print('Will write to file %s' % out_file, file=sys.stderr)
    with open(out_file, 'w') as f:
        for ex in process_dataset(dataset, args.tokenizer, args.workers):
            f.write(json.dumps(ex) + '\n')
    print('Total time: %.4f (s)' % (time.time() - t0))


