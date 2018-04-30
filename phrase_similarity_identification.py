import spacy
import wmd
import time

from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES, LOOKUP
lemmatizer = Lemmatizer(index=LEMMA_INDEX, exceptions=LEMMA_EXC, rules=LEMMA_RULES, lookup=LOOKUP)

t0 = time.time()
# nlp = spacy.load('en_vectors_web_lg')
nlp = spacy.load('en_vectors_web_lg', create_pipeline=wmd.WMD.create_spacy_pipeline)
t1 = time.time()
print('Loaded word vectors in {}s'.format(t1-t0))

candidates = [
            "Because he is warned it's not safe.",
            "Because he wants to get some fresh air.",
            "Because he is called away.",
            "Because he has to get some stuff.",
            "Because he senses danger."
        ]
answer = 'Sensing danger'

# answer_embedding = nlp(answer)
# candidate_embeddings = [nlp(c) for c in candidates]
#
# print('Answer: {}'.format(answer))
# for i in range(len(candidates)):
#     print('Candidate {}: \"{}\"\t -\t{}'.format(
#         i, candidates[i], answer_embedding.similarity(candidate_embeddings[i])))

lemmatized_candidates = [' '.join([lemmatizer.lookup(word.strip('.,!-')) for word in c.split()]) for c in candidates]

lemmatized_answer = ' '.join([lemmatizer.lookup(word.strip('.,!-')) for word in answer.split()])
print('\nLemmatized Answer: {}'.format(lemmatized_answer))

lemmatized_answer_embedding = nlp(lemmatized_answer)
lemmatized_candidate_embeddings = [nlp(c) for c in lemmatized_candidates]
for i in range(len(lemmatized_candidates)):
    print('Answer-Candidate {}: \"{}\"\t -\t{}'.format(
        i, lemmatized_candidates[i], lemmatized_answer_embedding.similarity(lemmatized_candidate_embeddings[i])))
    print('Candidate-Answer {}: \"{}\"\t -\t{}'.format(
        i, lemmatized_candidates[i], lemmatized_candidate_embeddings[i].similarity(lemmatized_answer_embedding)))
