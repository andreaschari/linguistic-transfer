# Preprocesses msmarco data to train BGE-M3 model

import random
import json, gzip, pickle
import tqdm
import math
from os.path import exists, join, basename
from sentence_transformers import util


NEGS_PER_QUERY=1
TOTAL_STEPS=200000
BATCH_SIZE=8
NUM_SAMPLING_ROUNDS= math.ceil((TOTAL_STEPS * BATCH_SIZE) / 502939)
examples = 502939 * NUM_SAMPLING_ROUNDS
print(f"#> Number of sampling rounds: {NUM_SAMPLING_ROUNDS}")
CE_MARGIN=3.0

# Sampling code based on ColBERT-XM codebase in https://github.com/ant-louis/xm-retrievers

def download_if_not_exists(data_filepath, file_url: str):
    save_path = join(data_filepath, basename(file_url))
    if not exists(save_path):
        util.http_get(file_url, save_path)
    return save_path

# PATHS
# change these paths to the appropriate paths on your system
# INPUT
# path to the collection file
collection_filepath = "data/mmarco.v2.fr.docs.tsv"
# path to the training queries
training_queries_filepath = "mmarco.v2.cafr.train.judged.tsv"
data_filepath = "data" # path to save any downloaded data
# OUTPUT
# path to save the training tuples
training_tuples_filepath = f'data/triples.mmarco.v2.cafr.train.HN.{NEGS_PER_QUERY+1}way.{examples/1e6:.1f}M.jsonl'

# Load training queries
training_queries = {}
with open(training_queries_filepath, 'r') as fIn:
    for line in fIn:
        qid, query = line.strip().split('\t')
        training_queries[qid] = query
print(f"#> Loaded training queries: {len(training_queries)}")

# Load collection
collection = {}
with open(collection_filepath, 'r') as fIn:
    for line in fIn:
        pid, passage = line.strip().split('\t')
        collection[pid] = passage
print(f"#> Loaded collection: {len(collection)}")

# Load CE scores for query-passage pairs: ce_scores[qid][pid] -> score.
url = 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz'
ce_scores_file = download_if_not_exists(data_filepath, url)
with gzip.open(ce_scores_file, 'rb') as fIn:
    ce_scores = pickle.load(fIn)

num_training_examples = 0
# Load hard negatives mined from BM25 and 12 different dense retrievers.
url = 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz'
hard_negatives_filepath = download_if_not_exists(data_filepath, url)

print("#> Sampling training triples...")
with gzip.open(hard_negatives_filepath, 'rt') as fIn, open(training_tuples_filepath, 'w') as fOut:
    for round_idx in tqdm.tqdm(range(NUM_SAMPLING_ROUNDS), desc='Sampling round'):
        fIn.seek(0)
        random.seed(42 + round_idx)
        for line in tqdm.tqdm(fIn):
            # Load the training sample: {"qid": ..., "pos": [...], "neg": {"bm25": [...], "msmarco-MiniLM-L-6-v3": [...], ...}}
            data = json.loads(line)
            qid = data['qid']
            pos_pids = data['pos']
            if len(pos_pids) == 0:
                continue

            # Set the CE threshold as the minimum positive score minus a margin.
            pos_min_ce_score = min([ce_scores[qid][pid] for pid in pos_pids])
            ce_score_threshold = pos_min_ce_score - CE_MARGIN

            # Sample one positive passage
            sampled_pos_pid = random.choice(pos_pids)

            # Sample N hard negatives and their CE scores.
            neg_pids = []
            neg_systems = list(data['neg'].keys())
            for system_name in neg_systems:
                neg_pids.extend(data['neg'][system_name])

            filtered_neg_pids = [pid for pid in list(set(neg_pids)) if ce_scores[qid][pid] <= ce_score_threshold]
            sampled_neg_pids = random.sample(filtered_neg_pids, min(NEGS_PER_QUERY, len(filtered_neg_pids)))

            if len(sampled_neg_pids) == NEGS_PER_QUERY:
                # returned sample in the format: {"query": str, "pos": List[str], "neg":List[str]}

                sample = {
                    "query": training_queries[str(qid)],
                    "pos": [collection[str(sampled_pos_pid)]],
                    "neg": [collection[str(pid)] for pid in sampled_neg_pids]
                }
                fOut.write(json.dumps(sample, ensure_ascii=False) + '\n')
                num_training_examples += 1

print(f"#> Number of training examples created: {num_training_examples}")
print(f"#> Training triples saved to {training_tuples_filepath}")


