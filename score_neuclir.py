# Scores ColBERT-XM rankings of neuclir using pyterrier and ir_measures
import pyterrier as pt
from ir_measures import R, nDCG
import logging
import pandas as pd
import numpy as np
import os
import argparse

# set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# parse arguments
parser = argparse.ArgumentParser(description='Scores ColBERT-XM rankings of neuclir using pyterrier and ir_measures')
parser.add_argument('--ranking_file', type=str, help='Path to the TSV file with the rankings')
parser.add_argument('--mapping_file', type=str, help='Path to the TSV file with the mapping of the doc_id')

args = parser.parse_args()
ranking_file = args.ranking_file
mapping_file = args.mapping_file


# load the dataset
dataset = pt.get_dataset("irds:neuclir/1/fa/trec-2023")

# load the tsv file with the rankings
logging.info(f"Loading rankings from {ranking_file}")

# load the rankings
rankings = pd.read_csv(ranking_file, sep='\t', header=None, names=['qid', 'docno', 'rank', 'score'])

# rankings have the index of the document collection as "doc_id", we need to map it to the original doc_id
# mapping files are found in the huggingface dataset (Check repo docs)

# load the mapping file
logging.info(f"Loading mapping from {mapping_file}")
mappings = pd.read_csv(mapping_file, sep='\t', header=None, names=['doc_id_old', 'docno'])
# print(mappings.head(2))
logging.info("Processing rankings")
# merge the rankings with the mapping replacing the "doc_id" of the rankings with the original doc_id
rankings = pd.merge(rankings, mappings, on='docno')
# drop "doc_id" and rename "doc_id_old" to "doc_id"
rankings = rankings.drop(columns=['docno'])
rankings = rankings.rename(columns={'doc_id_old': 'docno'})
# reorder the columns: query_id, doc_id, rank, score
rankings = rankings[['qid', 'docno', 'rank', 'score']]
# cast qid to object
rankings['qid'] = rankings['qid'].astype(str)
# cast docno to object
rankings['docno'] = rankings['docno'].astype(str)

# # # neuclir topics processing
topics = dataset.get_topics()
topics = topics.rename(columns={'title': 'query'})
# # # use only qid and query columns
topics = topics[['qid', 'query']]

# Run experiment
logging.info("Evaluating ColBERT-XM rankings")
experiment = pt.Experiment(
    [rankings],
    topics,
    dataset.get_qrels(),
    [R@1000, nDCG@20]
)

print(experiment)
