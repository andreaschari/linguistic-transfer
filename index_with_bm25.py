# Uses PyTerrier and pyterrier-xlang to build a BM25 index for a dataset
import argparse
import pyterrier as pt
import logging
from pyterrier_xlang.preprocess import anserini_tokenizer
from pyterrier_pisa import PisaIndex

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

argparser = argparse.ArgumentParser()
argparser.add_argument("--language", type=str, help="language of the dataset", required=True)
argparser.add_argument("--dataset", type=str, help="dataset to index", required=True)
argparser.add_argument("--index", type=str, help="index path", required=True)
args = argparser.parse_args()

dataset = args.dataset
index = args.index
language = args.language


# Load xlang preprocessing pipeline
preproc = anserini_tokenizer(language)
logging.info(f"Preprocessing pipeline loaded for {language}: {preproc}")
# Load dataset
dataset = pt.get_dataset(f"irds:{dataset}")
# Index dataset
idx = PisaIndex(index, stemmer="none", text_field="text", overwrite=True, threads=64)
# (preproc >> idx.toks_indexer(scale=1)).index(dataset.get_corpus_iter()) # for languages that need tokenization (e.g. Chinese)
(preproc >> idx).index(dataset.get_corpus_iter())
logging.info(f"Indexing complete for {dataset}")
