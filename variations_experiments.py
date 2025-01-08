import argparse
import pyterrier as pt
import pandas as pd
import os
import logging
from pyterrier_xlang.preprocess import anserini_tokenizer
from pyterrier_dr import BGEM3, FlexIndex
from pyterrier_pisa import PisaIndex
from pyterrier_t5 import mT5ReRanker
from ir_measures import R, MRR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

argparser = argparse.ArgumentParser()
argparser.add_argument("--lang", type=str, help="language of queries (same as index)", required=True)
argparser.add_argument("--index", type=str, help="index path", required=True)
argparser.add_argument("--model", type=str, help="first stage retrieval model", required=True, choices=["bm25", "bgem3", "bge-m3-lt-cafr", "bge-m3-lt-afdt"])
argparser.add_argument("--dataset", type=str, help="dataset to use (irds format)", default="mmarco/v2")
argparser.add_argument("--rerank", action='store_true', help="whether to rerank the results")
argparser.add_argument("--rerank_model", type=str, help="reranking model", default="mt5", choices=["mt5", "mt5-base-lt-afdt", "mt5-base-lt-cafr"])
argparser.add_argument("--evaluate", action='store_true', help="whether to evaluate the results")

args = argparser.parse_args()

lang = args.lang
index_path = args.index
first_stage_model = args.model
dataset_name = args.dataset
rerank = args.rerank
rerank_model = args.rerank_model
evaluate = args.evaluate

TRANSLATIONS_DIR = "/root/data/google_translations" # change this to the directory where the translations are stored
RETRIEVAL_RESULTS_DIR = "/root/data/retrieval_results" # change this to the directory where the retrieval results are stored

# Check if the index directory exists
if not os.path.isdir(index_path):
    raise FileNotFoundError(f"The index: {index_path} does not exist")

# Language Pairs
lang_pairs = {
    "dt": "af",
    "zh": "yue",
    "id": "ms",
    "it": "scn",
    "fr": "ca", 
    # "fr": "oc",# Uncomment this and comment the previous line if you want to use Occitan instead of Catalan
    "en": "gd",
}

if lang not in lang_pairs:
    raise ValueError(f"Invalid language {lang}")

# Load Queries for both languages
lang2 = lang_pairs[lang]
# Load Translated Queries
lang2_topics = pd.read_csv(f"{TRANSLATIONS_DIR}/{lang2}.tsv", sep="\t", header=None, names=["qid", "query"])
# cast all queries to string
lang2_topics["query"] = lang2_topics["query"].astype(str)
lang2_topics["qid"] = lang2_topics["qid"].astype(str)
logging.info(f"Loaded {len(lang2_topics)} queries for {lang2}")

if lang == "en":
    dataset = pt.get_dataset('irds:msmarco-passage/dev/small')
else:
    dataset = pt.get_dataset(f"irds:{dataset_name}/{lang}/dev/small")
topics = dataset.get_topics(tokenise_query=False)
logging.info(f"Loaded {len(topics)} queries for {lang}")

if first_stage_model == "bm25":
    # Load xlang preprocessors
    # in case the preprocessing tools use a different language code
    preprocessing_langs = {
        "dt": "nl",
        "zh": "zh",
        "id": "id",
        "it": "it",
        "fr": "fr",
        "en": "en"
    }
    preproc = anserini_tokenizer(preprocessing_langs[lang]) # might need to refactor to allow more tokenizers in the future
    logging.info(f"Loaded preprocessor for {lang} and {lang2}")

    # Load Pisa Index
    idx = PisaIndex(index_path)
    logging.info(f"Loaded index {index_path}")

    # # Retrieval Pipeline
    pipeline = preproc >> idx.bm25(verbose=True)
elif first_stage_model in ["bge-m3", "bge-m3-lt-cafr", "bge-m3-lt-afdt"]:
    if first_stage_model == "bge-m3":
        factory = BGEM3(batch_size=32, max_length=1024, verbose=True)
    else:
        factory = BGEM3(batch_size=32, max_length=1024, verbose=True, model=f"andreaschari/{first_stage_model}")
    encoder = factory.query_encoder()

    # Load Pisa Index
    idx = FlexIndex(index_path)
    logging.info(f"Loaded index {index_path}")

    # # Retrieval Pipeline
    pipeline = encoder >> idx.np_retriever()


# Run First Stage Retrieval
dataset_name = dataset_name.replace("/", "")
# create results folder if it does not exist
os.makedirs(f"{RETRIEVAL_RESULTS_DIR}/{first_stage_model}", exist_ok=True)

def retrieve_and_save_results(lang, topics):
    result_file = f"{RETRIEVAL_RESULTS_DIR}/{first_stage_model}/{first_stage_model}_{dataset_name}_{lang}.res.gz"
    if os.path.isfile(result_file):
        logging.info(f"Results for {first_stage_model} retrieval for {lang} already exists")
        res = pt.io.read_results(result_file)
        res = res.merge(topics, on="qid")
        logging.info("Loaded results from disk")
    else:
        logging.info(f"Running {first_stage_model} retrieval for {lang} queries")
        res = pipeline(topics)
        pt.io.write_results(res, result_file)
        logging.info("Saved results to disk")
    return res

res1 = retrieve_and_save_results(lang, topics)
res2 = retrieve_and_save_results(lang2, lang2_topics)

if rerank:
    # Load Model
    if rerank_model == "mt5":
        reranker = mT5ReRanker(verbose=True)
    elif rerank_model == "mt5-base-lt-cafr":
        reranker = mT5ReRanker(model='andreaschari/mt5-base-lt-cafr', verbose=True)
    elif rerank_model == "mt5-base-lt-afdt":
        reranker = mT5ReRanker(model='andreaschari/mt5-base-lt-afdt', verbose=True)
    else:
        raise ValueError(f"Invalid reranking model {rerank_model}")

    pipeline = pt.text.get_text(dataset, "text") >> reranker
    # create results folder if it does not exist
    os.makedirs(f"{RETRIEVAL_RESULTS_DIR}/{rerank_model}", exist_ok=True)

    def rerank_and_save_results(lang, res):
        result_file = f"{RETRIEVAL_RESULTS_DIR}/{rerank_model}/{rerank_model}_{first_stage_model}_{dataset_name}_{lang}.res.gz"
        if os.path.isfile(result_file):
            logging.info(f"Re-ranked results for {first_stage_model} retrieval for {lang} already exists")
            return pt.io.read_results(result_file)
        else:
            logging.info(f"Re-ranking {first_stage_model} results for {lang}")
            res = pipeline(res)
            pt.io.write_results(res, result_file)
            logging.info("Saved re-ranked results to disk")
            return res

    res1 = rerank_and_save_results(lang, res1)
    res2 = rerank_and_save_results(lang2, res2)

if evaluate:
    # change res1 and res2 for whatever you want to evaluate
    logging.info("Evaluating results between variaties")
    experiment = pt.Experiment(
        [res1, res2],
        dataset.get_topics(),
        dataset.get_qrels(),
        [MRR@10, R@1000],
        names=[f"{lang}", f"{lang2}"],
        baseline=0,
        correction="b"
    )
    print(experiment)
    logging.info("Done")