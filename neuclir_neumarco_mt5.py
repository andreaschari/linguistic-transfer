import pyterrier as pt
import logging
import argparse
from pyterrier_t5 import mT5ReRanker
from ir_measures import R, nDCG

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

argparser = argparse.ArgumentParser()

argparser.add_argument("--dataset", type=str, help="dataset to use (irds format)", default="neumarco/ru/dev/small")
argparser.add_argument("--first_stage", type=str, help="results path for original bgem3", required=True)
argparser.add_argument("--model", type=str, help="reranking model", default="mt5", choices=["mt5", "mt5-base-lt-afdt", "mt5-base-lt-cafr"])
argparser.add_argument("--out_path", type=str, help="output path for the results", required=True)

args = argparser.parse_args()

dataset_name = args.dataset
model = args.model
out_path = args.out_path
first_stage = args.first_stage 


reranker = mT5ReRanker(verbose=True, model=f"andreaschari/{model}")

logging.info(f"Running mT5 reranker for {dataset_name}")
dataset = pt.get_dataset(dataset_name)
res = pt.io.read_results(first_stage)

# # # neuclir topics processing
if "neuclir" in dataset_name:
    topics = dataset.get_topics()
    topics = topics.rename(columns={'title': 'query'})
    # use only qid and query columns
    topics = topics[['qid', 'query']]
else:
    topics = dataset.get_topics()

res = res.merge(topics, on="qid")

logging.info("Loaded results from disk")

pipeline = pt.text.get_text(dataset, "text") >> reranker
logging.info("Running reranking")
res_reranked = pipeline(res)
logging.info("Reranked results")

dataset_name = dataset_name.replace("/", "")
pt.io.write_results(res_reranked, f"{out_path}/{model}/{model}_bgem3_{dataset_name}.gz")
logging.info("Saved reranked results to disk")

# Run experiment
logging.info("Evaluating mT5 rankings")
experiment = pt.Experiment(
    [res_reranked],
    topics,
    dataset.get_qrels(),
    [R@1000, nDCG@20]
)

print(experiment)
