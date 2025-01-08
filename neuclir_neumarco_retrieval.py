import pyterrier as pt
from ir_measures import R, nDCG, MRR
from pyterrier_dr import FlexIndex, BGEM3
import logging
import argparse
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

argparser = argparse.ArgumentParser()

argparser.add_argument("--dataset", type=str, help="dataset to use (irds format)", default="neumarco/ru/dev/small")
argparser.add_argument("--index", type=str, help="index path for original bgem3", required=True)
argparser.add_argument("--trans_index", type=str, help="index path for fine-tuned bgem3", required=True)
argparser.add_argument("--model", type=str, help="variations of fine-tuned bgem3", required=True, choices=["bge-m3-lt-cafr", "bge-m3-lt-afdt"])
argparser.add_argument("--out_path", type=str, help="output path for the results", required=True)

args = argparser.parse_args()

dataset_name = args.dataset
index_path = args.index
trans_index_path = args.tca_index
model = args.model
out_path = args.out_path

# load the dataset
dataset = pt.get_dataset(dataset_name)
logging.info("Loaded dataset")

# load the index
index = FlexIndex(index_path, verbose=True)
trans_index = FlexIndex(trans_index_path, verbose=True)

# load the BGE-M3 encoder
orig_factory = BGEM3(batch_size=32, max_length=1024, verbose=True)
orig_encoder = orig_factory.encoder()

trans_factory = BGEM3(batch_size=32, max_length=1024, verbose=True, model=f"andreaschari/{model}")
encoder = trans_factory.encoder()
logging.info("Loaded BGEM3 encoders")

# create the pipeline
pipeline_org = orig_encoder >> index.np_retriever()
pipeline_trans = encoder >> trans_index.np_retriever()

# # # neuclir topics processing

if "neuclir" in dataset_name:
    topics = dataset.get_topics()
    topics = topics.rename(columns={'title': 'query'})
    # # # use only qid and query columns
    topics = topics[['qid', 'query']]
else:
    topics = dataset.get_topics()

# run retrieval
logging.info("Running retrieval")
res_org = pipeline_org(topics)
res_trans = pipeline_trans(topics)
# save the results to disk
# check if the output folders exist
if not os.path.isdir(f"{out_path}/bgem3"):
    os.makedirs(f"{out_path}/bgem3")
if not os.path.isdir(f"{out_path}/{model}"):
    os.makedirs(f"{out_path}/{model}")

dataset_save = dataset_name.replace("/", "")
pt.io.write_results(res_org, f"{out_path}/bgem3/bgem3_{dataset_save}.gz")
pt.io.write_results(res_trans, f"{out_path}/{model}/{model}_{dataset_save}.gz")

logging.info("Saved results to disk")
# run the experiment
logging.info("Running experiment")
experiment = pt.Experiment(
    [res_org, res_trans],
    topics,
    dataset.get_qrels(),
    eval_metrics=[R@1000, MRR@10],
    verbose=True,
    names=["bgem3", "bgem3_tca"],
    baseline=0,
    correction='b'
)

print(experiment)
