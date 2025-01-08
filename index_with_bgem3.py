"""use pyterrier_dr to build Flex indices using BGE-M3 encodings"""
import argparse
import pyterrier as pt
from pyterrier_dr import BGEM3, FlexIndex
# from pyterrier_caching import IndexerCache
# pt.init()


# Create an argument parser
parser = argparse.ArgumentParser(description='Indexing script')

# Add an argument for the language and the dataset
parser.add_argument('--dataset', help='Dataset to index (ir-datasets name format)')
parser.add_argument('--batch_size', help='Batch size for encoding', default=64)
parser.add_argument('--max_length', help='Max length for encoding', default=1024)
parser.add_argument('--model', help='BGE-M3 model to use for encoding', choices=["bgem3", "bge-m3-lt-cafr", "bge-m3-lt-afdt"])
parser.add_argument('--index', help='Index path')

# Parse the command line arguments
args = parser.parse_args()
dataset = args.dataset
model = args.model
tlang = args.tlang
batch_size = int(args.batch_size)
max_length = int(args.max_length)
index_path = args.index

# create a BGEM3 encoder
if model == "bgem3":
    factory = BGEM3(batch_size=batch_size, max_length=max_length, verbose=True)
else:
    factory = BGEM3(batch_size=batch_size, max_length=max_length, verbose=True, model=f"andreaschari/{model}")

index = FlexIndex(f"{index_path}/{dataset}_{model}", verbose=True)
encoder = factory.doc_encoder()

print(f"Building Flex index for {dataset} dataset...")

# for neuCLIR
# indexing_pipeline = pt.apply.text(lambda x: '{title}\n{text}'.format(**x)) >> encoder >> index
indexing_pipeline = encoder >> index

indexing_pipeline.index(pt.get_dataset(f"irds:{dataset}").get_corpus_iter())
