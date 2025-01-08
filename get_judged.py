# gets the judged subset of an mmarco dataset based on the msmarco train judegd set and returns a tsv
import pandas as pd
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--msmarco_path", type=str, help="path to msmarco train judged tsv", required=True)
argparser.add_argument("--mmarco_path", type=str, help="path to mmarco train tsv", required=True)
argparser.add_argument("--output_path", type=str, help="path to save the judged subset", required=True)

args = argparser.parse_args()

msmarco_path = args.msmarco_path
mmarco_path = args.mmarco_path
output_path = args.output_path

# Load both tsv files
msmarco_train = pd.read_csv(msmarco_path, sep="\t", header=None, names=["qid", "query"])

mmarco_train = pd.read_csv(mmarco_path, sep="\t", header=None, names=["qid", "query"])

# Get the intersection of the two datasets based on the qid

judged_subset = mmarco_train[mmarco_train["qid"].isin(msmarco_train["qid"])]

# Save the judged subset to a tsv file
judged_subset.to_csv(output_path, sep="\t", header=False, index=False)

print(f"Saved judged subset to {output_path}")