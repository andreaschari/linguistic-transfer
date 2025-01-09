# Improving Low-Resource Retrieval Effectiveness using Zero-Shot Linguistic Similarity Transfer

Code for the ECIR 2025 IR4Good Paper

## Abstract

Globalisation and colonisation have led the vast majority of
the world to use only a fraction of languages, such as English and French,
to communicate, excluding many others. This has severely affected the
survivability of many now-deemed vulnerable or endangered languages,
such as Occitan and Sicilian. These languages often share some char-
acteristics, such as elements of their grammar and lexicon, with other
high-resource languages, e.g. French or Italian. They can be clustered
into groups of language varieties with various degrees of mutual intel-
ligibility. Current search systems are not usually trained on many of
these low-resource varieties, leading search users to express their needs
in a high-resource one instead. This problem is further complicated when
most information content is expressed in a high-resource language, in-
hibiting even more retrieval in low-resource languages. We show that
current search systems are not robust across language varieties, severely
affecting retrieval effectiveness. Therefore, it would be desirable for these
systems to leverage the capabilities of neural models to bridge the dif-
ferences between these varieties. This can allow users to express their
needs in their low-resource variety and retrieve the most relevant doc-
uments in a high-resource one. To address this, we propose fine-tuning
neural rankers on pairs of language varieties, thereby exposing them to
their linguistic similarities. We find that this approach improves the per-
formance of the varieties upon which the models were directly trained,
thereby regularising these models to generalise and perform better even
on unseen language variety pairs. We additionally explore whether this
approach can transfer across language families, where we observe mixed
results, which opens doors for future research.

## Setup
The instructions here are focused on setting up a conda environment.

This code was developed and tested with Python 3.10. 

First you should create a virtual environment and load the dependencies in the `requirements.txt` file.

```bash
conda create -n lt --file requirements.txt
conda activate lt
```

To run the retrieval and re-ranking experiments for BM25, BGE-M3 and mT5, you will need to install the following dependencies to run `variations_experiments.py` (The pyterrier dependencies are included in the `requirements.txt` file):

- [pyterrier](https://pyterrier.readthedocs.io/)
- [pyterrier_xlang](https://github.com/seanmacavaney/pyterrier_xlang)
- [pyterrier_dr[bgem3]](https://github.com/terrierteam/pyterrier_dr) (Check BGE-M3 Encoder section of `pyterrier_dr` for installation instructions.)
- [pyterrier_pisa](https://github.com/terrierteam/pyterrier_pisa)
- [pyterrier_t5](https://github.com/terrierteam/pyterrier_t5)

To run retrieval using ColBERT-XM please use our fork of the `xm-retrievers` package from [here](https://github.com/andreaschari/xm-retrievers).

## Retrieval

### BM25, BGE-M3 and mT5

#### Preparation

To run BM25 and BGE-M3 retrieval experiments you need to build an index. You can use the `index_with_bgem3.py` and `index_with_bm25.py` scripts to build the indexes for BGE-M3 and BM25 respectively.

Example usage:

```python
python index_with_bgem3.py --dataset mmarco/v2/dt --model bge-m3-lt-cafr --index /path/to/output
```

or for BM25:

```python
python index_with_bm25.py --dataset mmarco/v2/dt --index /path/to/output --language nl
```

#### mMARCO

The `variations_experiments.py` is the main script used to run the BM25, BGE-M3 and mT5 MMARCO retrieval experiments.

The script requires you to change the following variables:

1. `TRANSLATIONS_DIR` change this to the directory where the query translations are stored
2. `RETRIEVAL_RESULTS_DIR` change this to the directory where the retrieval results are to be stored

Example usage:

```python
python variations_experiments.py --lang ca --model bge-m3-lt-cafr --index /path/to/index --evaluate
```

This will run use BGE-M3 (Catalan-French) model  and run retrieval experiments on Catalan and French queries on the French collection, save the results to the `RETRIEVAL_RESULTS_DIR` and evaluate the results.

#### neuMARCO and neuCLIR

To run the neuMARCO and neuCLIR retrieval experiments you can use the `neuclir_neumarco_retrieval.py` and the `neuclir_neumarco_mt5.py` script for BGE-M3 and mT5 respectively.

### ColBERT-XM

To run the experiments you can use the `run_multi_vector_biencoder.sh` script from the `xm-retrievers` repo. (Check the `xm-retrievers` repository for more usage details.)

#### mMARCO

To reproduce the `mMARCO` retrieval experiments using ColBERT-XM, you need to make the following tweaks to the `mmarco.py` file in `xm-retrievers/src/data`:

1. For evaluation make sure the `data_filepaths['test_queries']` is set to the path of the translated queries file. e.g. the Catalan queries

2. For training make sure the `data_filepaths['train_queries']` is set to the path of the translated queries file. e.g. the Catalan queries

#### neuMARCO

To reproduce the `neuMARCO` retrieval experiments using ColBERT-XM, you need to make the following tweaks to the `mmarco.py` file in `xm-retrievers/src/data`:

1. Make sure the `data_filepaths['collection']` is set to the path of the neumarco collection file.

2. Comment out the `data_filepaths['train_queries']` and `data_filepaths['test_queries']` and leave the original `self.download_if_not_exists(url)` call.

#### neuCLIR

To reproduce the `neuCLIR` retrieval experiments you can use the `xm-retrievers` fork and just change the `DATA` variable of `xm-retrievers/scripts/run_multi_vector_biencoder.sh` to "neuclir". (Check the `xm-retrievers` repository for more usage details.)

## Fine-tuning the models

First requirement for BGE-M3 and mT5 fine-tuning is to create a jsonl file of the training data + mined negatives. For reproducibility you can find our traning jsonl files in the released dataset (see Checkpoints and Dataset on Hugging Face 🤗 Section).

If you want to generate your own training data with negatives with a different configuration, first download the mMARCO dataset of desired language. Then use the `finetune_preprocessing.py` script to create the training jsonl file. You need to change paths in the `PATHS` section of the script to match your setup.

The script uses the sampling code from the [xm-retrievers](https://github.com/ant-louis/xm-retrievers) repository.

### BGE-M3

You can use the `finetune_bgem3.sh` script to fine-tune the BGE-M3 model on the mMARCO dataset. Only requirement is setting the `output_dir` variable to the desired output directory and `train_data` to the path of the training jsonl file.

### mT5

You can use the `finetune_mt5.py` script to fine-tune the mT5 model on the mMARCO dataset. There are a few paths in the script that need to be changed to match your setup such as `MODEL_SAVE_PATH` and `QUERIES_PATH`.

### ColBERT-XM

To finetune ColBERT-XM you can use the `run_multi_vector_biencoder.sh` script from the `xm-retrievers` repo. (Check the `xm-retrievers` repository for more usage details.)

## Checkpoints and Dataset on Hugging Face 🤗

The translated data used in the paper are available [here](https://huggingface.co/datasets/andreaschari/mmarco-lt).

### Model Checkpoints

| Model       | Hugging Face URL                                                                 |
|-------------|-------------------------------------------------------------------------------|
| BGE-M3 (Catalan-French)   | [Download](https://huggingface.co/andreaschari/bge-m3-lt-cafr)|
| BGE-M3 (Afrikaans-Dutch)  | [Download](https://huggingface.co/andreaschari/bge-m3-lt-afdt)|
| mT5 (Catalan-French)        | [Download](https://huggingface.co/andreaschari/mt5-unicamp-lt-cafr)|
| mT5 (Afrikaans-Dutch)       | [Download](https://huggingface.co/andreaschari/mt5-unicamp-lt-afdt)|
| ColBERT-XM (Catalan-French) | [Download](https://huggingface.co/andreaschari/colbert-xm-lt-cafr)|
| ColBERT-XM (Afrikaans-Dutch) | [Download](https://huggingface.co/andreaschari/colbert-xm-lt-afdt)|

## Citation

WIP
