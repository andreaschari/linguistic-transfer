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

## Reproducing the experiments

### Setup
The instructions here are focused on setting up a conda environment.

This code was developed and tested with Python 3.10.

First, create a virtual environment:

```bash
conda create -n lt
conda activate lt
```

To run the retrieval and re-ranking experiments for BM25, BGE-M3 and mT5, you will need to install the following dependencies to run `variations_experiments.py`:

- [pyterrier](https://pyterrier.readthedocs.io/)
- [pyterrier_xlang](https://github.com/seanmacavaney/pyterrier_xlang)
- [pyterrier_dr[bgem3]](https://github.com/terrierteam/pyterrier_dr) (Check BGE-M3 Encoder section of `pyterrier_dr` for installation instructions.)
- [pyterrier_pisa](https://github.com/terrierteam/pyterrier_pisa)
- [pyterrier_t5](https://github.com/terrierteam/pyterrier_t5)

To run retrieval using ColBERT-XM, please use our fork of the `xm-retrievers` package from [here](https://github.com/andreaschari/xm-retrievers).

### BM25, BGE-M3 and mT5

#### Preparation

To run BM25 and BGE-M3 retrieval experiments, you need to build an index. You can use the `index_with_bgem3.py` and `index_with_bm25.py` scripts to build the indexes for BGE-M3 and BM25, respectively.

Example usage:

```python
python index_with_bgem3.py --dataset mmarco/v2/dt --model bge-m3-lt-cafr --index /path/to/output
```

Or for BM25:

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

This will use the BGE-M3 (Catalan-French) model and run retrieval experiments on Catalan and French queries on the French collection, save the results to the `RETRIEVAL_RESULTS_DIR` and evaluate the results.

#### neuMARCO and neuCLIR

To run the neuMARCO and neuCLIR retrieval experiments, you can use the `neuclir_neumarco_retrieval.py` and the `neuclir_neumarco_mt5.py` script for BGE-M3 and mT5, respectively.

Example usage:

```python
python neuclir_neumarco_retrieval.py ---dataset neuclir/1/fa --index path/to/original_bgem3/index --trans_index /path/to/index/of/finetuned_bgem3/index --model bge-m3-lt-cafr --out_path /path/to/save/retrieval/res
```

```python
python neuclir_neumarco_mt5.py ---dataset neuclir/1/fa --first_stage /path/to/first_stage/results --model mt5-base-lt-cafr --out_path /path/to/save/retrieval/res
```

### ColBERT-XM

To run the experiments, you can use the `run_multi_vector_biencoder.sh` script from the `xm-retrievers` repo. (Check the `xm-retrievers` repository for usage details.)

#### mMARCO

To reproduce the `mMARCO` retrieval experiments using ColBERT-XM, you need to make the following tweaks to the `mmarco.py` file in `xm-retrievers/src/data`:

1. For evaluation, ensure the `data_filepaths['test_queries']` is set to the path of the translated queries file. e.g. the Catalan queries

2. For training, ensure the `data_filepaths['train_queries']` is set to the path of the translated queries file. e.g. the Catalan queries

#### neuMARCO

To reproduce the `neuMARCO` retrieval experiments using ColBERT-XM, you need to make the following tweaks to the `mmarco.py` file in `xm-retrievers/src/data`:

1. Make sure the `data_filepaths['collection']` is set to the path of the neumarco collection file.

2. Comment out the `data_filepaths['train_queries']` and `data_filepaths['test_queries']` and leave the original `self.download_if_not_exists(url)` call.

#### neuCLIR

To reproduce the `neuCLIR` retrieval experiments, you can use the `xm-retrievers` fork and just change the `DATA` variable of `xm-retrievers/scripts/run_multi_vector_biencoder.sh` to "neuclir". (Check the `xm-retrievers` repository for more usage details.)

To evaluate the results, you can use the `score_neuclir.py` script. You must download [some mapping files](https://huggingface.co/datasets/andreaschari/neuclir-mappings) to map the ColBERT-XM rankings to the original neuCLIR document IDs.

## Fine-tuning the models

The first requirement for BGE-M3 and mT5 fine-tuning is to create a JSON file of the training data + mined negatives. For reproducibility, you can find our training JSONL files in the released dataset (see Checkpoints and Dataset on Hugging Face 🤗 Section) for the link.

If you want to generate your own training data with negatives with a different configuration, first download the mMARCO dataset of the desired language. Then, use the `finetune_preprocessing.py` script to create the training JSONL file. You need to change paths in the `PATHS` section of the script to match your setup.

The script uses the sampling code from the [xm-retrievers](https://github.com/ant-louis/xm-retrievers) repository.

### BGE-M3

You can use the `finetune_bgem3.sh` script to fine-tune the BGE-M3 model on the mMARCO dataset. The only requirement is setting the `output_dir` variable to the desired output directory and `train_data` to the path of the training JSONL file.

### mT5

You can use the `finetune_mt5.py` script to fine-tune the mT5 model on the mMARCO dataset.

### ColBERT-XM

To finetune ColBERT-XM you can use the `run_multi_vector_biencoder.sh` script from the `xm-retrievers` repo. (Check the `xm-retrievers` repository for more usage details.)

## Using the fine-tuned models in PyTerrier

### BGE-M3 retrieval (using `pyterrier-dr[bgem3]`)

More detailed instructions for BGE-M3 retrieval can be found in the `pyterrier-dr` [documentation](https://github.com/terrierteam/pyterrier_dr) under the `BGE-M3 Encoder` Section.

The example below shows how to use one of the Linguistic Transfer variants of BGE-M3 for retrieval by changing the default `model_name` to one of the BGE-M3 models in the section below.

```python
import pyterrier as pt
from pyterrier_dr import BGEM3, FlexIndex

dataset = pt.get_dataset(f"irds:mmarco/v2/fr/dev/small")
topics = dataset.get_topics(tokenise_query=False)

factory = BGEM3(batch_size=32, max_length=1024, verbose=True, model_name="andreaschari/bge-m3-lt-afdt")
encoder = factory.query_encoder()

index = FlexIndex(f"mmarco/v2/fr_bgem3", verbose=True) #replace index with your location

pipeline = encoder >> idx.np_retriever()

first_stage_res = pipeline(topics)
```

### mT5 re-ranking (using `pyterrier-t5`)

More detailed instructions for mT5 retrieval using PyTerrier can be found in the `pyterrier-t5` documentation [here](https://github.com/terrierteam/pyterrier_t5)

The example below shows how to use one of the Linguistic Transfer variants of mT5 to rerank some first-stage retrieval results (such as the output of the example above) by changing the default `model_name` to one of the mT5 models in the section below.

```python
import pyterrier as pt
from pyterrier_t5 import mT5ReRanker

reranker = mT5ReRanker(model='andreaschari/mt5-base-lt-cafr', verbose=True)
pipeline = pt.text.get_text(dataset, "text") >> reranker

reranked_res = pipeline(first_stage_res)
```

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

```
@InProceedings{10.1007/978-3-031-88717-8_22,
author="Chari, Andreas
and MacAvaney, Sean
and Ounis, Iadh",
title="Improving Low-Resource Retrieval Effectiveness Using Zero-Shot Linguistic Similarity Transfer",
booktitle="Advances in Information Retrieval",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="290--306",
isbn="978-3-031-88717-8"
}
```
