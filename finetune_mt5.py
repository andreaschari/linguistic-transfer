import ir_datasets
import pandas as pd
import torch
import wandb
import logging

from pyterrier_t5 import mT5ReRanker
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW
from tqdm import tqdm

_logger = ir_datasets.log.easy()
torch.cuda.empty_cache()
torch.manual_seed(0)

# SETUP
BATCH_SIZE = 4
MAX_EPOCHS = 1
LEARNING_RATE = 5e-12
## change the path below to your configuration
MODEL_SAVE_PATH = '/root/nfs/CLIR/data/models/mt5-unicamp-tdt-5e-12'
OUTPUTS = ['yes', 'no']


def iter_train_samples():
  ## change the path below to your configuration
  QUERIES_PATH = '/root/nfs/CLIR/data/mmarco.v2.dt_af.train.judged.tsv'
  ## change the dataset to the one you want to use
  dataset = ir_datasets.load('mmarco/v2/dt/train')
  docs = dataset.docs_store()
  # Load translated queries
  translated_queries_df = pd.read_csv(f'{QUERIES_PATH}', sep='\t', names=['qid', 'query'], on_bad_lines="skip")
  queries = {str(query['qid']) : query['query'] for _, query in translated_queries_df.iterrows()}
  while True:
    for qid, dida, didb in dataset.docpairs_iter():
      if qid in queries:
        yield 'Query: ' + queries[qid] + ' Document: ' + docs.get(dida).text + ' Relevant:', OUTPUTS[0]
        yield 'Query: ' + queries[qid] + ' Document: ' + docs.get(didb).text + ' Relevant:', OUTPUTS[1]

train_iter = iter_train_samples()

model = MT5ForConditionalGeneration.from_pretrained("unicamp-dl/mt5-base-mmarco-v2").cuda()
tokenizer = T5Tokenizer.from_pretrained("unicamp-dl/mt5-base-mmarco-v2")
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

reranker = mT5ReRanker(verbose=False, batch_size=BATCH_SIZE)
reranker.REL = tokenizer.encode(OUTPUTS[0])[0]
reranker.NREL = tokenizer.encode(OUTPUTS[1])[0]

# Initialize wandb
wandb.init(project="mt5-training", config={
  "batch_size": BATCH_SIZE,
  "max_epochs": MAX_EPOCHS,
  "learning_rate": LEARNING_RATE,
  "model": "google/mt5-base"
})

epoch = 0
model.train()

_logger.info("Starting training")
_logger.info(f"Batch size: {BATCH_SIZE}")
_logger.info(f"Max epochs: {MAX_EPOCHS}")

while epoch < MAX_EPOCHS:
    total_loss = 0
    count = 0
    for _ in tqdm(range(497188 // BATCH_SIZE), desc=f"Epoch {epoch}"):
      inp, out = [], []
      for _ in range(BATCH_SIZE):
        inp_, out_ = next(train_iter)
        inp.append(inp_)
        out.append(out_)
      inp_ids = tokenizer(inp, return_tensors='pt', padding=True).input_ids.cuda()
      out_ids = tokenizer(out, return_tensors='pt', padding=True).input_ids.cuda()
      loss = model(input_ids=inp_ids, labels=out_ids).loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      total_loss += loss.item()
      count += 1

      # Log loss to wandb
      wandb.log({"loss": total_loss / count})
    _logger.info(f'epoch {epoch} loss {total_loss / count}')
     # save the model
    model.save_pretrained(f'{MODEL_SAVE_PATH}/epoch-{epoch}')
    _logger.info("Saved model to disk")
    epoch += 1
# Log final checkpoint to wandb
wandb.save(f'{MODEL_SAVE_PATH}/epoch-final-{epoch}')
wandb.finish()
_logger.info("Finished training")