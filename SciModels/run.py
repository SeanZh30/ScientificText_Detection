import math
import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import AutoTokenizer
from datamodule import TextDataModule
from model import TextClassificationModel
# from datasets import load_dataset
from utils import read_json

DATASET_NAME = "SciText"
TRAIN_PATH = "./dataset/train.jsonl"
TEST_PATH = "./dataset/test.jsonl"
VAL_PATH = "./dataset/validation.jsonl"

train_dataset = read_json(TRAIN_PATH)
test_dataset = read_json(TEST_PATH)
val_dataset = read_json(VAL_PATH)


config = dict(
    model_name="allenai/scibert_scivocab_cased", # SciBERT
    # model_name="FacebookAI/roberta-base", # RoBERTa
    # model_name="microsoft/deberta-base", # DeBERTa
    # model_name="gpt2-large",
    n_epochs=3,
    gradient_accumulation_steps=1,
    batch_size=16,
    lr=2e-5,
    schedule_name="linear",
    weight_decay=0.0,
    warmup_updates=0,
    max_grad_norm=1.0,
    max_seq_length=512
    
)

num_update_steps_per_epoch = math.ceil(len(train_dataset) / config["gradient_accumulation_steps"])
config["total_num_updates"] = config["n_epochs"] * num_update_steps_per_epoch

labels = set()
for instance in train_dataset:
    labels.add(instance["label"])
config["num_classes"] = len(labels)

pl.seed_everything(42)

tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# tokenizer.pad_token = tokenizer.eos_token
data_module = TextDataModule(train_dataset, val_dataset, test_dataset, tokenizer, batch_size=config["batch_size"], max_seq_length=config["max_seq_length"])
model = TextClassificationModel(**config)


checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename=DATASET_NAME + "-checkpoint-SciBert512-{epoch}-{step}",
    verbose=False,
    monitor='val_loss',
    mode='min',
    save_weights_only=True,
    # every_n_train_steps=3000,
    save_top_k=1,
    save_last=False
)

# logger = WandbLogger(name="LM_SciText", save_dir=f"gpt-2-{DATASET_NAME}")
logger = WandbLogger(name="LM_SciText", save_dir=f"SciBert512-{DATASET_NAME}")


trainer = pl.Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    max_epochs=config["n_epochs"],
    # precision='bf16',  # for decreasing the GPU using rate
    #precision='bf16-true',  # for decreasing the GPU using rate
    #对large gpt2 要用bf16-true
    accelerator="gpu",
    devices=1,
    accumulate_grad_batches=config["gradient_accumulation_steps"],
    val_check_interval=0.25,
    gradient_clip_val=config["max_grad_norm"],
)

trainer.fit(model, data_module)
