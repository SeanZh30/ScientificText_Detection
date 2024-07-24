import json
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from argparse import ArgumentParser
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from datamodule import TextDataModule
from model import TextClassificationModel
from utils import read_json


logger = logging.getLogger(__name__)

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

def run(args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    # tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token is None:
      tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    DATASET_NAME = "SciText"

    dataset_paths = {
        "train": "./dataset/train.jsonl",
        "test": "./dataset/test.jsonl",
        "val": "./dataset/validation.jsonl"
    }

    train_dataset = read_json(dataset_paths["train"])
    test_dataset = read_json(dataset_paths["test"])
    val_dataset = read_json(dataset_paths["val"])

    # dataset = load_dataset(DATASET_NAME, cache_dir="datasets")

    # train_dataset = list(dataset["train"])
    # test_dataset = list(dataset["test"])
    # val_dataset = list(dataset["val"])

    labels = set()
    for instance in train_dataset:
        labels.add(instance["label"])
    config["num_classes"] = len(labels)
    
    data_module = TextDataModule(train_data=train_dataset,
                                 val_data=val_dataset,
                                 test_data=test_dataset,
                                 tokenizer=tokenizer,
                                 batch_size=config["batch_size"],
                                 max_seq_length=config["max_seq_length"],
                                 output_hidden_states=False)
    data_module.setup()

    if os.path.isdir(args.output_dir) is False:
        os.mkdir(args.output_dir)
    checkpoints = os.listdir(args.checkpoint_dir)
    for checkpoint_name in checkpoints:
        model_path = args.checkpoint_dir + "/" + checkpoint_name
        logger.info(f"Start to load the model from {model_path}")
        model = TextClassificationModel(**config)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["state_dict"])
        del checkpoint
        model.to("cuda")
        torch.cuda.empty_cache()

        output_path = args.output_dir + "/" + checkpoint_name.replace(".ckpt", "")

        # output_hidden_states(data_loader=data_module.train_dataloader(),
        #                      model=model,
        #                      output_path=output_path)

        accuracy = calculate_accuracy(data_loader=data_module.test_dataloader(),
                                      # data_loader=data_module.val_dataloader(),
                                      model=model,
                                      output_path=output_path)

        logger.info(f"Accuracy: {accuracy}")

        del model


def calculate_accuracy(data_loader: DataLoader, model, output_path: str):
    iterator = tqdm(data_loader)
    model.eval()

    all_preds, all_labels, all_ids, all_loss = [], [], [], []
    match_count, total_count = 0., 0.
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(iterator):
            for k, v in batch_data.items():
                if type(v) is not list:
                    batch_data[k] = v.to("cuda")

            loss, logits, _ = model(
                batch_data["text_input_ids"],
                attention_mask=batch_data["text_attention_mask"],
                labels=batch_data["labels"]
            )
            preds = torch.argmax(logits, dim=-1)
            all_preds += preds.detach().cpu().tolist()
            all_labels += batch_data["labels"].detach().cpu().tolist()
            all_loss.append(loss.detach().cpu().item())
            all_ids += batch_data["ids"]

            match_count += torch.sum(preds == batch_data["labels"]).detach().cpu().item()
            total_count += len(preds)
    ##
    cm = confusion_matrix(all_labels, all_preds)
    cm_file_path = os.path.join(output_path, "confusion_matrix.npy")
    np.save(cm_file_path, cm)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    heatmap_path = os.path.join(output_path, "confusion_matrix.png")
    plt.savefig(heatmap_path)
    plt.show()

    f1_macro = f1_score(all_labels, all_preds, average='macro')
    print(f"F1 Score (Macro): {f1_macro}")

    metrics_path = os.path.join(output_path, "metrics.json")
    results = {
        "accuracy": np.mean(all_preds == all_labels),
        "f1_score_macro": f1_macro
    }
    ##
    with open(metrics_path, "w") as f:
        json.dump(results, f)

    with open(output_path + "-results.json", "w", encoding="utf-8") as f:
        for i, loss, pred, label in zip(all_ids, all_loss, all_preds, all_labels):
            f.write(json.dumps({"id": i, "loss": loss, "pred": pred, "label": label}) + "\n")
    f.close()

    return match_count / total_count


def output_hidden_states(data_loader: DataLoader, model, output_path: str):
    iterator = tqdm(data_loader)
    model.eval()

    hidden_states_list, all_ids = [], []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(iterator):
            for k, v in batch_data.items():
                if type(v) is not list:
                    batch_data[k] = v.to("cuda")

            loss, logits, hidden_states = model(
                batch_data["text_input_ids"],
                attention_mask=batch_data["text_attention_mask"],
                labels=batch_data["labels"]
            )

            # last_hidden_state = hidden_states[-1][:, -1, :].detach().cpu()
            # hidden_states_list.append(last_hidden_state)

            logits = logits.detach().cpu()
            hidden_states_list.append(logits)

            all_ids += batch_data["ids"]

    hidden_states_list = torch.cat(hidden_states_list, dim=0)
    with open(output_path + "-hidden-states.pickle", "wb") as handle:
        pickle.dump(hidden_states_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_path + "-ids.json", "w", encoding="utf-8") as f:
        for i in all_ids:
            f.write(str(i) + "\n")
    f.close()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-checkpoint-dir", "--checkpoint-dir", help="checkpoint directory", type=str)

    parser.add_argument("-output-dir", "--output-dir", help="output directory", type=str)

    args = parser.parse_args()
    run(args)
