import torch
import transformers
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoModel
from torchmetrics import Accuracy


class TextClassificationModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # config = BertConfig.from_pretrained(self.hparams.model_name, output_hidden_states=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.model_name,
                                                                        return_dict=True,
                                                                        output_hidden_states=True,
                                                                        num_labels=self.hparams.num_classes)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        if self.hparams.num_classes == 2:
            self.accuracy = Accuracy(task="binary", num_classes=self.hparams.num_classes)
        else:
            self.accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return output.loss, output.logits, output.hidden_states

    def training_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']

        loss, logits, _ = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']

        loss, logits, _ = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        preds = torch.argmax(logits, dim=-1)
        accuracy = self.accuracy(preds, labels)
        step_output = dict(val_step_loss=loss, val_step_accuracy=accuracy)

        self.validation_step_outputs.append(step_output)

        return step_output

    def on_validation_epoch_end(self):
        batch_n = len(self.validation_step_outputs)

        avg_loss = sum(x['val_step_loss'] for x in self.validation_step_outputs) / batch_n
        avg_accuracy = sum(x['val_step_accuracy'] for x in self.validation_step_outputs) / batch_n

        self.log('val_loss', avg_loss, prog_bar=True, logger=True, on_epoch=True)
        self.log('val_accuracy', avg_accuracy, prog_bar=True, logger=True, on_epoch=True)

    def test_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']

        loss, logits, hidden_state = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        '''
        configure optimizers
        '''
        # Apply weight decay to all parameters except for the biases and the weight for Layer Normalization
        no_decay = ["bias", "LayerNorm.weight"]

        # Per-parameter optimization.
        # Each dict defines a parameter group and contains the list of parameters to be optimized in a key `params`
        # Other keys should match keyword arguments accepted by the optimizers and
        # will be used as optimization params for the parameter group
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # optimizer = torch.optim.Adagrad(
        #     optimizer_grouped_parameters,
        #     lr=self.hparams.lr,
        #     weight_decay=self.hparams.weight_decay,
        # )

        if self.hparams.schedule_name == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_updates,
                num_training_steps=self.hparams.total_num_updates,
            )
        elif self.hparams.schedule_name == "polynomial":
            scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_updates,
                num_training_steps=self.hparams.total_num_updates,
                lr_end=self.hparams.lr_end,
            )

        lr_dict = {
            "scheduler": scheduler,  # scheduler instance
            "interval": "step",  # The unit of the scheduler's step size. 'step' or 'epoch
            "frequency": 1,  # corresponds to updating the learning rate after every `frequency` epoch/step
            "name": f"LearningRateScheduler-{self.hparams.schedule_name}",  # Used by a LearningRateMonitor callback
        }

        return [optimizer], [lr_dict]
    