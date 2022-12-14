import os
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from transformers import pipeline
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_metric

from datasets import load_dataset
imdb = load_dataset("imdb")

# small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])
# small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])

small_train_dataset = imdb["train"].shuffle(seed=42)
small_test_dataset = imdb["test"].shuffle(seed=42)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)


from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)




def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}




repo_name = "laihanel/sentiment-analysis_gwu"

training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# trainer.train()
trainer.evaluate()
trainer.push_to_hub()



