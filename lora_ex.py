import torch
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# Create a sample dataset
# Expanded with more positive and negative examples to improve training stability.
data = {
    "text": [
        "API latency",
        "Service downtime",
        "User experience",
        "Cpu usage is high",
        "Memory usage is high",
        "Database connection error",
        "Normal operation",
        "System running smoothly",
        "Disk space low",
        "Network latency",
        "Application crash",
        "Everything working fine",
        "Response time slow",
        "All systems nominal",
        "Performance degraded",
        "User experience is good",
        "Service is healthy",
        "No errors found",
        "Routine maintenance mode",
        "Latency is within acceptable range",
        "Performance is stable",
        "Backup completed successfully",
        "All services are online",
        "CPU usage is within normal limits",
        "Memory utilization is normal",
        "Disk usage is healthy",
        "Network throughput is stable",
        "Response time is acceptable",
        "Scheduled update completed",
        "No outage reported",
        "Server overload",
        "Database timeout",
        "Authentication failures",
        "High error rate",
        "Service unavailable",
        "System failure",
        "Disk read error",
        "Packet loss detected",
        "Timeout while connecting",
        "High request queue length"
    ],
    "labels": [
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1
    ]
}
dataset = Dataset.from_dict(data)

# Split the dataset into train and test sets
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

model_name="distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Add Label names
model.config.id2label = {0: "Normal", 1: "Issue"}
model.config.label2id = {"Normal": 0, "Issue": 1}

#predict function
def predict(text,model,tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
    
    return model.config.id2label[predicted_class_id]

print("Prediction before training:", predict("Service downtime", model, tokenizer))

# Tokenzie the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True,max_length=64)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("labels", "label")
test_dataset = test_dataset.rename_column("labels", "label")

#set Pytorch format 
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def print_trainable_parameters(model, message):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{message}: Trainable parameters: {trainable_params} | Total parameters: {total_params} | Trainable percentage: {100 * trainable_params / total_params:.2f}%")

print_trainable_parameters(model, "Before applying LoRA")

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_lin", "v_lin"], 
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["classifier", "pre_classifier"])

print("Applying LoRA to the model...")

model = get_peft_model(model, lora_config)
model.to(device="cuda" if torch.cuda.is_available() else "cpu")
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    do_eval=True,
    eval_steps=10,
    save_steps=10,
)

#Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {
        "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1)),
        "precision": precision_recall_fscore_support(p.label_ids, p.predictions.argmax(-1), average="binary")[0],
        "recall": precision_recall_fscore_support(p.label_ids, p.predictions.argmax(-1), average="binary")[1],
        "f1": precision_recall_fscore_support(p.label_ids, p.predictions.argmax(-1), average="binary")[2],
    }
)

#start  Lora fine-tuning
print("Starting LoRA fine-tuning...")
trainer.train()
print("LoRA fine-tuning completed.")
eval_results = trainer.evaluate()
for key, value in eval_results.items():
    print(f"{key}: {value}")

# Print a classification report for the test dataset
predictions = trainer.predict(test_dataset)
pred_labels = predictions.predictions.argmax(-1)
print("\nClassification Report:")
print(classification_report(predictions.label_ids, pred_labels, target_names=["Normal", "Issue"]))

print("Prediction after training:", predict("User experience", model, tokenizer))