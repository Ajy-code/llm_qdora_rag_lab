# Fine-tuning sur tout le dataset avec la méthode LoRa
import os
import json
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForImageTextToText, BitsAndBytesConfig
from trl import SFTTrainer,SFTConfig
from datasets import Dataset
import torch

dossier_script=os.path.dirname(os.path.abspath(__file__))
train_file=os.path.join(dossier_script,"..","data","formatted","train_sft.jsonl")
val_file = os.path.join(dossier_script, "..", "data", "formatted", "val_sft.jsonl")

#Lecture des exemples pour le fine-tuning
train=[]
with open(train_file,"r",encoding="utf-8") as f:
    for exemple in f:
        train.append(json.loads(exemple))

val = []
with open(val_file, "r", encoding="utf-8") as f:
    for exemple in f:
        val.append(json.loads(exemple))

#Mise des données au bon format
val = Dataset.from_list(val)
train=Dataset.from_list(train)

#Création du tokenizer 
tokenizer=AutoTokenizer.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512-BF16")
#Quantification du model
quantization_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model=AutoModelForImageTextToText.from_pretrained(
    "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    quantization_config=quantization_config
)
#Mise du model au bon format
model=prepare_model_for_kbit_training(model)
#Paramètrage des parametre de sft LoRa
lora_config=LoraConfig(
    r=16,
    lora_alpha=2*16,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear"
)
#Mise en place du fine-tuning LoRa
output_dir=os.path.join(dossier_script,"..","data","lora_sft")
os.makedirs(output_dir,exist_ok=True)

training_arg=SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=1, 
    eval_accumulation_steps=1,    
    learning_rate=2e-4,
    logging_steps=2,
    save_steps=10,
    seed=42,
    bf16=True,
    max_length=1500,
    eval_strategy="epoch"
)

trainer=SFTTrainer(
    model,
    peft_config=lora_config,
    args=training_arg,
    train_dataset=train,
    processing_class=tokenizer,
    eval_dataset=val,
)

print("Début du fine-tuning")
train_result=trainer.train()

#Extraction des métriques globales d'évaluation/d'entrainement
metrics=train_result.metrics
eval_metrics = trainer.evaluate()

#On force la sauvegarde
trainer.save_metrics("eval", eval_metrics)
trainer.save_metrics("train",metrics)
trainer.save_state()

print("Fine-tuning terminé")

max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
print(f"Mémoire GPU max réservée : {max_memory:.2f} Go")
