#Premier test de la méthode LoRa.
#Comme j'ai une démarche pédagogique, je vais faire manuellement le greffage des adapteur, et afficher les paramètres entrainable du nouveaux modèles, pour que je puisse directement voir si je me suis trompé
#A terme, utilisation de l'approche TRL automatisé
import os
import json
import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from peft import get_peft_model
from trl import SFTTrainer,SFTConfig
from transformers import TrainingArguments,BitsAndBytesConfig,AutoTokenizer,AutoModelForImageTextToText
from datasets import Dataset

dossier_script=os.path.dirname(os.path.abspath(__file__))
train_file=os.path.join(dossier_script,"..","data","formatted","train_sft.jsonl")

#Ouverture du fichier train, je prend quelques exemples pour faire mon dry_run_lora 
train=[]
with open(train_file,"r",encoding="utf-8") as f:
    for exemple in f:
        train.append(json.loads(exemple))

train_dry=Dataset.from_list(train[:int(len(train)*0.6)])

#Le tokenizer du model
tokenizer=AutoTokenizer.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512-BF16")
#Quantification du model
config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model=AutoModelForImageTextToText.from_pretrained(
    "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    quantization_config=config
)
#Conversion du modèle quantifié au bon format
model=prepare_model_for_kbit_training(model)
#Configuration des paramètres pour le fine-tuning
lora_config=LoraConfig(
    r=16,
    lora_alpha=2*16,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none"
)
#Greffage de l'adaptateur
model=get_peft_model(
    model, 
    peft_config=lora_config
    )
#Verification que seul les matrices A,B gréffer de faible rang sont entrainable
model.print_trainable_parameters()
#Réalisation d'un petit train pour vérifier que tout a bien été configurer
output_dir=os.path.join(dossier_script,"..","data","lora_dry_run")
os.makedirs(output_dir,exist_ok=True)

training_arg=SFTConfig(
    output_dir=output_dir,
    max_steps=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=2,
    save_steps=10,
    seed=42,
    bf16=True,
    max_length=1500,
)

trainer=SFTTrainer(
    model,
    args=training_arg,
    train_dataset=train_dry ,
    processing_class=tokenizer
)

#Lancement try_run_lora ( pour tester que le fine-tuning marche )
print("Lancement du LoRa Dry run")
train_result=trainer.train()

#Extraction des métriques globales d'entrainement
metrics=train_result.metrics

#On force la sauvegarde
trainer.save_metrics("train",metrics)
trainer.save_state()

print("LoRa Dry run terminé")