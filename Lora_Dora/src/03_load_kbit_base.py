#Préparer la base quantifié pour l'entrainement avec peft
import os
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForImageTextToText
from peft import prepare_model_for_kbit_training
import torch

#Charge le tokenizer
tokenizer=AutoTokenizer.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512-BF16")

#Charge le modèle quantifié
quantization_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model=AutoModelForImageTextToText.from_pretrained(
    "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    quantization_config=quantization_config
)

#Convertir le modele quantifié au bon format pour l'entrainement
model=prepare_model_for_kbit_training(model)

#Afficher la taille du modèle, les modules principaux
print(f"La taille du model en GO: {model.get_memory_footprint()/1024**3}")
print(model)

