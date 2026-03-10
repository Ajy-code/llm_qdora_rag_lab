import torch
import time
import gc
import os
import json
# Ordre strict à PyTorch pour optimiser le rangement dans la VRAM
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Nettoyage agressif de la mémoire fantôme avant le lancement
torch.cuda.empty_cache()
gc.collect()

from transformers import AutoTokenizer
#Ministral 3 8B est un modele ImageTextToText
from transformers import AutoModelForImageTextToText,AutoModelForCausalLM
# BitsAndBytesConfig est la class qui permet de faire la quantififcation
from transformers import BitsAndBytesConfig

tokenizer=AutoTokenizer.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512-BF16")
#BitsAndBytesConfig réglage de la quantification 4bit
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
#Charge le model avec la quantization_config
model=AutoModelForImageTextToText.from_pretrained(
    "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    quantization_config=quantization_config,
    device_map="auto",
)

#Objectif: observer l'influence des paramètres (top_p(prend les p% meilleur),do_sample(sélectionne un échantillon ou que le meilleur),temperature(métrique de créativité, plus c'est faible, moins il est créatif))
message=[{"role":"user","content":"Écris une courte histoire de 5 lignes sur un développeur qui découvre l'IA."}]
inputs=tokenizer.apply_chat_template(
    message,
    return_tensors="pt",
    add_generation_prompt=True,
    return_dict=True
).to(model.device)
#Les configs que je vais tester
model_config=[
    {"name": "Mode Déterministe (Greedy)", "max_new_tokens": 150, "do_sample": False},
    {"name": "Mode Équilibré", "max_new_tokens": 150, "do_sample": True, "temperature": 0.5, "top_p": 0.9},
    {"name": "Mode Très Créatif", "max_new_tokens": 150, "do_sample": True, "temperature": 1.2, "top_p": 0.95}
]
Resultat=[]
for config in model_config:
    config_name=config["name"]
    config.pop("name")
    #début du chrono
    debut=time.time()
    outputs_curr=model.generate(**inputs,**config)
    #fin du chrono
    duree=time.time()-debut
    dict_curr={}
    dict_curr[f"Réponse du mode: {config_name}"]=tokenizer.decode(outputs_curr[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)
    dict_curr["duree"]=duree
    Resultat.append(dict_curr)
# Crée le dossier s'il n'existe pas
os.makedirs("outputs", exist_ok=True)
# Ouvre le fichier en mode écriture ("w") et y injecte le dictionnaire
with open("outputs/compare_parameters_model.json", "w", encoding="utf-8") as fichier:
    json.dump(Resultat, fichier, indent=4, ensure_ascii=False)

