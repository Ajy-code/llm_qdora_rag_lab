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
quantization_config=BitsAndBytesConfig(
    load_in_8bit=True
)
model=AutoModelForImageTextToText.from_pretrained(
    "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    quantization_config=quantization_config,
    device_map="auto"
    )
# Calcule le chemin absolu du dossier où se trouve ton script '06_baseline.py' (le dossier 'src')
dossier_script = os.path.dirname(os.path.abspath(__file__))

# Construit le chemin absolu : src -> remonte (..) -> inputs -> baseline_prompt.json
chemin_fichier = os.path.join(dossier_script, "..", "inputs", "baseline_prompt.json")

with open(chemin_fichier, "r", encoding="utf-8") as file:
    baseline = json.load(file)

resultat=[]
model_config={"max_new_tokens": 1500, "do_sample": False}
for e in baseline:
    messages=[{"role":"user","content":e["prompt"]}]
    inputs=tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True
    ).to(model.device)
    t=time.time()
    outputs=model.generate(**inputs,**model_config)
    duree=time.time()-t
    dict_curr={}
    dict_curr["id"]=e["id"]
    dict_curr["categorie"]=e["categorie"]
    dict_curr["prompt"]=e["prompt"]
    dict_curr["reponse"]=tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)
    dict_curr["duree"]=round(duree,2)
    resultat.append(dict_curr)

# Crée le dossier s'il n'existe pas
os.makedirs("outputs", exist_ok=True)
# Ouvre le fichier en mode écriture ("w") et y injecte le dictionnaire
with open("outputs/baseline_result.json", "w", encoding="utf-8") as fichier:
    json.dump(resultat, fichier, indent=4, ensure_ascii=False)
