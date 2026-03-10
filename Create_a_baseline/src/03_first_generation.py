import torch
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

print(f"Empreinte mémoire : {model.get_memory_footprint() / 1e9:.2f} Go")

#Il est temps de le faire générer du texte
message=[{"role":"user","content":"Que penses tu de la conssomation énergetique par l'ia actuelle ?"}]
# return_tensors="pt" veut dire "format PyTorch". .to("cuda") l'envoie direct dans la VRAM
inputs=tokenizer.apply_chat_template(
    message,
    return_tensors="pt",
    add_generation_prompt=True,
    return_dict=True
    ).to(model.device)
#Les ** sont nécéssaire pour déballer le dictionnaire que représente inputs
outputs=model.generate(**inputs,max_new_tokens=1500)
#Le modèle attendant un batch (cf durant l'entrainement), il renvoie outputs qui est un batch de réponses
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True))

messages1=[{"role":"user","content":"Quand s'est faite la révolution française"}]
messages2=[{"role":"user","content":"Fais moi un résumé en 5 ligne de: L'apprentissage automatique[1],[2] (en anglais : machine learning, litt. « apprentissage machine[1],[2] »), apprentissage artificiel[1] ou apprentissage statistique est un champ d'étude de l'intelligence artificielle qui se fonde sur des approches mathématiques et statistiques pour donner aux ordinateurs la capacité d'« apprendre » à partir de données, c'est-à-dire d'améliorer leurs performances à résoudre des tâches sans être explicitement programmés pour chacune. Plus largement, il concerne la conception, l'analyse, l'optimisation, le développement et l'implémentation de telles méthodes. On parle d'apprentissage statistique car l'apprentissage consiste à créer un modèle dont l'erreur statistique moyenne est la plus faible possible. "}]
messages3=[{"role":"user","content":"Répond sous forme JSON: Qu'est ce que l'IA"}]
messages=[messages1,messages2,messages3]
resultat=[]
for mess in messages:
    inputs_curr=tokenizer.apply_chat_template(
    mess,
    return_tensors="pt",
    add_generation_prompt=True,
    return_dict=True
    ).to("cuda")
    outputs_curr=model.generate(**inputs_curr,max_new_tokens=1500)
    dict_curr={}
    dict_curr["prompt"]=mess[0]["content"]
    dict_curr["réponse"]=tokenizer.decode(outputs_curr[0][inputs_curr["input_ids"].shape[1]:],skip_special_tokens=True)
    resultat.append(dict_curr)
# Crée le dossier s'il n'existe pas
os.makedirs("outputs", exist_ok=True)
# Ouvre le fichier en mode écriture ("w") et y injecte le dictionnaire
with open("outputs/generations_run_01.json", "w", encoding="utf-8") as fichier:
    json.dump(resultat, fichier, indent=4, ensure_ascii=False)

