import time
import torch
import os
import json
import gc
from transformers import AutoTokenizer
#Ministral 3 8B est un modele ImageTextToText
from transformers import AutoModelForImageTextToText,BitsAndBytesConfig

tokenizer=AutoTokenizer.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512-BF16")

#Les chemins des différents fichiers
dossier_script=os.path.dirname(os.path.abspath(__file__))
file_configs=os.path.join(dossier_script,"..","configs","bnb_configs.json")
file_baseline=os.path.join(dossier_script,"..","..","Create_a_baseline","inputs","baseline_prompt.json")

#configs est un dictionnaire contenant les config de quantization
with open(file_configs,"r",encoding="utf-8") as f:
    configs=json.load(f)

#Mappage en utilisant torch.type
dtype_map={"float16":torch.float16,"bfloat16":torch.bfloat16}

#baseline est un dictionnaire avec toute les caractéristique de la baseline
with open(file_baseline,"r",encoding="utf-8") as f:
    baseline=json.load(f)

model_config={"max_new_tokens": 1500, "do_sample": False}

#tests des différentes config
configs_results={}
for config_name,config in configs.items():
    config_dict={}
    #mappage
    if "bnb_4bit_compute_dtype" in config:
        config["bnb_4bit_compute_dtype"] = dtype_map[config["bnb_4bit_compute_dtype"]]
    #Création d'une instance de BitsAndBytesConfig
    config_true = BitsAndBytesConfig(**config)
    t=time.perf_counter()
    #Création du modèle avec la config voulue
    model=AutoModelForImageTextToText.from_pretrained(
        "mistralai/Ministral-3-3B-Instruct-2512-BF16",
        quantization_config=config_true,
        device_map="auto"
    )
    duree=time.perf_counter()-t
    config_dict["time_charge"]=duree
    #Poid du modèle quantifié avec "config"
    config_dict["model_weight(GO)"]=model.get_memory_footprint()/(1024 ** 3)
    resultat=[]
    #Test de l'efficacité/la précision du modèle 
    for e in baseline:
        messages=[{"role":"user","content":e["prompt"]}]
        inputs=tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True
        ).to(model.device)
        t=time.perf_counter()
        outputs=model.generate(**inputs,**model_config)
        duree=time.perf_counter()-t
        dict_curr={}
        dict_curr["id"]=e["id"]
        dict_curr["categorie"]=e["categorie"]
        dict_curr["prompt"]=e["prompt"]
        dict_curr["reponse"]=tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)
        dict_curr["duree"]=round(duree,2)
        resultat.append(dict_curr)
    config_dict["response"]=resultat
    configs_results[config_name]=config_dict
    #On vide le cache, la VRAM
    del model
    gc.collect()
    torch.cuda.empty_cache()

# Chemin de sauvegarde (à adapter selon ton arborescence exacte)
file_output = os.path.join(dossier_script, "..", "outputs", "bnb", "bnb_benchmark_results.json")

# On s'assure que le dossier existe
os.makedirs(os.path.dirname(file_output), exist_ok=True)

with open(file_output, "w", encoding="utf-8") as f:
    json.dump(configs_results, f, indent=4, ensure_ascii=False)

