#Comparer le modèle quantifié avec Imatrix et celui quantifié normalement
import os
import json
from llama_cpp import Llama
from tqdm import tqdm
import time

#Chemin des différents fichiers
dossier_script=os.path.dirname(os.path.abspath(__file__))
baseline_file=os.path.join(dossier_script,"..","..","Create_a_baseline","inputs","baseline_prompt.json")
model_imatrix_path=os.path.join(dossier_script,"..","models","gguf_quantized","Ministral-3B-Q4_K_M-IMATRIX.gguf")
model_quant_path=os.path.join(dossier_script,"..","models","gguf_quantized","Ministral-3B-Q4_K_M.gguf")

#Ouverture du fichier baseline
with open(baseline_file,"r",encoding="utf-8") as f:
    baseline=json.load(f)

model_imatrix=Llama(
    model_path=model_imatrix_path,
    verbose=False,
    n_ctx=2048,
)

model_quant=Llama(
    model_path=model_quant_path,
    verbose=False,
    n_ctx=2048
)

results=[]
for e in tqdm(baseline, desc="Évaluation des modèles"):
    dict_curr={}
    dict_curr["prompt"]=e["prompt"]
    dict_curr["id"]=e["id"]
    dict_curr["categorie"]=e["categorie"]
    reps={}
    rep_imatrix=model_imatrix.create_chat_completion(
        messages=[{"role":"user","content":e["prompt"]}],
        max_tokens=150
    )
    rep_quant=model_quant.create_chat_completion(
        messages=[{"role":"user","content":e["prompt"]}],
        max_tokens=150
    )
    reps["model_imatrix"]=rep_imatrix["choices"][0]["message"]["content"]
    reps["model_quant"]=rep_quant["choices"][0]["message"]["content"]
    dict_curr["reponses"]=reps
    results.append(dict_curr)

#Enregistrement des résultats
results_file=os.path.join(dossier_script,"..","outputs","test","result_test_comparaison_imatrix_quant.json")
os.makedirs(os.path.dirname(results_file),exist_ok=True)

with open(results_file,"w",encoding="utf-8") as f:
    json.dump(results,f,indent=4,ensure_ascii=False)



