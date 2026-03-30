import os
import json
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig, AutoTokenizer
from llama_cpp import Llama
import torch
import gc

dossier_script=os.path.dirname(os.path.abspath(__file__))
MODEL_ref=os.path.join(dossier_script,"..","models","gguf_ref","Ministral-3B-BF16.gguf")
MODEL_quant=os.path.join(dossier_script,"..","models","gguf_quantized","Ministral-3B-Q4_K_M.gguf")
MODEL_imatrix=os.path.join(dossier_script,"..","models","gguf_quantized","Ministral-3B-Q4_K_M-IMATRIX.gguf")
test_compare_file=os.path.join(dossier_script,"..","outputs","test","test_compare.txt")
RAG_test_file=os.path.join(dossier_script,"..","prompts","RAG_test_efficiency.txt")

tokenizer=AutoTokenizer.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512-BF16")

with open(RAG_test_file,"r",encoding="utf-8") as f:
    texte_base=f.read() 

def preparer_contexte(texte_base, aiguille, position):
    """
    Insère une phrase (aiguille) dans un texte long à une position donnée.
    position attendue : 'debut', 'milieu', ou 'fin'
    """
    # 1. Découper le texte en une liste de mots (basé sur les espaces)
    mots = texte_base.split()
    
    # 2. Calculer l'index d'insertion
    if position == "debut":
        index = 0
    elif position == "milieu":
        # Division entière (//) pour avoir un index valide
        index = len(mots) // 2 
    elif position == "fin":
        index = len(mots)
    else:
        raise ValueError("Position invalide. Choisir: debut, milieu, fin.")
        
    # 3. Insérer l'aiguille dans la liste
    mots.insert(index, aiguille)
    
    # 4. Reconstruire et retourner le texte final
    texte_final = " ".join(mots)
    return texte_final

aiguille="Le code secret d'annulation du serveur central est 8943-XYZ"
models=[MODEL_imatrix,MODEL_quant,MODEL_ref]

#Prompt tests
prompts=["Génère un dictionnaire JSON strict contenant 3 capitales européennes. Ne renvoie que le JSON, aucun texte avant ou après."
         ,"Explique le concept de l'attention dans les réseaux de neurones en utilisant une métaphore avec une bibliothèque. Sois précis et détaillé."
         ,"Combien d'yeux a un cheval normal ? Réponds en une seule phrase."
         ,f"Voici un document de référence : {preparer_contexte(texte_base=texte_base,aiguille=aiguille,position="millieu")}; Question : Quel est le code secret d'annulation du serveur central ? "]

results=[]
for model in models:
    model_name=os.path.basename(model)
    llm=Llama(
        model_path=model,
        n_ctx=4096,
        n_gpu_layers=32,
        verbose=False
    )
    for prompt in prompts:
        message=[{"role":"user","content":prompt}]
        rep=llm.create_chat_completion(
            messages=message,
            max_tokens=150,
            temperature=0.1
        )
        reponse=rep["choices"][0]["message"]["content"]
        dict_curr={
            "model_name":model_name,
            "prompt":prompt,
            "reponse":reponse.strip()
        }
        results.append(dict_curr)
    del llm
    gc.collect()

torch.cuda.empty_cache()
model=AutoModelForImageTextToText.from_pretrained(
    "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    device_map="auto"
)
model_config={"max_new_tokens": 1500, "do_sample": False}

for prompt in prompts:
        message=[{"role":"user","content":prompt}]
        inputs=tokenizer.apply_chat_template(
            message,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True
        ).to(model.device)
        outputs=model.generate(**model_config,**inputs)
        reponse=tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)
        dict_curr={
            "model_name":"Ministral-3-3B-Instruct-2512-BF16-HF",
            "prompt":prompt,
            "reponse":reponse
        }
        results.append(dict_curr)

with open(test_compare_file,"w",encoding="utf-8") as f:
    json.dump(results,f,indent=4)

