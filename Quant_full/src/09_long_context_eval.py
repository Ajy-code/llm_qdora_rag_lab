import os
from llama_cpp import Llama
import gc
import json

dossier_script=os.path.dirname(os.path.abspath(__file__))
RAG_test_file=os.path.join(dossier_script,"..","prompts","RAG_test_efficiency.txt")
MODEL_ref=os.path.join(dossier_script,"..","models","gguf_ref","Ministral-3B-BF16.gguf")
MODEL_quant=os.path.join(dossier_script,"..","models","gguf_quantized","Ministral-3B-Q4_K_M.gguf")
MODEL_imatrix=os.path.join(dossier_script,"..","models","gguf_quantized","Ministral-3B-Q4_K_M-IMATRIX.gguf")

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

resp=[]
aiguille="Le code secret d'annulation du serveur central est 8943-XYZ"
models=[MODEL_imatrix,MODEL_quant,MODEL_ref]
positions=["debut","milieu","fin"]
for model in models:
    nom_modele = os.path.basename(model)
    llm=Llama(
        model_path=model,
        n_ctx=4096,
        n_gpu_layers=32,
        verbose=False
    )
    for pos in positions:
        message=[{"role":"user","content":f"Voici un document de référence : {preparer_contexte(texte_base=texte_base,aiguille=aiguille,position=pos)}; Question : Quel est le code secret d'annulation du serveur central ? "}]
        rep=llm.create_chat_completion(
            messages=message,
            max_tokens=150,
            temperature=0.1
        )

        reponse_modele=rep["choices"][0]["message"]["content"]
        succes = "8943-XYZ" in reponse_modele
        
        # Stockage dans le dictionnaire
        dict_curr = {
            "modele": nom_modele,
            "position": pos,
            "reponse_brute": reponse_modele.strip(),
            "exactitude": succes
        }
        resp.append(dict_curr)
    
    del llm
    gc.collect()

results_file=os.path.join(dossier_script,"..","outputs","test","test_rag.json")

with open(results_file,"w",encoding="utf-8") as f:
    json.dump(resp,f,indent=4)


