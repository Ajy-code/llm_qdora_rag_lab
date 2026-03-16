#Nettoyage final du dataset ( on élimine les doublons, on réarrange la forme, etc )
import os
import json
import hashlib
import re

dossier_script=os.path.dirname(os.path.abspath(__file__))
cheminSFTExemple=os.path.join(dossier_script,"..","data","interim","quality_passed_examples.jsonl")

#nettoyage va enlever les esapce au début et à la fin s'il y en a, améliorer la forme
def nettoyage(texte):
    text=texte.strip()
    texte_propre = re.sub(r'\n{3,}', '\n\n', text)
    return texte_propre

#On concatène de sorte à éviter les faux positifs
def concatenation(instruction,response,contexte):
    return f"{instruction}|{response}|{contexte}"


cleaned_examples=[]
rejected_duplicates=[]
with open(cheminSFTExemple,"r",encoding="utf-8") as f:
    deja_vus = set()
    for line in f:
        exemple=json.loads(line)
        #Je réarrange les textes dans le bon formats
        exemple["instruction"]=nettoyage(exemple["instruction"])
        exemple["response"]=nettoyage(exemple["response"])
        exemple["context"]=nettoyage(exemple["context"])
        texte_totale=concatenation(exemple["instruction"],exemple["response"],exemple["context"])
        empreinte_ex=hashlib.sha256(texte_totale.encode('utf-8')).hexdigest()
        if empreinte_ex in deja_vus:
            rejected_duplicates.append(exemple)
        else:
            deja_vus.add(empreinte_ex)
            cleaned_examples.append(exemple)

def create_a_jsonl(path,liste):
    with open(path,"w",encoding="utf-8") as f:
        for elem in liste:
            f.write(json.dumps(elem,ensure_ascii=False) + '\n')

dirs_clean_ex=os.path.join(dossier_script,"..","data","SFTExemple")
os.makedirs(dirs_clean_ex,exist_ok=True)
chemin_clean=os.path.join(dossier_script,"..","data","SFTExemple","cleaned_examples.jsonl")
chemin_rejects=os.path.join(dossier_script,"..","data","rejects","rejected_duplicates.jsonl")

create_a_jsonl(chemin_clean,cleaned_examples)
create_a_jsonl(chemin_rejects,rejected_duplicates)





