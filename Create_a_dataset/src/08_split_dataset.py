#Split le dataset

import os
import random
import json
import collections

random.seed(42)

dossier_script=os.path.dirname(os.path.abspath(__file__))
cheminSFTExemple=os.path.join(dossier_script,"..","data","SFTExemple","cleaned_examples.jsonl")

#Ouvrir le fichier avec les "bon" exemples, dict_exemples est un dictionnaire d'exemple trié selon le challenge_type (les clés)
dict_exemples=collections.defaultdict(list)
with open(cheminSFTExemple,"r",encoding="utf-8") as f:
    for line in f:
        exemple=json.loads(line)
        challenge_type=exemple.get("challenge_type","Introuvable")
        dict_exemples[challenge_type].append(exemple)

#Création des liste d'entrainement et de validation
train=[]
val=[]

for listes_exemple in dict_exemples.values():
    n=int(0.8*len(listes_exemple))
    random.shuffle(listes_exemple)
    train.extend(listes_exemple[:n])
    val.extend(listes_exemple[n:])
    
random.shuffle(train)
random.shuffle(val)

def create_a_jsonl(path,liste):
    with open(path,"w",encoding="utf-8") as f:
        for elem in liste:
            f.write(json.dumps(elem,ensure_ascii=False) + '\n')

dirs_train_ex=os.path.join(dossier_script,"..","data","train")
dirs_val_ex=os.path.join(dossier_script,"..","data","val")
os.makedirs(dirs_train_ex,exist_ok=True)
os.makedirs(dirs_val_ex,exist_ok=True)
chemin_train=os.path.join(dossier_script,"..","data","train","train.jsonl")
chemin_val=os.path.join(dossier_script,"..","data","val","val.jsonl")

create_a_jsonl(chemin_train,train)
create_a_jsonl(chemin_val,val)


