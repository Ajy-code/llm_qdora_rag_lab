#Conversion des données dans le bon format pour le fine-tuning
import json
import os

dossier_script=os.path.dirname(os.path.abspath(__file__))
train_file=os.path.join(dossier_script,"..","data","train.jsonl")
val_file=os.path.join(dossier_script,"..","data","val.jsonl")

#Lecture des exemple pour l'entrainement
train=[]
with open(train_file,"r",encoding="utf-8") as f:
    for exemple in f:
        train.append(json.loads(exemple))

#Lecture de l'entrainenement pour la validation
val=[]
with open(val_file,"r",encoding="utf-8") as f:
    for exemple in f:
        val.append(json.loads(exemple))

#Conversion des exemples au bon format
def conversion(exemples):
    converted=[]
    for exemple in exemples:
        dict_curr = {
            "prompt": exemple["context"] + '\n\n' + exemple["instruction"],
            "completion": exemple["response"]
        }
        converted.append(dict_curr)
    return converted
train_sft=conversion(train)
val_sft=conversion(val)

#Création des fichiers train_sft.jsonl et val_sft.jsonl
def create_file(path,liste):
    with open(path,"w",encoding="utf-8") as f:
        for elem in liste:
            f.write(json.dumps(elem,ensure_ascii=False) + '\n')


dir_formatted=os.path.join(dossier_script,"..","data","formatted")
os.makedirs(dir_formatted,exist_ok=True)
train_sft_file=os.path.join(dossier_script,"..","data","formatted","train_sft.jsonl")
val_sft_file=os.path.join(dossier_script,"..","data","formatted","val_sft.jsonl")

create_file(train_sft_file,train_sft)
create_file(val_sft_file,val_sft)

print("Conversion terminé")
