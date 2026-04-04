#Objectif: avoir des stats sur le nb de token de mes exemples
#Important pour éviter le Out of Memory, pour pouvoir débugguer
import os
import json
from transformers import AutoTokenizer

#Ouvrir les fichiers val_sft.jsonl et train_sft.jsonl
dossier_script=os.path.dirname(os.path.abspath(__file__))
train_sft_file=os.path.join(dossier_script,"..","data","formatted","train_sft.jsonl")
val_sft_file=os.path.join(dossier_script,"..","data","formatted","val_sft.jsonl")

def open_file(path_file):
    elems=[]
    with open(path_file,"r",encoding="utf-8") as f:
        for exemple in f:
            elems.append(json.loads(exemple))
    return elems

train=open_file(train_sft_file)
val=open_file(val_sft_file)

#Chargement du tokenizer
tokenizer=AutoTokenizer.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512-BF16")

#Avoir des statistiques intéressantes sur les exemples
def stats_exemples(exemples):
    results=[]
    max_nb_tokens=-1
    max_nb_tokens_prompt=-1
    max_nb_tokens_response=-1
    mean_tokens=0
    mean_tokens_prompt=0
    mean_tokens_response=0
    for exemple in exemples:
        nb_tokens_prompt=len(tokenizer.encode(exemple["prompt"]))
        nb_tokens_response=len(tokenizer.encode(exemple["completion"]))
        nb_tokens=nb_tokens_prompt+nb_tokens_response
        dict_curr={
            "nb_tokens_prompt":nb_tokens_prompt,
            "nb_tokens_response":nb_tokens_response,
            "nb_tokens":nb_tokens
        }
        results.append(dict_curr)
        if nb_tokens>max_nb_tokens:
            max_nb_tokens=nb_tokens
        if max_nb_tokens_prompt<nb_tokens_prompt:
            max_nb_tokens_prompt=nb_tokens_prompt
        if max_nb_tokens_response<nb_tokens_response:
            max_nb_tokens_response=nb_tokens_response
        mean_tokens+=nb_tokens
        mean_tokens_prompt+=nb_tokens_prompt
        mean_tokens_response+=nb_tokens_response
    mean_tokens=mean_tokens/len(exemples)
    mean_tokens_prompt=mean_tokens_prompt/len(exemples)
    mean_tokens_response=mean_tokens_response/len(exemples)
    dict_stats={
        "mean_tokens":mean_tokens,
        "mean_tokens_prompt":mean_tokens_prompt,
        "mean_tokens_response":mean_tokens_response,
        "max_nb_tokens":max_nb_tokens,
        "max_nb_tokens_prompt":max_nb_tokens_prompt,
        "max_nb_tokens_response":max_nb_tokens_response
    }
    results.append(dict_stats)
    return results

stats_train=stats_exemples(train)
stats_val=stats_exemples(val)

#Création des fichiers stats_train_sft.jsonl et stats_val_sft.jsonl
def create_file(path,liste):
    with open(path,"w",encoding="utf-8") as f:
        for elem in liste:
            f.write(json.dumps(elem,ensure_ascii=False) + '\n')

dir_test=os.path.join(dossier_script,"..","data","test")
os.makedirs(dir_test,exist_ok=True)
stats_train_sft_file=os.path.join(dossier_script,"..","data","test","stats_train_sft.jsonl")
stats_val_sft_file=os.path.join(dossier_script,"..","data","test","stats_val_sft.jsonl")

create_file(stats_train_sft_file,stats_train)
create_file(stats_val_sft_file,stats_val)
