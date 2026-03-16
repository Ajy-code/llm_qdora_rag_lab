#Objectif: Faire un tableau de bord de mes exemples
import os 
import json
import collections

dossier_script=os.path.dirname(os.path.abspath(__file__))
chemin_train=os.path.join(dossier_script,"..","data","train","train.jsonl")

train=[]
with open(chemin_train,"r",encoding="utf-8") as f:
    for line in f:
        exemple=json.loads(line)
        train.append(exemple)

total_len_instruction=0
total_len_response=0
total_len_contexte=0
compteur_challenge = collections.Counter()
for exemple in train:
    total_len_instruction += len(exemple["instruction"])
    total_len_contexte += len(exemple["context"])
    total_len_response += len(exemple["response"])
    compteur_challenge.update([exemple["challenge_type"]])

mean_len_instruction=total_len_instruction/len(train)
mean_len_response=total_len_response/len(train)
mean_len_contexte=total_len_contexte/len(train)

for challenge_type, nb_exemples in compteur_challenge.items():
    print(f"{challenge_type} : {nb_exemples} exemples")

print(f"La longueur moyenne d'un contexte est de {mean_len_contexte}")
print(f"La longueur moyenne d'une instruction est de {mean_len_instruction}")
print(f"La longueur moyenne d'une réponse est de {mean_len_response}")
