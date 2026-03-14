#Objectif: Créer un plan de génération des données

import os
import json
import itertools
import random
from collections import defaultdict

# Calcule le chemin absolu du dossier où se trouve ton script '06_baseline.py' (le dossier 'src')
dossier_script = os.path.dirname(os.path.abspath(__file__))

# Construit le chemin absolu : src -> remonte (..) -> data -> taxonomy -> taxonomy_axes.json
chemin_fichier = os.path.join(dossier_script, "..", "data", "taxonomy","taxonomy_axes.json")

with open(chemin_fichier,"r") as f:
    taxonomy_axes=json.load(f)

#Pour avoir toutes les "clès"( le nom des lignes ) de taxonomy_axes
taxo_keys=list(taxonomy_axes.keys())
combinaisons=list(itertools.product(*taxonomy_axes.values()))
combinaisons_brute=[]

#Associer clés et valeurs de combinaison
for combinaison in combinaisons:
    dict_curr=dict(zip(taxo_keys,combinaison))
    combinaisons_brute.append(dict_curr)
    
    
print(f"Total de combinaisons brutes générées : {len(combinaisons_brute)}")

#Il faut filtrer les combinaisons absurdes

def is_valid_combination(combo):
    # 1. Règle du JSON
    if combo["output_format"] == "json" and combo["constraint_type"] != "strict_json_keys":
        return False
    if combo["output_format"] != "json" and combo["constraint_type"] == "strict_json_keys":
        return False
        
    # 2. Règle du Texte brut
    if combo["output_format"] == "plain_text" and (combo["challenge_type"] == "format_sensitive" or combo["preservation_constraint"] == "preserve_markdown"):
        return False
        
    # 3. Règle de difficulté
    if combo["difficulty"] == "easy" and (combo["noise_level"] != "clean" or combo["challenge_type"] != "straightforward"):
        return False
    if combo["challenge_type"] == "straightforward" and combo["difficulty"] == "hard":
        return False
        
    # 4. Règle Source/Contrainte
    if combo["source_type"] == "user_manual" and (combo["challenge_type"] == "mixed_code_text" or combo["preservation_constraint"] == "preserve_cli_commands"):
        return False

    # Si la combinaison survit à tous les filtres, elle est valide
    return True

combinaisons_finale=[]
#Application du filtre
for comb in combinaisons_brute:
    if is_valid_combination(comb):
        #Avoir un id associé pour chaque combinaison finale
        comb["plan_id"] = f"plan_{len(combinaisons_finale):05d}"
        combinaisons_finale.append(comb)


print(f"Total de combinaisons brutes : {len(combinaisons_finale)}")

#Création des "groupes" de combinaison par challenge_type qui sont par défaut des listes vides
groupes = defaultdict(list)
for combo in combinaisons_finale:
    challenge_type=combo["challenge_type"]
    groupes[challenge_type].append(combo)

#Création de listes finale d'échantillon
liste_finale=[]
for groupe in groupes.values():
    liste_finale.extend(random.sample(groupe,min(20,len(groupe))))

print(f"Le nombre d'éléments de ma liste finale : {len(liste_finale)}")

# Construit le chemin absolu : src -> remonte (..) -> data -> taxonomy -> sampled_generation_plan.jsonl
chemin_fichier1 = os.path.join(dossier_script, "..", "data", "taxonomy","sampled_generation_plan.jsonl")

with open(chemin_fichier1,"w",encoding="utf-8") as f:
    #J'ajoute chaque combinaison de ma liste finale à mon fichier
    for combo in liste_finale:
        f.write(json.dumps(combo,ensure_ascii=False) + "\n")














