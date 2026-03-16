import os
import json
import random

# --- CONFIGURATION ---
NOMBRE_A_REVOIR = 5  # Change ce nombre si tu veux en lire plus ou moins
random.seed(42)      # Optionnel : pour relire les mêmes si tu relances le script

dossier_script = os.path.dirname(os.path.abspath(__file__))
chemin_train = os.path.join(dossier_script, "..", "data", "train", "train.jsonl")

# --- CHARGEMENT DES DONNÉES ---
train_data = []
with open(chemin_train, "r", encoding="utf-8") as f:
    for line in f:
        train_data.append(json.loads(line))

# --- ÉCHANTILLONNAGE ---
# On s'assure de ne pas demander plus d'exemples qu'il n'y en a dans le fichier
echantillon_taille = min(NOMBRE_A_REVOIR, len(train_data))
echantillon = random.sample(train_data, echantillon_taille)

# --- BOUCLE DE REVUE VISUELLE ---
print("\n" + "#" * 60)
print(f"LANCEMENT DE LA REVUE MANUELLE ({echantillon_taille} EXEMPLES)")
print("#" * 60 + "\n")

for i, exemple in enumerate(echantillon, 1):
    defi = exemple.get("challenge_type", "Non défini")
    
    print(f"\n[{i}/{echantillon_taille}] --- CATÉGORIE / DÉFI : {defi.upper()} ---")
    
    print("\n" + "-" * 20 + " INSTRUCTION " + "-" * 20)
    print(exemple.get("instruction", ""))
    
    print("\n" + "-" * 20 + " CONTEXTE " + "-" * 20)
    print(exemple.get("context", ""))
    
    print("\n" + "-" * 20 + " RÉPONSE (Générée) " + "-" * 20)
    print(exemple.get("response", ""))
    
    print("\n" + "=" * 60)
    
    # La pause tactique
    if i < echantillon_taille:
        input("👉 Appuie sur [ENTRÉE] pour voir l'exemple suivant (ou Ctrl+C pour quitter)...")
    else:
        print("✅ Revue terminée ! Tu as lu tous les échantillons.")
        