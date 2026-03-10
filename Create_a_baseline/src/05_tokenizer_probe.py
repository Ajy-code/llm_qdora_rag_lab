import sys
import torch
import transformers
from transformers import AutoTokenizer

#Sonder le tokenizer, pour comprendre comment il fonctionne, ses spécificité
cas_a_tester = {
    "Phrase courte": "L'IA avance vite.",
    
    "Paragraphe long": "L'apprentissage automatique est un champ d'étude de l'intelligence artificielle qui se fonde sur des approches mathématiques et statistiques pour donner aux ordinateurs la capacité d'apprendre à partir de données. Il concerne la conception, l'analyse, l'optimisation et l'implémentation de telles méthodes.",
    
    "Français technique": "La rétropropagation du gradient (backpropagation) ajuste les poids du réseau de neurones en minimisant la fonction de perte via une descente de gradient stochastique.",
    
    "Code Python": "def calcul_moyenne(liste):\n    total = sum(liste)\n    return total / len(liste)",
    
    "Structure JSON": '{"modele": "Ministral-3B", "quantification": "8-bit", "parametres": [0.7, 0.9]}',
    
    "Liste à puces": "Étapes du RAG :\n- Extraction du texte\n- Chunking\n- Embedding vectoriel\n- Recherche sémantique",
    
    "Dates, montants, pourcentages": "Le 10 octobre 2025, le budget alloué au projet QDoRA est passé à 45 000,00 €, représentant une hausse stricte de 22.5%."
}
tokenizer=AutoTokenizer.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512-BF16")
for categorie,contenu in cas_a_tester.items():
    print(categorie)
    print(contenu)
    sequence=tokenizer.encode(contenu,add_special_tokens=False)
    print(f"Les premiers IDs sont {sequence[:5]}")
    print(f"Le nombre de IDs total:{len(sequence)}")
    print(tokenizer.decode(sequence))


