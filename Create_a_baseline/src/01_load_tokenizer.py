import sys
import torch
import transformers
from transformers import AutoTokenizer

#Familiarisation avec le tokenizer,l'outil qui permet de convertir les tokens en indice et vice versa
tokenizer=AutoTokenizer.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512-BF16")
#Permet d'avoir la taille du vocabulaire du modèle choisi
print(tokenizer.vocab_size)
# Testons l'encodage et le décodage
text="Hello World !!"
sequence=tokenizer.encode(text)
print(sequence)
# Juste pour voir le nombre de token qu'à le text
print(len(sequence))
# Verifier que sequence correspond bien à text
print(tokenizer.decode(sequence))
#Observation du comportement du tokenizer
texts=["L'ingénierie des LLMs est fascinante, n'est-ce pas ?","def somme(a, b):\n    return a + b",'{"nom": "Agent", "version": 1.0}']
for t in texts:
    print(t)
    print(tokenizer.encode(t))
    print(len(tokenizer.encode(t)))

