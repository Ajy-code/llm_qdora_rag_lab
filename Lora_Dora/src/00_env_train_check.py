import os
import transformers
import torch

#SEED global pour la reproductibilité
SEED = 42
transformers.set_seed(SEED)
print(f"Seed globale configurée à : {SEED}")

#Vérification de la configuration du CPU/GPU
print(f"La version de torch est: {torch.__version__}")
print(f"Le nombre de thread est de: {os.cpu_count()}")

if torch.cuda.is_available():
    print("Le GPU est bien detecté par CUDA")
    print(f"Le nom du GPU est: {torch.cuda.get_device_name(0)}")
    print(f"La VRAM en GO du GPU est de: {torch.cuda.get_device_properties(0).total_memory/1024**3}")
    if torch.cuda.is_bf16_supported():
        print("Le GPU supporte nativement le format bfloat16 (bf16).")
    else:
        print("Attention : Le GPU ne supporte pas nativement le bf16. Il faudra utiliser fp16.")
else:
    print("Il n'y a pas de GPU ou CUDA n'arrive pas à détecter le GPU")

try:
    import bitsandbytes
    print(f"La version de bitsandbytes est: {bitsandbytes.__version__}")
except ImportError:
    print(f"La bibliothèque bitsandbytes n'est pas installé")

try:
    import transformers
    print(f"La version de transfomers est: {transformers.__version__}")
except ImportError:
    print(f"La bibliothèque transformers n'est pas installé")

try:
    import peft
    print(f"La version de peft est: {peft.__version__}")
except ImportError:
    print(f"La bibliothèque peft n'est pas installé")

try:
    import trl
    print(f"La version de trl est: {trl.__version__}")
except ImportError:
    print(f"La bibliothèque trl n'est pas installé")

try:
    import datasets
    print(f"La version de datasets est: {datasets.__version__}")
except ImportError:
    print(f"La bibliothèque datasets n'est pas installé")
