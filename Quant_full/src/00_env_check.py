#Script qui sert à verifier l'environnement

import os
import torch


#Cartographier le CPU
print(f"La machine possède {os.cpu_count()} cœurs logiques (threads)")
#Pythorch arrive t'il à communiquer avec les coeurs CUDA du GPU 
if torch.cuda.is_available():
    print("La machine a bien un GPU")
    print(f"Le nom du GPU: {torch.cuda.get_device_name(0)}")
    print(f"La VRAM du GPU en GO est de: {torch.cuda.get_device_properties(0).total_memory/(1024**3)}")
else:
    print("Il n'ya pas de GPU ou Pythorch n'arrive pas à communiquer avec les coeurs CUDA du GPU")

#Vérifier les versions des bibliothèques importantes (Mise en place du bloc try pour savoir si la bibliothèque est bien installé)
try:
    import bitsandbytes
    print(f"Version de bitsandbytes: {bitsandbytes.__version__}")
except ImportError:
    print("ATTENTION : bitsandbytes n'est pas installé. La quantification Hugging Face échouera.")

try:
    import transformers
    print(f"Version de transformers :{transformers.__version__}")
except ImportError:
    print("ATTENTION : transformers n'est pas installé.")

try:
    import llama_cpp
    print(f"Version de llam_cpp: {llama_cpp.__version__}")
except ImportError:
    print("ATTENTION : llama_cpp n'est pas installé.")