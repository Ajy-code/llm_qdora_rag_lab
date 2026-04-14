import os
import torch
import importlib.metadata

#Vérification de l'environnement
print(f"Le nombre de thread est de: {os.cpu_count()}")
if torch.cuda.is_available():
    print("La machine a bien un GPU")
    print(f"Le nom du GPU: {torch.cuda.get_device_name(0)}")
    print(f"La VRAM du GPU en GO: {torch.cuda.get_device_properties(0).total_memory/1024**3}")
    if torch.cuda.is_bf16_supported():
        print("Le GPU supporte le format bf16")
else:
    print("La machine n'a pas de GPU ou CUDA ne détecte pas le GPU")

try:
    import lancedb
    print(f"La version de la bibliothèque lancedb est: {lancedb.__version__}")
except ImportError:
    print(f"La bibliothèque lancedb n'est pas installé")

try:
    import sentence_transformers
    print(f"La version de la bibliothèque sentence_transformers est: {sentence_transformers.__version__}")
except ImportError:
    print(f"La bibliothèque sentence_transformers n'est pas installé")

try:
    import pypdf
    print(f"La version de la bibliothèque pypdf est: {pypdf.__version__}")
except ImportError:
    print(f"La bibliothèque pypdf n'est pas installé")

try:
    import pymupdf
    print(f"La version de la bibliothèque pymupdf est: {pymupdf.__version__}")
except ImportError:
    print(f"La bibliothèque pymupdf n'est pas installé")

try:
    import flashrank
    # On demande la version aux métadonnées de l'environnement virtuel (.venv)
    version_flashrank = importlib.metadata.version("flashrank")
    print(f"La version de la bibliothèque flashrank est: {version_flashrank}")
except ImportError:
    print("La bibliothèque flashrank n'est pas installée.")
except importlib.metadata.PackageNotFoundError:
    print("Le package flashrank est introuvable dans les métadonnées.")

#Vérification du runtime Python natif (Transformers)
try:
    import transformers
    print(f"Runtime HF (transformers) : Installé (version {transformers.__version__})")
except ImportError:
    print("Runtime HF (transformers) : Non installé")

#Vérification du runtime C++ léger (Llama.cpp Python)
try:
    import llama_cpp
    # llama_cpp_python n'a pas toujours de __version__ standard, on vérifie juste son import
    print("Runtime llama.cpp (Python) : Installé")
except ImportError:
    print("Runtime llama.cpp (Python) : Non installé")