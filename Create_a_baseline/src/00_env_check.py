import sys
import torch
import transformers

print(sys.version)
print(torch.__version__)
if torch.cuda.is_available():
    print("Disponibilité CUDA")
    print(torch.cuda.get_device_name(0))
    print(f"La RAM du GPU (en GO) est de: {torch.cuda.get_device_properties(0).total_memory/(1024**3)}")
else:
    print("CUDA n'est pas disponible")
print(f"La version de bibliothèque transformers: {transformers.__version__}")
