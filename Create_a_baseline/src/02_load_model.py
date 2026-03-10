import torch
import gc
import os
# Ordre strict à PyTorch pour optimiser le rangement dans la VRAM
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Nettoyage agressif de la mémoire fantôme avant le lancement
torch.cuda.empty_cache()
gc.collect()

from transformers import AutoTokenizer
#Ministral 3 8B est un modele ImageTextToText
from transformers import AutoModelForImageTextToText,AutoModelForCausalLM
# BitsAndBytesConfig est la class qui permet de faire la quantififcation
from transformers import BitsAndBytesConfig

tokenizer=AutoTokenizer.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512-BF16")
#BitsAndBytesConfig réglage de la quantification 4bit
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
#Charge le model avec la quantization_config
model=AutoModelForImageTextToText.from_pretrained(
    "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    quantization_config=quantization_config,
    device_map="auto",
)

print(model.get_memory_footprint() / 1e9)