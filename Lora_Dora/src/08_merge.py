#Merge le méthode pour aprés faire la conversion au format GGUF
import os
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForImageTextToText
import gc

dossier_script=os.path.dirname(os.path.abspath(__file__))
adapter_lora_path=os.path.join(dossier_script,"..","data","lora_sft","checkpoint-17")
adapter_dora_path=os.path.join(dossier_script,"..","data","dora_sft","checkpoint-17")
save_path_lora = os.path.join(dossier_script, "..", "model", "model_lora")
save_path_dora = os.path.join(dossier_script, "..", "model", "model_dora")

os.makedirs(save_path_lora, exist_ok=True)
os.makedirs(save_path_dora, exist_ok=True)

tokenizer=AutoTokenizer.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512-BF16")


def fusionner_et_sauvegarder(adapter_path, save_path):
    #Charger la base vierge en bfloat16
    model = AutoModelForImageTextToText.from_pretrained(
        "mistralai/Ministral-3-3B-Instruct-2512-BF16",
        dtype=torch.bfloat16,
        device_map="cpu"
    )
    
    #Greffer l'adapteur
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    
    #Fusion
    modele_fusionne = peft_model.merge_and_unload()
    
    #Sauvegarder le modèle et le tokenizer
    modele_fusionne.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    #Nettoyer la mémoire GPU pour le prochain tour
    del model
    del peft_model
    del modele_fusionne
    torch.cuda.empty_cache()
    gc.collect()
    print("Fusion terminée et mémoire libérée.")

# Lancement des merges
fusionner_et_sauvegarder(adapter_lora_path, save_path_lora)
fusionner_et_sauvegarder(adapter_dora_path, save_path_dora)