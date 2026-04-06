import os
import torch
from transformers import AutoTokenizer,AutoModelForImageTextToText,BitsAndBytesConfig,TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import time
import json

def eval_test_adaper(model,tokenizer,adapter_path,baseline,all_results,trainer_state):
    peft_model=PeftModel.from_pretrained(
        model,
        adapter_path
    )
    #Avoir la dernière loss du set d'entrainement
    train_last_loss = None
    for log in reversed(trainer_state["log_history"]):
        if "loss" in log:
            train_last_loss = log["loss"]
            break
    results_test={
        "train_last_loss":train_last_loss,
        "train_runtime":all_results["train_runtime"],
        "train_samples_per_second":all_results["train_samples_per_second"],
        "eval_loss":all_results["eval_loss"]
    }
    #Test qualitatif du model
    results_test["baseline_test"]=[]
    for test in baseline:
        model_config={"max_new_tokens": 1500, "do_sample": False}
        messages=[{"role":"user","content":test["prompt"]}]
        inputs=tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True
        ).to(peft_model.device)
        t=time.time()
        outputs=peft_model.generate(**inputs,**model_config)
        duree=time.time()-t
        dict_curr={
            "id":test["id"],
            "categorie":test["categorie"],
            "prompt":test["prompt"],
            "response":tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True),
            "duree":round(duree,2)
        }
        results_test["baseline_test"].append(dict_curr)
    peft_model.unload()
    return results_test

dossier_script=os.path.dirname(os.path.abspath(__file__))
baseline_file=os.path.join(dossier_script,"..","..","Create_a_baseline","inputs","baseline_prompt.json")
adapter_path_lora=os.path.join(dossier_script,"..","data","lora_sft","checkpoint-17")
adapter_path_dora=os.path.join(dossier_script,"..","data","dora_sft","checkpoint-17")
all_results_file_lora=os.path.join(dossier_script,"..","data","lora_sft","all_results.json")
all_results_file_dora=os.path.join(dossier_script,"..","data","dora_sft","all_results.json")
trainer_state_file_lora=os.path.join(dossier_script,"..","data","lora_sft","trainer_state.json")
trainer_state_file_dora=os.path.join(dossier_script,"..","data","dora_sft","trainer_state.json")

#Ouverture des différents fichiers
def open_json(path_json):
    with open(path_json,"r",encoding="utf-8") as f:
        result=json.load(f)
    return result

baseline=open_json(baseline_file)
all_results_lora=open_json(all_results_file_lora)
all_results_dora=open_json(all_results_file_dora)
trainer_state_lora=open_json(trainer_state_file_lora)
trainer_state_dora=open_json(trainer_state_file_dora)

#Création du modèle quantifié
tokenizer=AutoTokenizer.from_pretrained(
    "mistralai/Ministral-3-3B-Instruct-2512-BF16"
)
quantization_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model=AutoModelForImageTextToText.from_pretrained(
    "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    quantization_config=quantization_config,
    device_map="auto"
)

results_eval_lora=eval_test_adaper(
    model=model,
    tokenizer=tokenizer,
    baseline=baseline,
    adapter_path=adapter_path_lora,
    all_results=all_results_lora,
    trainer_state=trainer_state_lora
    )

results_eval_dora=eval_test_adaper(
    model=model,
    tokenizer=tokenizer,
    baseline=baseline,
    adapter_path=adapter_path_dora,
    all_results=all_results_dora,
    trainer_state=trainer_state_dora
    )

#Sauvegarde des évaluations
outputs_eval_dir=os.path.join(dossier_script,"..","eval")
os.makedirs(outputs_eval_dir,exist_ok=True)

eval_lora_file=os.path.join(dossier_script,"..","eval","eval_lora.json")
eval_dora_file=os.path.join(dossier_script,"..","eval","eval_dora.json")

def save_file(path_file,dict_save):
    with open(path_file,"w",encoding="utf-8") as f:
        json.dump(dict_save,f,indent=4,ensure_ascii=False)

save_file(eval_lora_file,results_eval_lora)
save_file(eval_dora_file,results_eval_dora)
