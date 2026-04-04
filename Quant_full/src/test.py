from transformers import AutoModelForImageTextToText, BitsAndBytesConfig, AutoTokenizer
import torch

tokenizer=AutoTokenizer.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512-BF16")
model=AutoModelForImageTextToText.from_pretrained(
    "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    device_map="auto"
)

model_config={"max_new_tokens": 1500, "do_sample": False}
message=[{"role":"user","content":"Bonjour le chat"}]
inputs=tokenizer.apply_chat_template(
        message,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True
        ).to(model.device)
outputs=model.generate(**model_config,**inputs)
reponse=tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)

print(reponse)