#Vérification que le modèle convertit ( au format GGUF ) marche bien (sanity check)
from llama_cpp import Llama
import os

dossier_script=os.path.dirname(os.path.abspath(__file__))
MODEL_PATH=os.path.join(dossier_script,"..","models","gguf_ref","Ministral-3B-BF16.gguf")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    verbose=False
)

#Les prompts tests
sanity_prompts = [
    "Dis bonjour en une phrase.",
    "Donne-moi trois capitales européennes.",
    "Réponds uniquement par un JSON valide avec la clé 'answer'.",
    "Écris deux phrases puis arrête-toi."
]

print("Début du sanity check \n" )
for prompt in sanity_prompts:
    print(f"PROMPT: {prompt + "\n"}")
    reponse_brute=llm.create_chat_completion(
        messages=[{"role":"user","content":prompt}],
        max_tokens=150
    )
    response_text=reponse_brute["choices"][0]["message"]["content"]
    finish_reason = reponse_brute["choices"][0]["finish_reason"]
    print(f" RÉPONSE:\n{response_text}")
    print(f"RAISON D'ARRÊT : {finish_reason}")
    print("-" * 50)
    


