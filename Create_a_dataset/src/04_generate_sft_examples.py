import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
import openai
from pydantic import BaseModel, Field

#Lis le fichier .env et récupère les variables
load_dotenv()
API_KEY=os.getenv("API_KEY")
API_BASE_URL=os.getenv("API_BASE_URL")
MODEL_NAME=os.getenv("MODEL_NAME")

#client fait office d'interface entre "moi" et le modèle
client = openai.OpenAI(api_key=API_KEY,base_url=API_BASE_URL)

# Calcule le chemin absolu du dossier où se trouve ton script '06_baseline.py' (le dossier 'src')
dossier_script = os.path.dirname(os.path.abspath(__file__))
chemin_fichier1 = os.path.join(dossier_script,"..","data","taxonomy","sampled_generation_plan.jsonl")

#plan_info sera un dictionnaire ayant pour clé plan_id et pour valeur le "plan associé"(la combinaison de taxonomy axes)
plan_info={}
with open(chemin_fichier1,"r") as f:
    for line in f:
        #dict_plan est un dictionnaire associé à la line
        dict_plan=json.loads(line)
        plan_id=dict_plan.pop("plan_id","Introuvable")
        #J'associe "plan_id" à dict_plan
        plan_info[f"{plan_id}"]=dict_plan


chemin_fichier2 = os.path.join(dossier_script,"..","data","raw","raw_context.jsonl")
#raw_contexte contiendra tous les contexte, pour avoir la combinaison associé, il suffira de prendre la clé "plan_id" du contexte
#de prendre le dictionnaire (la valeur de la clé) associé dans plan_info
raw_context=[]
with open(chemin_fichier2,"r") as f:
    for line in f:
        dict_raw_context=json.loads(line)
        raw_context.append(dict_raw_context)
        
#Finalement parse impossible avec MODEL_NAME mais avoir en tête la possibilité d'utiliser des objet pydantics (en tant que moule)
class SFTExample(BaseModel):
    instruction: str = Field(description="L'instruction utilisateur (ex: 'Traduis ce texte en français' ou 'Résume ce log').")
    context: str = Field(description="Le document source exact fourni dans le prompt.")
    response: str = Field(description="La réponse finale de l'assistant (ex: la traduction ou le résumé).")

sftexample=[]
for dict_raw_ctx in tqdm(raw_context,desc="Génération des exemples"):
    #Combo est le dict de la combinaison associé à "plan_id"
    combo=plan_info[dict_raw_ctx["plan_id"]]
    contexte=dict_raw_ctx["context_raw"]

    prompt_system = f"""Tu es un Ingénieur Data expert en création de datasets d'entraînement (Supervised Fine-Tuning).
    Ta mission est de créer un exemple SFT parfait à partir d'un document source brut.

    Voici les paramètres imposés pour la création de cet exemple :
        - Type de tâche attendue : {combo['task_type']}
        - Défi technique présent : {combo['challenge_type']} (La réponse doit traiter cette spécificité avec précision).
        - Format de sortie exigé : {combo['output_format']} (L'instruction doit le réclamer, la réponse doit l'appliquer).
        - Contrainte métier stricte : {combo['constraint_type']} (La réponse doit impérativement respecter cette règle).
        - Élément à préserver : {combo['preservation_constraint']} (La réponse ne doit pas altérer ou traduire cet élément technique).

    Tu dois générer les 3 éléments suivants pour remplir le schéma JSON :

        1. "context" : Recopie EXACTEMENT et INTÉGRALEMENT le document source brut fourni ci-dessous, sans le modifier d'une seule lettre.
        2. "instruction" : Joue le rôle d'un utilisateur. Rédige une consigne claire (en français) demandant à l'assistant d'effectuer la tâche ({combo['task_type']}) sur le contexte. La consigne doit être naturelle et exiger expressément le format {combo['output_format']}.
        3. "response" : Joue le rôle de l'assistant IA parfait. Rédige la réponse finale en exécutant l'instruction sur le contexte. Tu DOIS surmonter le défi ({combo['challenge_type']}) et respecter strictement les contraintes ({combo['constraint_type']} et {combo['preservation_constraint']}).
    
    Tu DOIS renvoyer un objet JSON valide contenant EXACTEMENT et UNIQUEMENT ces trois clés : context, instruction, et response.    
    """
    message=[{"role":"system","content":prompt_system},
             {"role":"user","content":contexte}]
    response=client.chat.completions.create(
        model=MODEL_NAME,
        messages=message,
        response_format={"type": "json_object"}, # Pour avoir le bon format de sortie, un dictionnaire avec context,instruction,response
        temperature=0
    )
    #La réponse
    rep = response.choices[0].message.content
    #On le transforme en dictionnaire
    dict_triplet = json.loads(rep)
    dict_triplet["plan_id"]=dict_raw_ctx["plan_id"]
    dict_triplet["challenge_type"]=combo["challenge_type"]
    sftexample.append(dict_triplet)

chemin_example=os.path.join(dossier_script,"..","data","raw","raw_examples.jsonl")
with open(chemin_example,"w",encoding="utf-8") as f:
    for example in sftexample:
        f.write(json.dumps(example,ensure_ascii=False) + "\n")




