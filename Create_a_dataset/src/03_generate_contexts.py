import json
import os
from dotenv import load_dotenv
import openai

# Calcule le chemin absolu du dossier où se trouve ton script '06_baseline.py' (le dossier 'src')
dossier_script = os.path.dirname(os.path.abspath(__file__))

# Construit le chemin absolu : src -> remonte (..) -> data -> taxonomy -> sampled_generation_plan.jsonl
chemin_fichier = os.path.join(dossier_script, "..", "data", "taxonomy","sampled_generation_plan.jsonl")

combinaisons=[]
with open(chemin_fichier,"r") as f:
    for combo in f:
        dict_combo=json.loads(combo)
        combinaisons.append(dict_combo)

liste_prompts=[]
for combo in combinaisons:
    # 1. Logique de langue (très importante)
    langue_source = "Anglais" if combo['task_type'] == "translate_en_to_fr" else "Français"

    # 2. Le prompt dynamique (f-string)
    prompt_contexte = f"""Tu es un Ingénieur IT Senior expert en {combo['domain_subarea']}.
    Ta mission est de rédiger un court extrait brut et très réaliste de type : {combo['source_type']}.

    RÈGLES STRICTES DE RÉDACTION :
        - Langue : Le texte doit être IMPÉRATIVEMENT et UNIQUEMENT rédigé en {langue_source}.
        - Difficulté technique : {combo['difficulty']}. Utilise le jargon professionnel approprié.
        - Bruit (noise_level) : {combo['noise_level']}. (Si "noisy", inclus des abréviations ambiguës ou un formatage imparfait. Si "clean", fais un texte parfait).
        - Piège sémantique (challenge_type) : Le texte doit absolument contenir la caractéristique suivante : {combo['challenge_type']}.

    CONTRAINTES DE STRUCTURE (Prépare le terrain pour une future traduction) :
        - Assure-toi que le texte inclut des éléments liés à la contrainte : {combo['preservation_constraint']} (ex: si c'est "preserve_cli_commands", inclus de vraies commandes terminal).
        - Le format implicite de ce document source doit être compatible avec une future extraction de type : {combo['output_format']}.

    FORMAT DE SORTIE :
    Génère UNIQUEMENT le document source brut. N'ajoute AUCUNE introduction, AUCUNE politesse, et AUCUN commentaire. Ne réponds pas "Voici le texte".
        """
    #Génération d'un dictionnaire par prompt (méthode pro)
    prompt_dict={}
    prompt_dict["plan_id"]=combo["plan_id"]
    prompt_dict["prompt_text"]=prompt_contexte
    liste_prompts.append(prompt_dict)

# Construit le chemin absolu : src -> remonte (..) -> data -> taxonomy -> prompt_generation.jsonl
chemin_fichier1 = os.path.join(dossier_script, "..", "data", "taxonomy","prompt_generation.jsonl")

#Enregistrer tous les prompts dans un fichier
with open(chemin_fichier1,"w",encoding="utf-8") as f:
    for prompt in liste_prompts:
        f.write(json.dumps(prompt,ensure_ascii=False) + "\n")

#Lis le fichier .env et récupère les variables
load_dotenv()
API_KEY=os.getenv("API_KEY")
API_BASE_URL=os.getenv("API_BASE_URL")
MODEL_NAME=os.getenv("MODEL_NAME")

#client fait office d'interface entre "moi" et le modèle
client = openai.OpenAI(api_key=API_KEY,base_url=API_BASE_URL)

#L'entrée (les contextes) qui sera donnée à MODEL_NAME pour avoir le couple entrée-sortie 
contexts=[]
for prompt_dict in liste_prompts[:2]:
    message=[{"role": "system", "content": "Tu es un générateur de données synthétiques strict. Tu ne réponds qu'avec le texte brut demandé, sans aucun formatage markdown autour si ce n'est pas demandé."},
        {"role":"user","content":prompt_dict["prompt_text"]}]
    #La réponse du model, c'est un objet Pydantic
    response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=message,
    temperature=0,
    stream=False
    )
    rep=response.choices[0].message.content
    dict_curr={}
    dict_curr["context_id"]=f"ctx_{prompt_dict['plan_id']}"
    #plan_id agit comme une clé étrangère ici
    dict_curr["plan_id"]=prompt_dict["plan_id"]
    dict_curr["context_raw"]=rep
    contexts.append(dict_curr)

# Construit le chemin absolu : src -> remonte (..) -> data -> raw -> raw_content.jsonl 
chemin_raw = os.path.join(dossier_script, "..", "data", "raw","raw_context.jsonl")
os.makedirs(os.path.dirname(chemin_raw), exist_ok=True)

#Enregistrer tous les contextes dans un fichier
with open(chemin_raw,"w",encoding="utf-8") as f:
    for contexte in contexts:
        f.write(json.dumps(contexte,ensure_ascii=False) + "\n")




