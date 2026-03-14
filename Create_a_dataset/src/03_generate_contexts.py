import json
import os
from dotenv import load_dotenv
import openai
from tqdm import tqdm

# Calcule le chemin absolu du dossier où se trouve ton script '06_baseline.py' (le dossier 'src')
dossier_script = os.path.dirname(os.path.abspath(__file__))

# Construit le chemin absolu : src -> remonte (..) -> data -> taxonomy -> sampled_generation_plan.jsonl
chemin_fichier = os.path.join(dossier_script, "..", "data", "taxonomy","sampled_generation_plan.jsonl")

combinaisons=[]
with open(chemin_fichier,"r") as f:
    for combo in f:
        dict_combo=json.loads(combo)
        combinaisons.append(dict_combo)

#Mappage pour créer un prompt de qualité (Prompt expansion)
difficulty_map = {
    "easy": "vocabulaire technique accessible, structure claire, phrases courtes à moyennes, faible ambiguïté",
    "medium": "vocabulaire technique normal, structure réaliste, quelques implicites ou dépendances contextuelles légères",
    "hard": "densité technique plus élevée, structure plus compacte ou spécialisée, ambiguïtés plausibles, lecture plus exigeante",
}

noise_map = {
    "clean": "texte propre, bien formaté, sans artefacts",
    "semi_noisy": "légères irrégularités réalistes : ponctuation imparfaite, ligne légèrement cassée, abréviations usuelles, petite incohérence de formatage",
    "noisy": "texte plus dégradé mais exploitable, avec plusieurs artefacts plausibles : abréviations ambiguës, structure partiellement cassée, alignement imparfait ou fragment incomplet",
}

challenge_map = {
    "straightforward": "texte direct, clair, sans piège lexical notable",
    "terminology_ambiguity": "inclure quelques termes techniques dont la traduction dépend du contexte",
    "false_friends": "inclure un ou deux faux amis techniques plausibles",
    "mixed_code_text": "mêler naturellement prose technique et éléments non narratifs comme commande, config ou snippet",
    "format_sensitive": "faire dépendre une partie du sens de la structure, du balisage, des puces ou du découpage",
}

preservation_map = {
    "preserve_code_blocks": "inclure au moins un bloc de code ou de configuration crédible",
    "preserve_markdown": "inclure une structure markdown réaliste",
    "preserve_placeholders": "inclure des placeholders plausibles comme {user_id}, <tenant_id>, $REGION",
    "preserve_cli_commands": "inclure au moins une commande terminal crédible",
}

liste_prompts=[]
for combo in combinaisons:
    # 1. Logique de langue (très importante)
    langue_source = "Anglais" if combo['task_type'] == "translate_en_to_fr" else "Français"

    # 2. Le prompt dynamique (f-string)
    prompt_contexte = f"""
        Tu es un ingénieur IT senior spécialiste de {combo['domain_subarea']}.

        Produis un document source brut, court, réaliste et professionnel de type {combo['source_type']}.

        Contraintes :
            - langue unique : {langue_source}
            - difficulté : {difficulty_map[combo['difficulty']]}
            - bruit : {noise_map[combo['noise_level']]}
            - défi de traduction : {challenge_map[combo['challenge_type']]}
            - contrainte de préservation : {preservation_map[combo['preservation_constraint']]}

        Règles :
            - texte crédible, spécifique, techniquement plausible
            - adapté naturellement au type de document demandé
            - pas scolaire, pas explicatif, pas artificiel
            - ne jamais mentionner les consignes
            - ne rien ajouter avant ni après le document
            - longueur cible : 80 à 180 mots

        Retourne uniquement le document source brut.
        """
    #Génération d'un dictionnaire par prompt (méthode pro)
    prompt_dict={}
    prompt_dict["plan_id"]=combo["plan_id"]
    prompt_dict["prompt_text"]=prompt_contexte
    liste_prompts.append(prompt_dict)

#Lis le fichier .env et récupère les variables
load_dotenv()
API_KEY=os.getenv("API_KEY")
API_BASE_URL=os.getenv("API_BASE_URL")
MODEL_NAME=os.getenv("MODEL_NAME")

#client fait office d'interface entre "moi" et le modèle
client = openai.OpenAI(api_key=API_KEY,base_url=API_BASE_URL)

#L'entrée (les contextes) qui sera donnée à MODEL_NAME pour avoir le couple entrée-sortie 
contexts=[]
for prompt_dict in tqdm(liste_prompts,desc="Génération des contextes"):
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




