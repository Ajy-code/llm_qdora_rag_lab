#Là on va utiliser le LLM as a judge, ça sera notre filtre mais dans la réalité il faudrait au moins filtrer par ouput_formats avec un algo determinisite

import os
import openai
from dotenv import load_dotenv
import json
from tqdm import tqdm

load_dotenv()
API_KEY=os.getenv("API_KEY")
API_BASE_URL=os.getenv("API_BASE_URL")
MODEL_NAME=os.getenv("MODEL_NAME")

client=openai.OpenAI(api_key=API_KEY,base_url=API_BASE_URL)

dossier_script=os.path.dirname(os.path.abspath(__file__))
chemin_SFTExample=os.path.join(dossier_script,"..","data","interim","validated_examples.jsonl")

#SFTExample aura tous les bons examples
SFTExample=[]
with open(chemin_SFTExample,"r") as f:
    for example in f:
        SFTExample.append(json.loads(example))

chemin_plan = os.path.join(dossier_script,"..","data","taxonomy","sampled_generation_plan.jsonl")
#plan_info sera un dictionnaire ayant pour clé plan_id et pour valeur le "plan associé"(la combinaison de taxonomy axes)
plan_info={}
with open(chemin_plan,"r") as f:
    for line in f:
        #dict_plan est un dictionnaire associé à la line
        dict_plan=json.loads(line)
        plan_id=dict_plan.pop("plan_id","Introuvable")
        #J'associe "plan_id" à dict_plan
        plan_info[f"{plan_id}"]=dict_plan

exemple_accepted=[]
exemple_review=[]
exemple_rejectsLLM=[]

prompt_system="""Tu es un Lead QA Engineer et un Expert en Localisation Technique (Cloud, Cyber, DBA). 
Ta mission est d'auditer de manière stricte des triplets d'entraînement (Instruction, Contexte Source, Réponse) destinés à affiner un modèle de traduction.

Tu recevras les métadonnées exactes qui ont dicté la création de cet exemple. Tu dois évaluer la "Réponse" selon la grille suivante. Chaque critère est noté sur 5.

GRILLE D'ÉVALUATION :

1. Précision Sémantique et Domaine (Score sur 5)
- La traduction (task_type) est-elle exacte ? 
- Le vocabulaire est-il parfaitement adapté au domaine (domain_subarea) et au type de source (source_type) ?
- Le niveau de technicité respecte-t-il la difficulté exigée (difficulty) ? Si "hard", le jargon doit être pointu.

2. Gestion des Défis et du Bruit (Score sur 5)
- Le modèle a-t-il surmonté le défi spécifié (challenge_type) comme les "false_friends" ou l'ambiguïté ("terminology_ambiguity") ?
- Le modèle a-t-il géré correctement le niveau de bruit de la source (noise_level) ? Si la source est "noisy" ou "semi_noisy", la traduction doit refléter le sens technique sans halluciner d'informations pour "réparer" le texte.

3. Préservation et Contraintes Absolues (Score sur 5)
- 5 : Respect total.
- 1 : ÉCHEC FATAL. Si un élément de la contrainte de préservation (preservation_constraint) a été altéré ou traduit (ex: code_blocks, cli_commands, placeholders) OU si la contrainte structurelle (constraint_type) n'est pas respectée (ex: identifiers traduits, clés JSON non strictes), la note DOIT être 1.

4. Format de Sortie (Score sur 5)
- Le format cible (output_format) est-il rigoureusement respecté (ex: plain_text, json, preserve_source_format) ?

FORMAT DE TA RÉPONSE :
Tu dois impérativement répondre par un objet JSON valide, respectant cette structure exacte :

{
  "reasoning": {
    "domain_and_accuracy": "Bref commentaire...",
    "challenge_and_noise": "Bref commentaire...",
    "preservation_and_format": "Bref commentaire..."
  },
  "scores": {
    "accuracy_score": <int>,
    "challenge_score": <int>,
    "preservation_score": <int>,
    "format_score": <int>
  },
  "final_decision": "<ACCEPT | REJECT | REVIEW>"
}

RÈGLES DE DÉCISION :
- ACCEPT : Tous les scores sont de 4 ou 5.
- REVIEW : Au moins un score est de 3, et aucun n'est inférieur.
- REJECT : Au moins un score est de 1 ou 2."""

for exemple in tqdm(SFTExample, desc="Audit LLM en cours"):
    plan_id=exemple["plan_id"]
    combo=plan_info[plan_id]
    

    user_prompt = f"""
AUDIT D'UN NOUVEL EXEMPLE SFT.

=== PARAMÈTRES DE LA TAXONOMIE ===
- Domaine Technique : {combo['domain_subarea']}
- Type de Source : {combo['source_type']}
- Tâche de traduction : {combo['task_type']}
- Difficulté technique : {combo['difficulty']}
- Niveau de bruit de la source : {combo['noise_level']}
- Défi linguistique/technique : {combo['challenge_type']}
- Format de sortie exigé : {combo['output_format']}
- Type de contrainte globale : {combo['constraint_type']}
- Éléments à préserver ABSOLUMENT : {combo['preservation_constraint']}

=== DONNÉES À ÉVALUER ===
[INSTRUCTION]
{exemple['instruction']}

[CONTEXTE SOURCE]
{exemple['context']}

[RÉPONSE GÉNÉRÉE (À JUGER)]
{exemple['response']}

Analyse la [RÉPONSE GÉNÉRÉE] en fonction des 9 paramètres de la taxonomie ci-dessus et retourne ton verdict en JSON strict.
"""
    try: 
        message=[{"role":"system","content":prompt_system},{"role":"user","content":user_prompt}]
        #Envoie des prompt au model
        response=client.chat.completions.create(
            model=MODEL_NAME,
            messages=message,
            response_format={"type": "json_object"}, # Pour avoir le bon format de sortie, un dictionnaire avec context,instruction,response
            temperature=0
        )
        rep = response.choices[0].message.content
        #Transformation de la reponse en dictionnaire
        rep_dict=json.loads(rep)
        #Pour avoir l'avis du juge
        exemple["llm_judge_review"] = rep_dict
        decision = rep_dict.get("final_decision", "REVIEW") # Sécurité si la clé saute
        
        if decision == "ACCEPT":
            exemple_accepted.append(exemple) # On ajoute l'exemple SFT, pas juste l'avis !
        elif decision == "REJECT":
            exemple_rejectsLLM.append(exemple)
        else:
            exemple_review.append(exemple)
    except Exception as e:
        #Avoir l'avis du juge
        exemple["llm_judge_error"] = str(e)
        exemple_review.append(exemple)


chemin_accept = os.path.join(dossier_script, "..", "data", "interim", "quality_passed_examples.jsonl")
chemin_review = os.path.join(dossier_script, "..", "data", "interim", "to_review_examples.jsonl")
chemin_reject = os.path.join(dossier_script, "..", "data", "rejects", "rejected_quality.jsonl")

def save_to_jsonl(data_list, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

save_to_jsonl(exemple_accepted, chemin_accept)
save_to_jsonl(exemple_review, chemin_review)
save_to_jsonl(exemple_rejectsLLM, chemin_reject)

print(f"\nAudit terminé ! Acceptés: {len(exemple_accepted)}, À revoir: {len(exemple_review)}, Rejetés: {len(exemple_rejectsLLM)}")

