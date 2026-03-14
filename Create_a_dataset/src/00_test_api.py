# Objectif: Comprendre comment utiliser une clé API
from dotenv import load_dotenv
import os
import openai

#Lis le fichier .env et récupère les variables
load_dotenv()
API_KEY=os.getenv("API_KEY")
API_BASE_URL=os.getenv("API_BASE_URL")
MODEL_NAME=os.getenv("MODEL_NAME")

#client fait office d'interface entre "moi" et le modèle
client = openai.OpenAI(api_key=API_KEY,base_url=API_BASE_URL)

#Le message, avec un prompt system envoyé au modèle
mess=[{"role": "system", "content": "Sois le plus objectif possible"},
        {"role": "user", "content": "En quelque ligne, dis moi la diffeérence entre informatique quantique et informatique classique"},]

#La réponse du model, c'est un objet Pydantic
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=mess,
    temperature=0,
    stream=False
)

rep=response.choices[0].message.content
