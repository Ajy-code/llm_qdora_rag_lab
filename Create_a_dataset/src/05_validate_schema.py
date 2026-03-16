import os
import json
from pydantic import BaseModel, Field, ValidationError

dossier_script=os.path.dirname(os.path.abspath(__file__))
chemin_example=os.path.join(dossier_script,"..","data","raw","raw_examples.jsonl")

class SFTExample(BaseModel):
    # min_length=5 garantit qu'on rejette les textes trop courts ou vides
    instruction: str = Field(min_length=5)
    context: str = Field(min_length=10)
    response: str = Field(min_length=5)
    plan_id: str 
    challenge_type: str

examples_valides=[]
examples_rejetes=[]

with open(chemin_example,"r") as f:
    for example in f:
        try:
            dict_ex=json.loads(example)
            ex_valide=SFTExample(**dict_ex)
            examples_valides.append(dict_ex)
        except ValidationError as e:
            examples_rejetes.append(json.loads(example))

print(f"Le nombre d'examples valide est de : {len(examples_valides)}")
print(f"Le nombre d'examples rejetés est de : {len(examples_rejetes)}")

os.makedirs("data/interim",exist_ok=True)
chemin_ex_valides=os.path.join(dossier_script,"..","data","interim","validated_examples.jsonl")
with open(chemin_ex_valides,"w",encoding="utf-8") as f:
    for example in examples_valides:
        f.write(json.dumps(example,ensure_ascii=False) + "\n")

os.makedirs("data/rejects",exist_ok=True)
chemin_ex_rejetes=os.path.join(dossier_script,"..","data","rejects","rejected_examples.jsonl")
with open(chemin_ex_rejetes,"w",encoding="utf-8") as f:
    for example in examples_rejetes:
        f.write(json.dumps(example,ensure_ascii=False) + "\n")

