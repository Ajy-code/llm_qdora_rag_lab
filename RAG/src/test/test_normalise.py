import pytest
import pathlib
from pydantic import ValidationError
import sys
import os
#Ajoute le dossier parent au chemin de recherche de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#Import des fonctions depuis le fichier principal 
from normalize_doc import (
    repair_layout,
    remove_universal_noise,
    NormalizedDocument
)

# --- TEST 1 : La réparation des césures (Couche 2) ---
def test_repair_layout_recolle_cesure():
    # Arrange : Un texte typique de PDF avec un mot coupé par un tiret et un retour à la ligne
    texte_brut = "Le développe-\nment du projet RAG est crucial."
    
    # Act : On passe le texte dans ton usine
    resultat = repair_layout(texte_brut)
    
    # Assert : On vérifie que le mot est parfaitement recollé sans tiret ni saut de ligne
    assert resultat == "Le développement du projet RAG est crucial."

# --- TEST 2 : Le filtrage du bruit universel (Couche 3) ---
def test_remove_universal_noise_nettoie_espaces_fin_ligne():
    # Arrange : Un texte avec 4 espaces invisibles juste avant le saut de ligne
    texte_brut = "Première ligne avec espaces.    \nDeuxième ligne."
    
    # Act
    resultat = remove_universal_noise(texte_brut)
    
    # Assert : On vérifie que les espaces ont disparu, mais que le \n est conservé
    assert resultat == "Première ligne avec espaces.\nDeuxième ligne."

# --- TEST 3 : Le contrat de données strict (Pydantic) ---
def test_normalized_document_refuse_texte_vide():
    # Arrange : On prépare des fausses métadonnées valides
    chemin_bidon = pathlib.Path("faux_document.pdf")
    
    # Act & Assert : Le bloc "with pytest.raises" dit à Pytest : 
    # "Attention, le code qui suit DOIT planter avec une ValidationError. S'il ne plante pas, c'est que le test échoue !"
    with pytest.raises(ValidationError):
        # On tente de créer un document avec un texte composé uniquement d'espaces et de sauts de ligne
        doc_invalide = NormalizedDocument(
            doc_id="id_123",
            content_hash="hash_456",
            source_path=chemin_bidon,
            text_content="   \n   " 
        )