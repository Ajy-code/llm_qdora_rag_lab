# Objectif: Créer un taxonomie ( une grille de diversité ): pour s'assurer que le dataset ne sois pas uniforme

import os
import json

taxonomy_axes = {
    "domain_subarea": ["cloud_architecture", "cybersecurity_protocols", "database_administration"],
    "source_type": ["api_documentation", "release_notes", "user_manual"],
    "task_type": ["translate_en_to_fr", "translate_fr_to_en"],
    "difficulty": ["easy", "medium", "hard"],
    "noise_level": ["clean", "semi_noisy", "noisy"],
    "challenge_type": ["straightforward", "terminology_ambiguity", "false_friends", "mixed_code_text", "format_sensitive"],
    "output_format": ["plain_text", "preserve_source_format", "json"],
    "constraint_type": ["preserve_identifiers", "strict_json_keys"],
    "preservation_constraint": ["preserve_code_blocks", "preserve_markdown", "preserve_placeholders", "preserve_cli_commands"]
}
os.makedirs("data/taxonomy",exist_ok=True)

with open ("data/taxonomy/taxonomy_axes.json","w",encoding="utf-8") as f:
    json.dump(taxonomy_axes,f,indent=4,ensure_ascii=False)
