#Fais le mapping avec les bonnes fonctions en fonction du types
import layout 
LAYOUT_REPAIRERS = {
    "title": layout.repair_title_layout,
    "section_header": layout.repair_section_header_layout,
    "paragraph": layout.repair_paragraph_layout,
    "list_group": layout.repair_list_group_layout,
    "list_item": layout.repair_list_item_layout,
    "table": layout.repair_table_layout,
    "picture": layout.repair_picture_layout,
    "key_value": layout.repair_key_value_layout,
    "code": layout.repair_code_layout,
    "formula": layout.repair_formula_layout,
    "form": layout.repair_form_layout,
    "unknown": layout.repair_unknown_layout,
}

def get_layout_repairer(element_type: str):
    return LAYOUT_REPAIRERS.get(element_type, layout.repair_unknown_layout)