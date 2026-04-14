#Pour l'instant fonction universelle qui ne prend pas en compte le type
#Mais pour les prochaines versions du RAG, peut-être qu'il y aura modifications pour prendre en compte la normalization_unicode en fonction du type
#Nottament pour le code, ou les formules
import unicodedata
import logging

#Mise en place du logger pour pouvoir débugger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
    )
logger=logging.getLogger(__name__)

def normalize_unicode(text: str)-> str:
    if not text:
        return ""

    #Normalisation NFKC
    text_normalize_nfkc=unicodedata.normalize("NFKC", text)

    longueur_initiale_post_nfkc = len(text_normalize_nfkc)

    #Destruction des caractères parasites et standardisation
    text_clean_nfkc = (
        text_normalize_nfkc
        .replace("\u200B", "")  # Zero-width space
        .replace("\xAD", "")    # Soft hyphen
        .replace("\ufeff", "")  # BOM
        .replace("\xa0", " ")   # Espace insécable -> Espace normal
    )

    #Juste pour savoir le nombre de caractères qui a
    longueur_finale = len(text_clean_nfkc)
    if longueur_initiale_post_nfkc != longueur_finale:
        caracteres_supprimes = longueur_initiale_post_nfkc - longueur_finale
        logger.debug(f"Unicode: {caracteres_supprimes} caractères invisibles ou composites traités.")

    return text_clean_nfkc
