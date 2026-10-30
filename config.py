"""Central settings for the TN Agriculture RAG project.

Every other file imports its numbers and paths from here, so a value is
changed in one place only.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---- Files ---------------------------------------------------------------
PDF_PATH = BASE_DIR / "TN_Agriculture_Schemes.pdf"
INDEX_DIR = BASE_DIR / "faiss_index"
INDEX_META_PATH = INDEX_DIR / "index_meta.json"

# ---- Models --------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ---- Chunking ------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
# Pages with fewer English characters than this (after cleaning) are skipped.
# In this PDF they are cover pages and Tamil-only pages.
MIN_PAGE_CHARS = 80

# ---- Retrieval -----------------------------------------------------------
TOP_K = 4            # chunks sent to the LLM
MMR_FETCH_K = 20     # candidates fetched before MMR picks TOP_K
MMR_LAMBDA = 0.6     # 1.0 = pure similarity, 0.0 = maximum diversity
HISTORY_TURNS = 3    # earlier question/answer pairs given to the LLM

# ---- Answers -------------------------------------------------------------
NOT_FOUND_MESSAGE = "Information not available in the Agriculture Schemes document."

# ---- Scheme tags ---------------------------------------------------------
# Used as metadata on every chunk, and as a filter in the app.
# name -> regular-expression patterns (matched case-insensitively).
SCHEME_PATTERNS = {
    "KAVI-ADP": [r"KAVI[\s-]*ADP", r"Kalaignarin All Village"],
    "CM Mannuyir Kaathu (CM MK MKS)": [r"Mannuyir", r"CM\s*MK\s*MKS"],
    "CM Uzhavar Pathukappu (CMUPT)": [r"CMUPT", r"Uzhavar Pathukappu"],
    "PM-KISAN": [r"PM[\s-]*KISAN", r"Kisan Samman Nidhi"],
    "RKVY": [r"\bRKVY\b", r"Rashtriya Krishi Vikas"],
    "Food and Nutrition Security (FNS)": [r"\bFNS\b", r"Food and Nutrition Security"],
    "NMEO (Oilseeds & Oil Palm)": [r"\bNMEO\b", r"Edible Oils"],
    "Organic / Natural Farming (NMOF, NMNF)": [r"\bNMOF\b", r"\bNMNF\b", r"Organic Farming", r"Natural Farming"],
    "PKVY": [r"\bPKVY\b", r"Paramparagat Krishi"],
    "National Horticulture Mission (NHM)": [r"\bNHM\b", r"Horticulture Mission"],
    "Tamil Nadu Millet Mission": [r"Millet Mission"],
    "Palmyrah Development Mission": [r"Palmyrah"],
    "Integrated Coconut Development (ICDS)": [r"\bICDS\b", r"Coconut Development"],
    "Agricultural Mechanization (SMAM)": [r"\bSMAM\b", r"Mechani[sz]ation", r"e-Vaadagai"],
    "Solar Pumps / Solar Power": [r"Solar", r"TANGEDCO"],
    "Agriculture Extension (UATT, SMAE)": [r"\bUATT\b", r"\bSMAE\b", r"Agricultural Extension"],
    "Seed Certification & Quality": [r"Seed Certification", r"Seed Testing", r"Certified Seed"],
    "AGMARK": [r"AGMARK"],
    "Agriculture Infrastructure Fund (AIF)": [r"\bAIF\b", r"Agriculture Infrastructure Fund"],
    "eNAM": [r"\be-?NAM\b", r"National Agriculture Market"],
    "National Agriculture Development Project (NADP)": [r"\bNADP\b"],
    "PMKSY (Kisan SAMPADA / Krishi Sinchayee)": [r"\bPMKSY\b", r"Sinchayee", r"Kisan SAMPADA"],
    "Supply Chain Management (SCMP)": [r"\bSCMP\b", r"Supply Chain Management"],
    "TNIAMP (Irrigated Agriculture)": [r"TNIAMP", r"Irrigated Agriculture Modernization"],
    "Uzhavar Sandhai": [r"Uzhavar Sandhai"],
    "Rainfed / Dryland Development": [r"Rainfed", r"Dryland", r"Watershed"],
    "Crop Diversification (CDP)": [r"Crop Diversification", r"\bCDP\b"],
    "Agroforestry": [r"Agroforestry"],
    "Awards for Farmers": [r"\bAward", r"Prize"],
}
