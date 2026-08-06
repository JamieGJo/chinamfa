"""
make_data.py — generate JSON data files for websites/chinamfa/
Run from the project root or from websites/chinamfa/.

Source: data/raw-extracts/MFA/CMFA_PressCon_master.csv (canonical master)
Outputs (all to websites/chinamfa/data/):
  country_counts.json         — per-country per-year mention counts (2002–2026)
  spokesperson_by_year.json   — stacked bar timeline
  spokesperson_cards.json     — per-spokesperson stat cards
  press_breakdown.json        — who_asked categories + confrontation rates
  topics.json                 — top orgs in questions + top countries
  articles.json               — IO + confrontation rows for article explorer

Methodology for country_counts:
  2002–2023 (CMFA_v4): uses the NLP-extracted a_loc field (with US→United States
    normalisation), which is the same source as the original country_counts.json.
  2024–2026 (scraped): a_loc was never populated for these rows, so we use
    simple keyword matching on question + answer text. Results are comparable
    in magnitude but not strictly equivalent to NLP extraction.
"""

import json, os, re
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# Allow running from project root or from websites/chinamfa/
if not os.path.isdir(os.path.join(PROJECT_ROOT, "data")):
    PROJECT_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "..")

CSV_PATH = os.path.join(
    SCRIPT_DIR, "../../data/raw-extracts/MFA",
    "CMFA_PressCon_master.csv"
)
OUT_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load corpus ───────────────────────────────────────────────────────────────
print("Loading corpus …")
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"  Loaded {len(df):,} rows")

# Parse year for scraped 2024-2026 rows (year column is NaN for those)
df["year_clean"] = df["year"].fillna(
    pd.to_datetime(df["date"], errors="coerce").dt.year
).astype("Int64")

# ── 0: Country counts (rebuild extending to 2026) ─────────────────────────────
print("Building country_counts …")

# Canonical name → list of text aliases for keyword matching
COUNTRY_ALIASES = {
    "United States":    [r"\bUS\b", r"\bUSA\b", r"\bAmerica\b", r"\bUnited States\b"],
    "China":            [r"\bChina\b", r"\bChinese\b"],
    "Japan":            [r"\bJapan\b", r"\bJapanese\b"],
    "Taiwan":           [r"\bTaiwan\b"],
    "North Korea":      [r"\bNorth Korea\b", r"\bDPRK\b"],
    "Russia":           [r"\bRussia\b", r"\bRussian\b"],
    "South Korea":      [r"\bSouth Korea\b", r"\bROK\b"],
    "India":            [r"\bIndia\b", r"\bIndian\b"],
    "United Kingdom":   [r"\bUnited Kingdom\b", r"\bUK\b", r"\bBritain\b", r"\bBritish\b"],
    "Ukraine":          [r"\bUkraine\b", r"\bUkrainian\b"],
    "Pakistan":         [r"\bPakistan\b"],
    "Philippines":      [r"\bPhilippines\b", r"\bFilipino\b"],
    "Australia":        [r"\bAustralia\b", r"\bAustralian\b"],
    "Germany":          [r"\bGermany\b", r"\bGerman\b"],
    "France":           [r"\bFrance\b", r"\bFrench\b"],
    "Iran":             [r"\bIran\b", r"\bIranian\b"],
    "Afghanistan":      [r"\bAfghanistan\b", r"\bAfghan\b"],
    "Myanmar":          [r"\bMyanmar\b", r"\bBurma\b"],
    "Canada":           [r"\bCanada\b", r"\bCanadian\b"],
    "Israel":           [r"\bIsrael\b", r"\bIsraeli\b"],
    "Palestine":        [r"\bPalestine\b", r"\bPalestinian\b", r"\bGaza\b"],
    "Hong Kong":        [r"\bHong Kong\b"],
    "Xinjiang":         [r"\bXinjiang\b", r"\bUyghur\b"],
    "Tibet":            [r"\bTibet\b", r"\bTibetan\b"],
    "Syria":            [r"\bSyria\b", r"\bSyrian\b"],
    "Iraq":             [r"\bIraq\b", r"\bIraqi\b"],
    "Libya":            [r"\bLibya\b", r"\bLibyan\b"],
    "Vietnam":          [r"\bVietnam\b", r"\bVietnamese\b"],
    "Indonesia":        [r"\bIndonesia\b", r"\bIndonesian\b"],
    "Brazil":           [r"\bBrazil\b", r"\bBrazilian\b"],
    "Saudi Arabia":     [r"\bSaudi Arabia\b", r"\bSaudi\b"],
    "Turkey":           [r"\bTurkey\b", r"\bTurkish\b", r"\bTürkiye\b"],
    "Egypt":            [r"\bEgypt\b", r"\bEgyptian\b"],
    "South Africa":     [r"\bSouth Africa\b", r"\bSouth African\b"],
    "Sri Lanka":        [r"\bSri Lanka\b"],
    "Nepal":            [r"\bNepal\b", r"\bNepalese\b"],
    "Cambodia":         [r"\bCambodia\b", r"\bCambodian\b"],
    "Venezuela":        [r"\bVenezuela\b", r"\bVenezuelan\b"],
    "Cuba":             [r"\bCuba\b", r"\bCuban\b"],
    "Sudan":            [r"\bSudan\b", r"\bSudanese\b"],
    "Nigeria":          [r"\bNigeria\b", r"\bNigerian\b"],
    "Ethiopia":         [r"\bEthiopia\b", r"\bEthiopian\b"],
    "Kenya":            [r"\bKenya\b", r"\bKenyan\b"],
    "Lithuania":        [r"\bLithuania\b", r"\bLithuanian\b"],
    "Poland":           [r"\bPoland\b", r"\bPolish\b"],
    "Kazakhstan":       [r"\bKazakhstan\b", r"\bKazakh\b"],
    "Uzbekistan":       [r"\bUzbekistan\b"],
    "Belarus":          [r"\bBelarus\b", r"\bBelarusian\b"],
    "Serbia":           [r"\bSerbia\b", r"\bSerbian\b"],
    "Hungary":          [r"\bHungary\b", r"\bHungarian\b"],
    "Singapore":        [r"\bSingapore\b"],
    "Malaysia":         [r"\bMalaysia\b", r"\bMalaysian\b"],
    "Thailand":         [r"\bThailand\b", r"\bThai\b"],
    "Mongolia":         [r"\bMongolia\b", r"\bMongolian\b"],
    "Laos":             [r"\bLaos\b", r"\bLao\b"],
    "Maldives":         [r"\bMaldives\b"],
    "Solomon Islands":  [r"\bSolomon Islands\b"],
    "Papua New Guinea": [r"\bPapua New Guinea\b"],
    "Fiji":             [r"\bFiji\b"],
    "UAE":              [r"\bUAE\b", r"\bUnited Arab Emirates\b"],
    "Qatar":            [r"\bQatar\b", r"\bQatari\b"],
    "Jordan":           [r"\bJordan\b", r"\bJordanian\b"],
    "Lebanon":          [r"\bLebanon\b", r"\bLebanese\b"],
    "Yemen":            [r"\bYemen\b", r"\bYemeni\b"],
    "Morocco":          [r"\bMorocco\b", r"\bMoroccan\b"],
    "Algeria":          [r"\bAlgeria\b", r"\bAlgerian\b"],
    "Panama":           [r"\bPanama\b"],
    "DR Congo":         [r"\bDR Congo\b", r"\bDRC\b", r"\bDemocratic Republic of Congo\b"],
    "Zimbabwe":         [r"\bZimbabwe\b"],
    "Angola":           [r"\bAngola\b"],
    "Bangladesh":       [r"\bBangladesh\b"],
    "Italy":            [r"\bItaly\b", r"\bItalian\b"],
    "Spain":            [r"\bSpain\b", r"\bSpanish\b"],
    "Netherlands":      [r"\bNetherlands\b", r"\bDutch\b"],
    "Sweden":           [r"\bSweden\b", r"\bSwedish\b"],
    "Norway":           [r"\bNorway\b", r"\bNorwegian\b"],
    "Denmark":          [r"\bDenmark\b", r"\bDanish\b"],
    "Finland":          [r"\bFinland\b", r"\bFinnish\b"],
    "Czech Republic":   [r"\bCzech Republic\b", r"\bCzech\b", r"\bCzechia\b"],
    "Poland":           [r"\bPoland\b", r"\bPolish\b"],
    "Greece":           [r"\bGreece\b", r"\bGreek\b"],
    "Portugal":         [r"\bPortugal\b", r"\bPortuguese\b"],
    "Switzerland":      [r"\bSwitzerland\b", r"\bSwiss\b"],
    "Austria":          [r"\bAustria\b", r"\bAustrian\b"],
    "Belgium":          [r"\bBelgium\b", r"\bBelgian\b"],
    "Ireland":          [r"\bIreland\b", r"\bIrish\b"],
    "New Zealand":      [r"\bNew Zealand\b"],
    "Mexico":           [r"\bMexico\b", r"\bMexican\b"],
    "Argentina":        [r"\bArgentina\b", r"\bArgentine\b"],
    "Colombia":         [r"\bColombia\b", r"\bColombian\b"],
    "Chile":            [r"\bChile\b", r"\bChilean\b"],
    "Peru":             [r"\bPeru\b", r"\bPeruvian\b"],
    "Kosovo":           [r"\bKosovo\b"],
    "Macao":            [r"\bMacao\b", r"\bMacau\b"],
    "Azerbaijan":       [r"\bAzerbaijan\b"],
    "Armenia":          [r"\bArmenia\b", r"\bArmenian\b"],
}

# Compile patterns
COUNTRY_PATTERNS = {
    name: re.compile("|".join(aliases))
    for name, aliases in COUNTRY_ALIASES.items()
}

def text_country_mentions(text):
    """Return set of country names mentioned in text."""
    hits = set()
    if not isinstance(text, str) or not text.strip():
        return hits
    for name, pat in COUNTRY_PATTERNS.items():
        if pat.search(text):
            hits.add(name)
    return hits

# Split corpus: original NLP-extracted vs scraped (text-based)
orig_rows = df[df["source"] == "CMFA_v4"].copy()
scraped_rows = df[df["source"] == "scraped_2024_2026"].copy()
scraped_rows["year_clean"] = pd.to_datetime(scraped_rows["date"], errors="coerce").dt.year.astype("Int64")

# 2002–2023: rebuild from a_loc using same methodology as original
# (a_loc field, normalise 'US' → 'United States', count per row per year)
ALOC_REMAP = {"US": "United States", "ROK": "South Korea", "DPRK": "North Korea",
              "UK": "United Kingdom", "HK": "Hong Kong", "UAE": "UAE"}

print("  processing CMFA_v4 rows via a_loc …")
orig_counts = {}  # {country: {year: count}}
for _, row in orig_rows.iterrows():
    yr = row["year"]
    if pd.isna(yr):
        continue
    yr = int(yr)
    locs = [l.strip() for l in str(row["a_loc"]).split(";")]
    locs = [ALOC_REMAP.get(l, l) for l in locs]
    locs = [l for l in locs if l and l not in ("-", "nan", "")]
    for loc in set(locs):  # set to avoid double-count within a row
        orig_counts.setdefault(loc, {}).setdefault(yr, 0)
        orig_counts[loc][yr] += 1

# 2024–2026: text-based extraction from question + answer
print("  processing scraped rows via text matching …")
scraped_counts = {}
for _, row in scraped_rows.iterrows():
    yr = row["year_clean"]
    if pd.isna(yr):
        continue
    yr = int(yr)
    text = str(row.get("question", "")) + " " + str(row.get("answer", ""))
    for country in text_country_mentions(text):
        scraped_counts.setdefault(country, {}).setdefault(yr, 0)
        scraped_counts[country][yr] += 1

# Canonical country names that have ISO3 codes and can be rendered on the map
# (must match the ISO_MAP keys in index.html exactly)
MAP_COUNTRIES = {
    "United States","Japan","Taiwan","North Korea","Russia","Hong Kong","India",
    "Pakistan","Xinjiang","Philippines","United Kingdom","Afghanistan","Iran",
    "Ukraine","Australia","South Korea","Germany","France","Indonesia","Palestine",
    "Syria","Iraq","Canada","Israel","Myanmar","Sri Lanka","Vietnam","Libya",
    "Sudan","Egypt","Saudi Arabia","Turkey","Brazil","South Africa","Cuba",
    "Venezuela","Nepal","Cambodia","Singapore","Thailand","Malaysia","Italy",
    "Spain","Nigeria","Kenya","Ethiopia","Zimbabwe","Angola","DR Congo","Serbia",
    "Panama","Bangladesh","Mongolia","Laos","Uzbekistan","Hungary",
    "Solomon Islands","Fiji","Tonga","Vanuatu","Papua New Guinea","Maldives",
    "Bhutan","UAE","Qatar","Kuwait","Jordan","Lebanon","Yemen","Morocco",
    "Tunisia","Algeria","Belarus","Azerbaijan","Armenia","Lithuania","Poland",
    "Czech Republic","Sweden","Finland","Norway","Denmark","Netherlands",
    "Belgium","Portugal","Greece","Austria","Switzerland","Ireland","Croatia",
    "Seychelles","Albania","Niger","Kosovo","Macao","Tibet","China",
    # extra a_loc values that have ISO codes and are informative
    "New Zealand","Mexico","Argentina","Colombia","Chile","Peru",
}

# Merge — only keep MAP_COUNTRIES entries
merged = {}
for country in MAP_COUNTRIES:
    yr_data = {}
    for yr, cnt in orig_counts.get(country, {}).items():
        yr_data[f"{float(yr):.1f}"] = cnt
    for yr, cnt in scraped_counts.get(country, {}).items():
        yr_data[str(yr)] = cnt
    if yr_data:
        merged[country] = yr_data

all_years_orig = sorted({int(float(y)) for c in orig_counts.values() for y in c})
all_years_scraped = sorted({y for c in scraped_counts.values() for y in c})
all_years = sorted(set(all_years_orig) | set(all_years_scraped))

country_counts_out = {
    "years": [f"{float(y):.1f}" if y <= 2023 else str(y) for y in all_years],
    "countries": merged,
}

with open(os.path.join(OUT_DIR, "country_counts.json"), "w") as f:
    json.dump(country_counts_out, f)
print(f"  wrote country_counts.json ({len(merged)} countries, years {all_years[0]}–{all_years[-1]})")

# ── Spokesperson palette ──────────────────────────────────────────────────────
SP_COLOR = {
    "Kong Quan":      "#5e4fa2",
    "Zhang Qiyue":    "#9e78c6",
    "Liu Jianchao":   "#3288bd",
    "Qin Gang":       "#66c2a5",
    "Jiang Yu":       "#abdda4",
    "Ma Zhaoxu":      "#e6f598",
    "Liu Weimin":     "#fee08b",
    "Hong Lei":       "#fdae61",
    "Geng Shuang":    "#f46d43",
    "Lu Kang":        "#d53e4f",
    "Hua Chunying":   "#9e0142",
    "Zhao Lijian":    "#c0291a",
    "Wang Wenbin":    "#2166ac",
    "Mao Ning":       "#4dac26",
    "Lin Jian":       "#006d2c",
    "Guo Jiakun":     "#08519c",
}

# ── A: Spokesperson by year ───────────────────────────────────────────────────
print("Building spokesperson_by_year …")

TOP_SP = (
    df["spokesperson"]
    .value_counts()
    .head(16)
    .index.tolist()
)
# exclude 'missing' and '-' placeholders
TOP_SP = [s for s in TOP_SP if s not in ("missing", "-")]

# Sort by most recent year active (descending) so chart legend and cards read newest → oldest
_sp_year_max = (
    df[df["spokesperson"].isin(TOP_SP)]
    .groupby("spokesperson")["year_clean"]
    .max()
)
TOP_SP = sorted(TOP_SP, key=lambda sp: int(_sp_year_max.get(sp, 0)), reverse=True)

sp_yr = (
    df[df["spokesperson"].isin(TOP_SP)]
    .groupby(["year_clean", "spokesperson"])
    .size()
    .reset_index(name="n")
)

years = sorted(sp_yr["year_clean"].dropna().unique().tolist())
sp_by_year = {"years": [int(y) for y in years], "series": []}

for sp in TOP_SP:
    sub = sp_yr[sp_yr["spokesperson"] == sp].set_index("year_clean")["n"]
    counts = [int(sub.get(y, 0)) for y in years]
    sp_by_year["series"].append({
        "name": sp,
        "color": SP_COLOR.get(sp, "#999"),
        "data": counts,
    })

with open(os.path.join(OUT_DIR, "spokesperson_by_year.json"), "w") as f:
    json.dump(sp_by_year, f)
print(f"  wrote spokesperson_by_year.json ({len(years)} years, {len(TOP_SP)} spokespeople)")

# ── B: Spokesperson cards ─────────────────────────────────────────────────────
print("Building spokesperson_cards …")

io_mask = df["answer"].astype(str).str.contains("international order", case=False, na=False)
conf_cols = ["Threat", "Demand", "Urge", "Request", "Representations",
             "Active threat", "China attacked"]

# Biographical data — static, curated
SP_BIO = {
    "Kong Quan": {
        "era": "Hu Jintao",
        "post": "Ambassador to France & Monaco (2008–13); Ambassador to EU (2013–)",
        "note": "Born 1955, New Delhi (father posted as diplomat). 76th-generation descendant of Confucius; ancestral home Qufu, Shandong. Joined MFA 1977. Director-General, MFA Information Dept (2002–06). Director-General, Dept of European Affairs. Assistant Foreign Minister. Ambassador to France & Monaco (2008–13). Ambassador to EU, Brussels (2013–).",
    },
    "Zhang Qiyue": {
        "era": "Hu Jintao",
        "post": "Ambassador to Greece (2018–21)",
        "note": "Born October 1959, Beijing. Parents both diplomats. BA, Beijing Foreign Studies Univ 1982. Simultaneous interpreter, UN New York & Geneva (1982–98). Spokesperson 1999–2005 (third female spokesperson since the system was established in 1983). Ambassador to Belgium (2005). Ambassador to Indonesia (2008). Ambassador to Greece (2018–21).",
    },
    "Liu Jianchao": {
        "era": "Hu Jintao",
        "post": "Head, CPC International Liaison Dept (2022–25)",
        "note": "Born February 1964, Dehui, Jilin. BA English, Beijing Foreign Languages Institute 1986; Oxford Univ 1986–87. First Secretary, Embassy in UK (1995–98). Director-General, MFA Information Dept / Spokesperson (2006–09). Ambassador to Philippines (2009). Ambassador to Indonesia (2012). Head, CPC International Liaison Dept (2022–25).",
    },
    "Qin Gang": {
        "era": "Xi Jinping",
        "post": "Foreign Minister Dec 2022 – Jul 2023",
        "note": "Born March 1966, Tianjin. LLB International Politics, Univ of International Relations 1988. Joined MFA 1992. Director-General, MFA Information Dept / Spokesperson (2005–10). Envoy to UK (2010–11). Director-General, Protocol Dept (2014–17). Assistant Foreign Minister (2017). Vice Foreign Minister (2018). Ambassador to US (2021–22). Foreign Minister (Dec 2022 – Jul 2023).",
    },
    "Jiang Yu": {
        "era": "Hu Jintao",
        "post": "Ambassador to Romania (2019–22)",
        "note": "Born 1964, Beijing. BA English, China Foreign Affairs Univ 1987. Spokesperson 2006–2012. Deputy Commissioner, MFA Office in Hong Kong (2012). Ambassador to Albania (2015). Ambassador to Romania (2019–22).",
    },
    "Ma Zhaoxu": {
        "era": "Hu Jintao / Xi Jinping",
        "post": "Executive Vice Foreign Minister (2023–)",
        "note": "Born September 1963, Harbin, Heilongjiang. BA International Relations, Peking Univ. Director-General, MFA Information Dept / Spokesperson (2009–12). Permanent Representative to UN, New York (2018–19). Ambassador to Australia. Ambassador to US (2022). Executive Vice Foreign Minister (Jan 2023–).",
    },
    "Liu Weimin": {
        "era": "Hu Jintao / Xi Jinping",
        "post": "Ambassador to Tonga (2024–)",
        "note": "Spokesperson 2011–2013. Minister-Counselor, Embassy in US. Ambassador to Tonga (2024–).",
    },
    "Hong Lei": {
        "era": "Hu Jintao / Xi Jinping",
        "post": "Director-General, Protocol Dept; Assistant Foreign Minister (2024–)",
        "note": "Born August 1969, Fuyang, Zhejiang. BA, Beijing Language and Culture Univ 1991. Embassy in Netherlands; Consulate-General in San Francisco. Spokesperson 2010–2016. Consul-General in Chicago (2016–18). Director-General, Protocol Dept (2019–). Assistant Foreign Minister (2024–).",
    },
    "Geng Shuang": {
        "era": "Xi Jinping",
        "post": "Deputy Permanent Representative to UN, New York (2020–)",
        "note": "Born April 1973, Beijing. BA English, China Foreign Affairs Univ 1995; MA International Relations, Tufts Univ 2006. Counselor, Embassy in US (2011–15). Spokesperson 2016–2020. Deputy Permanent Representative to UN, New York (2020–).",
    },
    "Lu Kang": {
        "era": "Xi Jinping",
        "post": "Deputy Minister, CPC International Liaison Dept (2024–)",
        "note": "Born May 1968, Nanjing, Jiangsu. Joined MFA 1992. Permanent Mission to UN, New York (1996–99). Counselor, Embassy in Ireland. Minister, Embassy in US (2012–15). Director-General, MFA Information Dept / Spokesperson (2015–19). Ambassador to Indonesia (2022–24). Deputy Minister, CPC International Liaison Dept (2024–).",
    },
    "Hua Chunying": {
        "era": "Xi Jinping",
        "post": "Vice Foreign Minister (May 2024–)",
        "note": "Born April 1970, Huai'an, Jiangsu. BA English, Nanjing Univ 1992. Joined MFA 1992. Dept of Western Europe; Embassy in Singapore (1992–2003). Mission to EU (2003–10). Spokesperson (2012–22). Assistant Foreign Minister (2019–24). Vice Foreign Minister (May 2024–).",
    },
    "Zhao Lijian": {
        "era": "Xi Jinping",
        "post": "Deputy Director, Dept of Boundary and Ocean Affairs (2023–)",
        "note": "Born November 1972, Hebei. Joined MFA 1996. Postings in US and Pakistan. Deputy Chief of Mission, Embassy in Pakistan (2015–19). Deputy Director-General, MFA Information Dept / Spokesperson (2019–23). Deputy Director, Dept of Boundary and Ocean Affairs (Jan 2023–).",
    },
    "Wang Wenbin": {
        "era": "Xi Jinping",
        "post": "Ambassador to Cambodia (2024–)",
        "note": "Born April 1971, Anhui. BA French, China Foreign Affairs Univ 1993. Political Counselor, Embassy in Mauritius. Counselor, Dept of Policy Planning. Ambassador to Tunisia (2018). Spokesperson (July 2020 – June 2024). Ambassador to Cambodia (2024–).",
    },
    "Mao Ning": {
        "era": "Xi Jinping",
        "post": "Spokesperson; Director-General, MFA Dept of Information (2025–)",
        "note": "Born December 1972, Xiangtan, Hunan. BA English, Hunan Normal Univ 1993; LLB Diplomacy, China Foreign Affairs Univ 1995; MA International Policy, George Washington Univ 2006. Asian affairs specialist; Deputy Director-General, Dept of Asian Affairs. Spokesperson (Sept 2022–). Director-General, MFA Dept of Information (Jan 2025–).",
    },
    "Lin Jian": {
        "era": "Xi Jinping",
        "post": "Active spokesperson (2024–)",
        "note": "Born May 1977, Wuhan, Hubei. BA English, Beijing Foreign Studies Univ 1999. Embassy in Denmark; Embassy in Poland; Dept of European Affairs. Foreign Affairs Office, Xinjiang Production and Construction Corps (2020–24). Spokesperson (March 2024–).",
    },
    "Guo Jiakun": {
        "era": "Xi Jinping",
        "post": "Active spokesperson (2025–)",
        "note": "Born August 1980. Mongol ethnicity. BA English, Nankai Univ, Tianjin 2002. Third Secretary, Permanent Mission to UN. Dept of African Affairs; Policy Planning Dept. Spokesperson (Jan 2025–).",
    },
}

# Load CDBD biographical database
# Source: Villegas-Cruz, A. M. (2026). Who speaks for Beijing? Insights from a new
#   dataset on China's diplomatic elite. Journal of Current Chinese Affairs.
#   https://doi.org/10.1177/18681026261439122
#   Chinese Diplomatic Elite Biographical Dataset (CDBD), Harvard Dataverse.
CDBD_PATH = "/Users/jamiegruffydd-jones/Downloads/dataverse_files (2)/CDBD 02.08.2026.xlsx"
try:
    cdbd_df = pd.read_excel(CDBD_PATH)
    cdbd_lookup = {row["name_en"]: row for _, row in cdbd_df.iterrows()}
    print(f"  Loaded CDBD ({len(cdbd_df)} rows)")
except Exception as e:
    cdbd_lookup = {}
    print(f"  WARNING: CDBD not loaded — {e}")

EDU_MAP = {1: "Bachelor's", 2: "Master's", 3: "Doctoral"}

CONF_TYPES = ["Threat", "Active threat", "Demand", "Urge", "Request", "Representations", "China attacked"]

def _cdbd_str(val):
    s = str(val).strip()
    return "" if s in ("nan", "", "None") else s

def build_cdbd_fields(sp):
    row = cdbd_lookup.get(sp)
    if row is None:
        return {}
    def v(col):
        return _cdbd_str(row.get(col, ""))
    edu_code = row.get("edu")
    edu_str = EDU_MAP.get(int(edu_code), "") if pd.notna(edu_code) else ""
    return {
        "cdbd_name_zh":   v("name_zh"),
        "cdbd_born":      int(row["born"]) if pd.notna(row.get("born")) else None,
        "cdbd_birth_city":   v("birth_city"),
        "cdbd_birth_region": v("birth_region"),
        "cdbd_edu":        edu_str,
        "cdbd_major":      v("major"),
        "cdbd_univ":       v("university"),
        "cdbd_g_major":    v("g_major"),
        "cdbd_g_univ":     v("g_university"),
        "cdbd_edu_abroad": v("edu_abroad"),
        "cdbd_lang":       v("foreign_lang"),
        "cdbd_twitter":    v("twitter_handdle"),
        "cdbd_local_bg":   v("local_background"),
        "cdbd_ccp_bg":     v("ccp_background"),
        "cdbd_ministry_bg": v("ministry_background"),
        "cdbd_congress19": bool(row.get("ccpcongress_19") == 1),
        "cdbd_congress20": bool(row.get("ccpcongress_20") == 1),
        "cdbd_notes":      v("notes"),
    }

cards = []
for sp in TOP_SP:
    rows = df[df["spokesperson"] == sp]
    n_total = len(rows)
    n_io = io_mask[rows.index].sum()
    conf_mask = rows[conf_cols].notna().any(axis=1)
    n_conf = conf_mask.sum()
    yrs = rows["year_clean"].dropna()
    year_min = int(yrs.min()) if len(yrs) else None
    year_max = int(yrs.max()) if len(yrs) else None
    bio = SP_BIO.get(sp, {})

    # Confrontation type breakdown (exclusive priority, same as quarterly_io)
    breakdown = {}
    for ct in CONF_TYPES:
        if ct in rows.columns:
            breakdown[ct] = int(rows[ct].notna().sum())
        else:
            breakdown[ct] = 0

    card = {
        "name": sp,
        "color": SP_COLOR.get(sp, "#999"),
        "n_total": int(n_total),
        "n_io": int(n_io),
        "n_conf": int(n_conf),
        "conf_rate": round(float(n_conf / n_total * 100), 1) if n_total else 0,
        "year_min": year_min,
        "year_max": year_max,
        "era": bio.get("era", ""),
        "post": bio.get("post", ""),
        "note": bio.get("note", ""),
        "conf_breakdown": breakdown,
    }
    card.update(build_cdbd_fields(sp))
    cards.append(card)

with open(os.path.join(OUT_DIR, "spokesperson_cards.json"), "w") as f:
    json.dump(cards, f)
print(f"  wrote spokesperson_cards.json ({len(cards)} cards)")

# ── C: Press breakdown ────────────────────────────────────────────────────────
print("Building press_breakdown …")

# Categorise who_asked
CHINESE_STATE = {
    "CCTV", "Xinhua News Agency", "Global Times", "China Daily",
    "China News Service", "Shenzhen TV", "CRI", "Dragon TV",
    "Hubei Media Group", "Guancha.cn", "CRI Online",
    "People's Daily", "Guangming Daily",
}
CHINESE_OTHER = {
    "The Paper", "Beijing Youth Daily", "Beijing Daily", "Beijing Evening News",
    "China Review News", "Phoenix TV", "Caijing", "Caixin",
    "21st Century Business Herald",
}
WIRE_WESTERN = {"AFP", "Reuters", "Bloomberg", "AP", "UPI"}
JAPANESE = {"NHK", "Kyodo News"}
INDIAN = {"PTI", "Prasar Bharati", "Hindustan Times", "Times of India", "NDTV"}
RUSSIAN = {"RIA Novosti", "TASS", "RT"}

def classify_outlet(name):
    if not name or str(name).strip() in ("-", "nan", ""):
        return "Unknown / no attribution"
    n = str(name).strip()
    if n in CHINESE_STATE:
        return "Chinese state media"
    if n in CHINESE_OTHER:
        return "Chinese non-state media"
    if n in WIRE_WESTERN:
        return "Western wire services"
    if n in JAPANESE:
        return "Japanese press"
    if n in INDIAN:
        return "Indian press"
    if n in RUSSIAN:
        return "Russian press"
    # Heuristics
    if any(x in n for x in ("TV", "Radio", "News Agency")) and \
       not any(x in n for x in ("Reuters","AFP","Bloomberg","NHK","Yonhap","RIA","TASS")):
        return "Other broadcast"
    return "Other / international"

df["press_cat"] = df["who_asked"].apply(classify_outlet)
conf_any = df[conf_cols].notna().any(axis=1)

press_summary = []
for cat, grp in df.groupby("press_cat"):
    n = len(grp)
    n_conf = conf_any[grp.index].sum()
    n_io = io_mask[grp.index].sum()
    press_summary.append({
        "category": cat,
        "n": int(n),
        "n_conf": int(n_conf),
        "conf_rate": round(float(n_conf / n * 100), 1),
        "n_io": int(n_io),
    })
press_summary.sort(key=lambda x: -x["n"])

# Top individual outlets (excluding unknown)
top_outlets = (
    df[df["who_asked"].astype(str).str.strip() != "-"]["who_asked"]
    .value_counts()
    .head(20)
    .reset_index()
)
top_outlets.columns = ["outlet", "n"]
outlet_conf = []
for _, row in top_outlets.iterrows():
    grp = df[df["who_asked"] == row["outlet"]]
    n_conf = conf_any[grp.index].sum()
    outlet_conf.append({
        "outlet": row["outlet"],
        "n": int(row["n"]),
        "n_conf": int(n_conf),
        "conf_rate": round(float(n_conf / row["n"] * 100), 1),
        "category": classify_outlet(row["outlet"]),
    })

press_breakdown = {
    "categories": press_summary,
    "top_outlets": outlet_conf,
}

with open(os.path.join(OUT_DIR, "press_breakdown.json"), "w") as f:
    json.dump(press_breakdown, f)
print(f"  wrote press_breakdown.json ({len(press_summary)} categories, {len(outlet_conf)} outlets)")

# ── D: Topics ─────────────────────────────────────────────────────────────────
print("Building topics …")

# Top organisations in questions
def expand_field(series):
    return (
        series.dropna()
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
        .replace("-", pd.NA)
        .dropna()
    )

q_org_vals = expand_field(df["q_org"])
top_q_org = q_org_vals.value_counts().head(20)
_conf_by_org_tmp = {}  # filled below after conf_by_org is built
q_org_list = [{"label": k, "n": int(v)} for k, v in top_q_org.items()
              if k not in ("nan", "-", "")]

# Top locations in questions (non-China, non-trivial)
q_loc_vals = expand_field(df["q_loc"])
SKIP_LOC = {"China", "nan", "-", ""}
top_q_loc = q_loc_vals[~q_loc_vals.isin(SKIP_LOC)].value_counts().head(20)
q_loc_list = [{"label": k, "n": int(v)} for k, v in top_q_loc.items()]

# Top locations in answers
a_loc_vals = expand_field(df["a_loc"])
top_a_loc = a_loc_vals[~a_loc_vals.isin(SKIP_LOC)].value_counts().head(20)
a_loc_list = [{"label": k, "n": int(v)} for k, v in top_a_loc.items()]

# Confrontation rate by top organisation (q_org)
MERGE_ORG = {
    "UN Security Council": "UN Security Council",
    "Security Council": "UN Security Council",
    "United Nations": "UN",
}
conf_by_org = []
for org in [x["label"] for x in q_org_list[:15]]:
    org_rows = df[df["q_org"].astype(str).str.contains(re.escape(org), na=False)]
    if len(org_rows) < 5:
        continue
    n_conf = conf_any[org_rows.index].sum()
    conf_by_org.append({
        "org": org,
        "n": int(len(org_rows)),
        "n_conf": int(n_conf),
        "conf_rate": round(float(n_conf / len(org_rows) * 100), 1),
    })

# Back-fill conf_rate into q_org_list
_conf_rate_map = {d["org"]: d["conf_rate"] for d in conf_by_org}
for item in q_org_list:
    item["conf_rate"] = _conf_rate_map.get(item["label"], 0.0)

topics = {
    "top_q_org": q_org_list,
    "top_q_loc": q_loc_list,
    "top_a_loc": a_loc_list,
    "conf_by_org": conf_by_org,
}

with open(os.path.join(OUT_DIR, "topics.json"), "w") as f:
    json.dump(topics, f)
print(f"  wrote topics.json ({len(q_org_list)} orgs, {len(q_loc_list)} q_locs)")

# ── E: Article explorer (IO + confrontation rows) ─────────────────────────────
print("Building articles.json …")

union_mask = io_mask | conf_any
explorer_df = df[union_mask].copy()
print(f"  {len(explorer_df):,} rows in explorer set")

# Build tags list for each row
def build_tags(row):
    tags = []
    if pd.notna(row.get("Active threat")):
        tags.append("Active threat")
    if pd.notna(row.get("China attacked")):
        tags.append("Attacked abroad")
    if pd.notna(row.get("Threat")):
        tags.append("Threat")
    if pd.notna(row.get("Demand")):
        tags.append("Demand")
    if pd.notna(row.get("Urge")):
        tags.append("Urge")
    if pd.notna(row.get("Request")):
        tags.append("Request")
    if pd.notna(row.get("Representations")):
        tags.append("Representations")
    return tags

def clean_str(s, maxlen=None):
    s = str(s).strip()
    if s in ("-", "nan", ""):
        return ""
    if maxlen:
        s = s[:maxlen]
    return s

# International-order content hidden 2026-06-30 (Jamie): skip IO-mention-only rows and drop the
# "IO mention" tag, so the explorer shows only confrontational records. To re-enable, set False
# (and restore the #trends section, its nav link, the hero card and the IO filter option in index.html).
HIDE_IO = True
articles = []
for _, row in explorer_df.iterrows():
    tags = build_tags(row)
    if HIDE_IO and not tags:
        continue                       # IO-mention-only row → hidden
    is_io = bool(
        "international order" in str(row.get("answer", "")).lower()
    )
    if is_io and not HIDE_IO and "IO mention" not in tags:
        tags = ["IO mention"] + tags

    # date string
    yr = row.get("year_clean")
    date_str = clean_str(row.get("date", ""))
    if not date_str and yr:
        mo = row.get("month", "")
        dy = row.get("day", "")
        date_str = f"{int(yr)}-{mo}-{dy}" if mo and dy else str(int(yr))

    articles.append({
        "date": date_str,
        "year": int(yr) if pd.notna(yr) else None,
        "spokesperson": clean_str(row.get("spokesperson", "")),
        "who_asked": clean_str(row.get("who_asked", "")),
        "q_loc": clean_str(row.get("q_loc", ""), 100),
        "question": clean_str(row.get("question", ""), 400),
        "answer": clean_str(row.get("answer", ""), 600),
        "tags": tags,
        "sentiment": round(float(row["vader_answer"]), 3) if pd.notna(row.get("vader_answer")) else None,
        "link": clean_str(row.get("link", ""), 200),
    })

# Sort by date descending (most recent first)
articles.sort(key=lambda x: x["date"] or "", reverse=True)

with open(os.path.join(OUT_DIR, "articles.json"), "w") as f:
    json.dump(articles, f)
size_mb = os.path.getsize(os.path.join(OUT_DIR, "articles.json")) / 1e6
print(f"  wrote articles.json ({len(articles):,} records, {size_mb:.1f} MB)")

# ── E2: Quarterly IO breakdown ────────────────────────────────────────────────
print("Building quarterly_io …")

io_df = df[io_mask].copy()
io_df["date_p"] = pd.to_datetime(io_df["date"], errors="coerce")
io_df["quarter"] = io_df["date_p"].dt.to_period("Q").astype(str)
io_df["vader"] = pd.to_numeric(io_df.get("vader_answer", pd.Series(dtype=float)), errors="coerce")

# Exclusive confrontation assignment: highest-severity category wins
CONF_PRIORITY = ["Active threat", "Threat", "Demand", "Urge", "Representations", "China attacked"]

def assign_framing(row):
    for col in CONF_PRIORITY:
        if col in row and pd.notna(row[col]):
            return col
    return "IO only"

io_df["framing"] = io_df.apply(assign_framing, axis=1)

# Build complete quarter range from first to last date in full corpus
df["_date_p"] = pd.to_datetime(df["date"], errors="coerce")
q_min = df["_date_p"].min().to_period("Q")
q_max = df["_date_p"].max().to_period("Q")
all_quarters = [str(q_min + i) for i in range((q_max - q_min).n + 1)]
FRAMINGS = ["IO only", "Urge", "Representations", "Demand", "Threat", "Active threat"]

quarterly_io = []
for q in all_quarters:
    qrows = io_df[io_df["quarter"] == q]
    # Total IO exchanges in this quarter
    n_io = len(qrows)
    # Total MFA exchanges in this quarter (for rate)
    qdate = pd.Period(q, "Q")
    all_q = df[
        (pd.to_datetime(df["date"], errors="coerce").dt.to_period("Q").astype(str) == q)
    ]
    n_total = len(all_q)
    # Sentiment (mean VADER of IO rows)
    vader_mean = float(qrows["vader"].mean()) if qrows["vader"].notna().any() else None
    # Framing breakdown
    framing_counts = qrows["framing"].value_counts().to_dict()
    entry = {
        "quarter": q,
        "n_io": n_io,
        "n_total": n_total,
        "io_rate": round(n_io / n_total * 100, 2) if n_total else 0,
        "sentiment": round(vader_mean, 4) if vader_mean is not None else None,
    }
    for f in FRAMINGS:
        entry[f.lower().replace(" ", "_")] = framing_counts.get(f, 0)
    quarterly_io.append(entry)

with open(os.path.join(OUT_DIR, "quarterly_io.json"), "w") as f:
    json.dump(quarterly_io, f)
print(f"  wrote quarterly_io.json ({len(quarterly_io)} quarters)")

# ── F: Threats by year ────────────────────────────────────────────────────────
print("Building threats_by_year …")

ALL_CONF_COLS = ["Threat", "Active threat", "Demand", "Urge", "Request",
                 "Representations", "China attacked", "China protested"]

tby_years = sorted(df["year_clean"].dropna().unique())
threats_by_year = []
for yr in tby_years:
    rows = df[df["year_clean"] == yr]
    conf_mask = rows[ALL_CONF_COLS].notna().any(axis=1)
    # corpus_total = ALL exchanges that year (the whole-corpus denominator)
    entry = {"year": int(yr), "total": int(conf_mask.sum()), "corpus_total": int(len(rows))}
    for col in ALL_CONF_COLS:
        entry[col] = int(rows[col].notna().sum()) if col in rows.columns else 0
    threats_by_year.append(entry)

with open(os.path.join(OUT_DIR, "threats_by_year.json"), "w") as f:
    json.dump(threats_by_year, f)
print(f"  wrote threats_by_year.json ({len(threats_by_year)} years)")

# ── F2: Threats by calendar month (seasonality, aggregated across all years) ──
print("Building threats_by_month …")

df["_month"] = pd.to_datetime(df["date"], errors="coerce").dt.month
threats_by_month = []
for mo in range(1, 13):
    rows = df[df["_month"] == mo]
    conf_mask = rows[ALL_CONF_COLS].notna().any(axis=1)
    entry = {"month": mo, "total": int(conf_mask.sum()), "corpus_total": int(len(rows))}
    for col in ALL_CONF_COLS:
        entry[col] = int(rows[col].notna().sum()) if col in rows.columns else 0
    threats_by_month.append(entry)

with open(os.path.join(OUT_DIR, "threats_by_month.json"), "w") as f:
    json.dump(threats_by_month, f)
print(f"  wrote threats_by_month.json (12 months)")

# ── F3: Threats by actual year-month (continuous, 0-filled — for the rolling 3-monthly view) ──
print("Building threats_by_yearmonth …")
df["_ymp"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M")
_valid = df["_ymp"].dropna()
threats_by_yearmonth = []
if len(_valid):
    for p in pd.period_range(_valid.min(), _valid.max(), freq="M"):
        rows = df[df["_ymp"] == p]
        cm = rows[ALL_CONF_COLS].notna().any(axis=1) if len(rows) else None
        entry = {"ym": str(p), "year": int(p.year),
                 "total": int(cm.sum()) if cm is not None else 0,
                 "corpus_total": int(len(rows))}
        for col in ALL_CONF_COLS:
            entry[col] = int(rows[col].notna().sum()) if (col in rows.columns and len(rows)) else 0
        threats_by_yearmonth.append(entry)
with open(os.path.join(OUT_DIR, "threats_by_yearmonth.json"), "w") as f:
    json.dump(threats_by_yearmonth, f)
print(f"  wrote threats_by_yearmonth.json ({len(threats_by_yearmonth)} months)")

# ── F4: MFA DOMESTIC-QUESTION subset (question asked by a domestic Chinese outlet) ──
# Same confrontation series as the full MFA record, but restricted to exchanges whose QUESTION
# came from a Chinese outlet (press_cat = Chinese state or non-state media — CCTV, Xinhua, Global
# Times, People's Daily, China Daily, The Paper, etc.). China's own confrontational ANSWERS are
# still what's coded; this isolates the answers given to domestic (often teed-up) questioners.
# NOTE: who_asked is only populated from 2020, so these series necessarily begin in 2020.
# Powers the "MFA — domestic questions" source toggle on the Confrontation-over-time chart.
print("Building MFA domestic-question feeds …")
DOMESTIC_PRESS = {"Chinese state media", "Chinese non-state media"}
dom = df[df["press_cat"].isin(DOMESTIC_PRESS)].copy()
print(f"  {len(dom):,} domestic-question exchanges ({len(dom)/len(df)*100:.1f}% of corpus; from {int(dom['year_clean'].dropna().min())})")

dom_year = []
for yr in sorted(dom["year_clean"].dropna().unique()):
    rows = dom[dom["year_clean"] == yr]
    cm = rows[ALL_CONF_COLS].notna().any(axis=1)
    entry = {"year": int(yr), "total": int(cm.sum()), "corpus_total": int(len(rows))}
    for col in ALL_CONF_COLS:
        entry[col] = int(rows[col].notna().sum()) if col in rows.columns else 0
    dom_year.append(entry)
with open(os.path.join(OUT_DIR, "threats_by_year_domestic.json"), "w") as f:
    json.dump(dom_year, f)

dom["_month"] = pd.to_datetime(dom["date"], errors="coerce").dt.month
dom_month = []
for mo in range(1, 13):
    rows = dom[dom["_month"] == mo]
    cm = rows[ALL_CONF_COLS].notna().any(axis=1)
    entry = {"month": mo, "total": int(cm.sum()), "corpus_total": int(len(rows))}
    for col in ALL_CONF_COLS:
        entry[col] = int(rows[col].notna().sum()) if col in rows.columns else 0
    dom_month.append(entry)
with open(os.path.join(OUT_DIR, "threats_by_month_domestic.json"), "w") as f:
    json.dump(dom_month, f)

dom["_ymp"] = pd.to_datetime(dom["date"], errors="coerce").dt.to_period("M")
_dv = dom["_ymp"].dropna()
dom_ym = []
if len(_dv):
    for p in pd.period_range(_dv.min(), _dv.max(), freq="M"):
        rows = dom[dom["_ymp"] == p]
        cm = rows[ALL_CONF_COLS].notna().any(axis=1) if len(rows) else None
        entry = {"ym": str(p), "year": int(p.year),
                 "total": int(cm.sum()) if cm is not None else 0,
                 "corpus_total": int(len(rows))}
        for col in ALL_CONF_COLS:
            entry[col] = int(rows[col].notna().sum()) if (col in rows.columns and len(rows)) else 0
        dom_ym.append(entry)
with open(os.path.join(OUT_DIR, "threats_by_yearmonth_domestic.json"), "w") as f:
    json.dump(dom_ym, f)
print(f"  wrote threats_by_year_domestic ({len(dom_year)} yrs) / _month / _yearmonth ({len(dom_ym)} mo)")

# ── G: Threats by country ──────────────────────────────────────────────────────
print("Building threats_by_country …")

CONF_ISO_MAP = {
    # Abbreviations / aliases used in NLP-extracted q_loc
    'US':'USA','U.S.':'USA','USA':'USA','United States':'USA','America':'USA',
    'UK':'GBR','U.K.':'GBR','Britain':'GBR','United Kingdom':'GBR',
    'DPRK':'PRK','North Korea':'PRK',
    'ROK':'KOR','South Korea':'KOR',
    'Philippine':'PHL','Philippines':'PHL',
    'Japan':'JPN','Taiwan':'TWN','Russia':'RUS','Hong Kong':'HKG',
    'India':'IND','Pakistan':'PAK','Xinjiang':'CHN','Tibet':'CHN','Macao':'MAC',
    'Afghanistan':'AFG','Iran':'IRN','Ukraine':'UKR','Australia':'AUS',
    'Germany':'DEU','France':'FRA','Indonesia':'IDN','Palestine':'PSE',
    'Gaza':'PSE','Syria':'SYR','Iraq':'IRQ','Canada':'CAN',
    'Israel':'ISR','Myanmar':'MMR','Burma':'MMR',
    'Sri Lanka':'LKA','Vietnam':'VNM','Libya':'LBY','Sudan':'SDN',
    'Egypt':'EGY','Saudi Arabia':'SAU','Turkey':'TUR','Türkiye':'TUR',
    'Brazil':'BRA','South Africa':'ZAF','Cuba':'CUB','Venezuela':'VEN',
    'Nepal':'NPL','Cambodia':'KHM','Singapore':'SGP','Thailand':'THA',
    'Malaysia':'MYS','Kazakhstan':'KAZ','Italy':'ITA','Spain':'ESP',
    'Nigeria':'NGA','Kenya':'KEN','Ethiopia':'ETH','Zimbabwe':'ZWE',
    'Angola':'AGO','DR Congo':'COD','Serbia':'SRB','Panama':'PAN',
    'Bangladesh':'BGD','Mongolia':'MNG','Laos':'LAO','Uzbekistan':'UZB',
    'Hungary':'HUN','Solomon Islands':'SLB','Fiji':'FJI','Tonga':'TON',
    'Vanuatu':'VUT','Papua New Guinea':'PNG','Maldives':'MDV','Bhutan':'BTN',
    'UAE':'ARE','Qatar':'QAT','Kuwait':'KWT','Jordan':'JOR','Lebanon':'LBN',
    'Yemen':'YEM','Morocco':'MAR','Tunisia':'TUN','Algeria':'DZA',
    'Belarus':'BLR','Azerbaijan':'AZE','Armenia':'ARM','Lithuania':'LTU',
    'Poland':'POL','Czech Republic':'CZE','Sweden':'SWE','Finland':'FIN',
    'Norway':'NOR','Denmark':'DNK','Netherlands':'NLD','Belgium':'BEL',
    'Portugal':'PRT','Greece':'GRC','Austria':'AUT','Switzerland':'CHE',
    'Ireland':'IRL','Croatia':'HRV','Seychelles':'SYC','Albania':'ALB',
    'Niger':'NER','Kosovo':'XKX',
}

# q_loc (NLP entity extraction) is only populated for the original CMFA_v4 corpus
# (2002–2023) — it is 0% populated on scraped_2024_2026 rows, so without a fallback
# this map (and its year-sliced sibling) go silently empty from 2024 on. Fix: fall
# back to the same keyword text-match used for country_counts.json's 2024+ rows
# (text_country_mentions on question+answer), mapped through CONF_ISO_MAP. (Bug
# found + fixed 2026-08-06 — the map had nothing after 2023.)
def row_iso_locs(row):
    q_raw = str(row.get("q_loc", ""))
    if q_raw not in ("nan", ""):
        return [CONF_ISO_MAP[l] for l in
                (x.strip() for x in q_raw.split(";"))
                if l not in ("-", "nan", "") and l in CONF_ISO_MAP]
    # Fallback for rows with no q_loc (scraped 2024+): keyword match on Q+A text
    text = str(row.get("question", "")) + " " + str(row.get("answer", ""))
    return [CONF_ISO_MAP[name] for name in text_country_mentions(text) if name in CONF_ISO_MAP]

threats_by_country = {}
conf_rows = df[df[ALL_CONF_COLS].notna().any(axis=1)]
for _, row in conf_rows.iterrows():
    for iso in set(row_iso_locs(row)):
        if iso not in threats_by_country:
            threats_by_country[iso] = {"total": 0}
            for c in ALL_CONF_COLS:
                threats_by_country[iso][c] = 0
        threats_by_country[iso]["total"] += 1
        for c in ALL_CONF_COLS:
            if c in conf_rows.columns and pd.notna(row.get(c)):
                threats_by_country[iso][c] += 1

with open(os.path.join(OUT_DIR, "threats_by_country.json"), "w") as f:
    json.dump(threats_by_country, f)
print(f"  wrote threats_by_country.json ({len(threats_by_country)} countries)")

# ── G2: Threats by country by year (for year slider on confrontation map) ────────
print("Building threats_by_country_year …")
threats_by_country_year = {}
for _, row in conf_rows.iterrows():
    yr = int(row.get("year_clean", 0)) if pd.notna(row.get("year_clean")) else 0
    if yr == 0:
        continue
    for iso in set(row_iso_locs(row)):
        if iso not in threats_by_country_year:
            threats_by_country_year[iso] = {}
        if yr not in threats_by_country_year[iso]:
            threats_by_country_year[iso][yr] = {"total": 0}
            for c in ALL_CONF_COLS:
                threats_by_country_year[iso][yr][c] = 0
        threats_by_country_year[iso][yr]["total"] += 1
        for c in ALL_CONF_COLS:
            if c in conf_rows.columns and pd.notna(row.get(c)):
                threats_by_country_year[iso][yr][c] += 1

with open(os.path.join(OUT_DIR, "threats_by_country_year.json"), "w") as f:
    json.dump(threats_by_country_year, f)
print(f"  wrote threats_by_country_year.json ({len(threats_by_country_year)} countries)")

# ── H: Commentary comparison (MFA podium vs People's Daily commentariat) ─────────
# The MFA scheme codes spoken Q&A; the same confrontation categories are applied to
# official COMMENTARY (People's Daily 钟声/国纪平 in Chinese, "Zhong Sheng" in English) by
# Scraping/commentary_autocode.py. The two corpora differ enormously in size (≈36k Q&A vs a
# few hundred op-eds), so the comparison is on confrontational SHARE, never raw counts.
print("Building commentary_compare …")

# Request is EXCLUDED from the confrontational construct: the κ validation showed it is the
# mild "general-policy should" tier that sits below the confrontation line — the human coder
# barely used it (1×) while the model over-fired it (10×). Dropping it raised agreement from
# κ 0.51 → 0.61 and matched the human confrontational share. (Request is still coded and kept in
# the masters for the record; it just isn't counted here.) See analysis/validation/2026-06-29…/.
COMMENTARY_CATS = ["Threat", "Active threat", "Demand", "Urge", "Representations"]
# Ordinal intensity weights (audit: the binary `confrontational` flag is severity-blind — it lumps
# a mild Urge with an Active threat). A piece's severity = the weight of its STRONGEST act; a
# source's mean severity (0–5, incl. non-confrontational pieces at 0) is a severity-aware analogue
# of the confrontational share. Weights are a transparent ordinal scale, not tuned to results.
SEVERITY_W = {"Active threat": 5, "Threat": 4, "Demand": 3, "Representations": 2, "Urge": 1, "Request": 0.5}
COMMENTARY_DIR = os.path.join(SCRIPT_DIR, "../../data/raw-extracts/commentary")
COMMENTARY_SRC = [
    # key, label, sublabel, master path (relative to COMMENTARY_DIR)
    ("pd_zh", "People's Daily commentary (Chinese)", "钟声 · 和音 · 国际论坛 · 望海楼 · 国纪平",
     "pd_zh/pd_commentary_master.csv"),
    ("pd_en", "People's Daily “Zhong Sheng”", "English commentary",
     "pd_en/pd_en_commentary_master.csv"),
    ("cd", "China Daily editorials", "English commentary",
     "cd/cd_commentary_master.csv"),
]
# pd_en (English “Zhong Sheng”) has now been re-coded on the validated Sonnet+v4 coder (2026-06-30,
# Batch + codebook caching): 24.9% confrontational (Request excluded), consistent with the other
# sources and close to the Chinese 钟声 (32.4%). It is INCLUDED again. (Set True to drop it.)
PD_EN_RECODE_PENDING = False


def _year_series(frame, year_col):
    """Per-year {year, n, conf, <each cat>} where conf = any kept category fires."""
    out = []
    yrs = sorted(int(y) for y in frame[year_col].dropna().unique())
    for yr in yrs:
        rows = frame[frame[year_col] == yr]
        present = [c for c in COMMENTARY_CATS if c in rows.columns]
        conf_mask = rows[present].notna().any(axis=1) if present else pd.Series(False, index=rows.index)
        entry = {"year": int(yr), "n": int(len(rows)), "conf": int(conf_mask.sum())}
        for c in COMMENTARY_CATS:
            entry[c] = int(rows[c].notna().sum()) if c in rows.columns else 0
        out.append(entry)
    return out


def _src_summary(frame, year_col):
    present = [c for c in COMMENTARY_CATS if c in frame.columns]
    conf_mask = frame[present].notna().any(axis=1) if present else pd.Series(False, index=frame.index)
    n = int(len(frame))
    cats = {c: (int(frame[c].notna().sum()) if c in frame.columns else 0) for c in COMMENTARY_CATS}
    # per-piece severity = weight of the strongest act present; source mean over ALL pieces
    if present and n:
        sev = frame[present].apply(
            lambda r: max((SEVERITY_W[c] for c in present if pd.notna(r[c])), default=0), axis=1)
        mean_sev = round(float(sev.mean()), 2)
    else:
        mean_sev = 0.0
    conf_cats = {c: v for c, v in cats.items() if v > 0}
    dominant_mode = max(conf_cats, key=conf_cats.get) if conf_cats else None
    return {
        "n": n,
        "conf_n": int(conf_mask.sum()),
        "conf_share": round(conf_mask.mean() * 100, 1) if n else 0,
        "cats": cats,
        "mean_severity": mean_sev,                 # 0–5 ordinal intensity (severity-aware)
        "dominant_mode": dominant_mode,            # most common confrontational act for this source
        "by_year": _year_series(frame, year_col),
    }

commentary_sources = []

# MFA as the reference source (spoken Q&A), on the same six categories.
# UNIT NOTE: this counts each Q&A pair. Commentary sources count whole articles, so the
# per-exchange MFA share is not unit-comparable with them — see mfa_day below.
mfa_frame = df.copy()
mfa_frame["_y"] = mfa_frame["year_clean"]
mfa_sum = _src_summary(mfa_frame, "_y")
mfa_sum.update({"key": "mfa", "label": "MFA — per exchange", "sublabel": "share of Q&A pairs"})
commentary_sources.append(mfa_sum)

# MFA at the PRESS-CONFERENCE (day) level — one row per briefing, a category present if ANY
# Q&A that day carried it. This is the unit-comparable reference for the article-level
# commentary sources (a whole article ≈ a whole briefing as a "communication event").
_md = df.copy()
_md["_dt"] = pd.to_datetime(_md["date"], errors="coerce")
_md = _md[_md["_dt"].notna()]
_dcats = [c for c in COMMENTARY_CATS if c in df.columns]
_day_rows = []
for _d, _g in _md.groupby(_md["_dt"].dt.date):
    _rec = {"_y": _d.year}
    for c in COMMENTARY_CATS:
        _rec[c] = 1.0 if (c in _dcats and _g[c].notna().any()) else float("nan")
    _day_rows.append(_rec)
mfa_day_frame = pd.DataFrame(_day_rows)
mfa_day_sum = _src_summary(mfa_day_frame, "_y")
mfa_day_sum.update({"key": "mfa_day", "label": "MFA — per briefing",
                    "sublabel": "≥1 confrontational exchange/day"})
commentary_sources.append(mfa_day_sum)

# People's Daily commentary corpora (if their coded masters exist yet)
for key, label, sublabel, rel in COMMENTARY_SRC:
    if key == "pd_en" and PD_EN_RECODE_PENDING:
        print("  (skip pd_en: re-code pending — still on old coder; see PD_EN_RECODE_PENDING)")
        continue
    path = os.path.join(COMMENTARY_DIR, rel)
    if not os.path.exists(path):
        print(f"  (skip {key}: no master at {rel})")
        continue
    cdf = pd.read_csv(path, low_memory=False)
    cdf["_y"] = pd.to_datetime(cdf["date"], errors="coerce").dt.year
    cdf = cdf[cdf["_y"].notna()]
    s = _src_summary(cdf, "_y")
    s.update({"key": key, "label": label, "sublabel": sublabel})
    commentary_sources.append(s)
    print(f"  {key}: {s['n']} pieces, {s['conf_share']}% confrontational")

    # 钟声 (Zhong Sheng) broken out as its OWN source — the flagship, most confrontational
    # column — so its distinct signal isn't diluted into the all-columns PD aggregate.
    if key == "pd_zh" and "column" in cdf.columns:
        zs = cdf[cdf["column"].astype(str).str.contains("钟声", na=False)]
        if len(zs):
            zsum = _src_summary(zs, "_y")
            zsum.update({"key": "pd_zh_zs", "label": "People's Daily 钟声 (Zhong Sheng)",
                         "sublabel": "Chinese · flagship column"})
            commentary_sources.append(zsum)
            print(f"  pd_zh_zs (钟声 only): {zsum['n']} pieces, {zsum['conf_share']}% confrontational")

commentary_compare = {
    "categories": COMMENTARY_CATS,
    "severity_weights": SEVERITY_W,
    "sources": commentary_sources,
    "note": ("The same confrontation categories are applied to spoken MFA Q&A and to official "
             "People's Daily foreign-affairs commentary (钟声, 和音, 国际论坛, 望海楼, 国纪平; the "
             "translated “Zhong Sheng” in English) and China Daily editorials. 钟声 is also shown "
             "on its own — the flagship, most confrontational column. Corpora differ in size, so "
             "all comparisons are shares, not counts. Commentary is topic-agnostic."),
}
with open(os.path.join(OUT_DIR, "commentary_compare.json"), "w") as f:
    json.dump(commentary_compare, f, ensure_ascii=False)
print(f"  wrote commentary_compare.json ({len(commentary_sources)} sources)")

# Per-piece EXPLORER feed — the confrontational commentary pieces, so readers can read the actual
# editorials. Chinese pieces (pd_zh) carry zh=True so the site renders a Google-Translate link;
# English pieces (pd_en / China Daily) link straight to the source. Only the confrontational pieces
# are shipped (the aggregate charts above already carry the denominator), keeping the feed small.
print("Building commentary_articles (explorer feed) …")
SRC_LABEL = {"pd_zh": "People's Daily", "pd_en": "People's Daily (English)", "cd": "China Daily"}
commentary_articles = []
for key, label, sublabel, rel in COMMENTARY_SRC:
    if key == "pd_en" and PD_EN_RECODE_PENDING:
        continue
    path = os.path.join(COMMENTARY_DIR, rel)
    if not os.path.exists(path):
        continue
    a = pd.read_csv(path, low_memory=False)
    a["_dt"] = pd.to_datetime(a["date"], errors="coerce")
    a = a[a["_dt"].notna()]
    present = [c for c in COMMENTARY_CATS if c in a.columns]
    conf = a[a[present].notna().any(axis=1)] if present else a.iloc[0:0]
    for _, r in conf.iterrows():
        commentary_articles.append({
            "date": r["_dt"].strftime("%Y-%m-%d"),
            "year": int(r["_dt"].year),
            "src": key,
            "src_label": SRC_LABEL.get(key, key),
            "column": str(r.get("column", "") or ""),
            "title": str(r.get("title", "") or "").strip(),
            "codes": [c for c in present if pd.notna(r[c])],
            "zh": key == "pd_zh",                       # Chinese → render a translate link
            "url": str(r.get("url", "") or ""),
        })
commentary_articles.sort(key=lambda x: x["date"], reverse=True)
with open(os.path.join(OUT_DIR, "commentary_articles.json"), "w") as f:
    json.dump(commentary_articles, f, ensure_ascii=False)
print(f"  wrote commentary_articles.json ({len(commentary_articles)} confrontational pieces)")

# ── State-media PARALLEL feeds (for the map + over-time toggles) ───────────────────────────────
# Combined PD (Chinese + English) + China Daily commentary on the SAME confrontation categories as
# the MFA series (the kept five — Request excluded, matching the rest of the commentary analysis),
# so the map and the over-time chart can toggle the spoken MFA record against the written one.
print("Building state-media parallel feeds (sm_threats_*) …")
sm_frames = []
for key, label, sublabel, rel in COMMENTARY_SRC:
    if key == "pd_en" and PD_EN_RECODE_PENDING:
        continue
    p = os.path.join(COMMENTARY_DIR, rel)
    if os.path.exists(p):
        f = pd.read_csv(p, low_memory=False)
        f["_dt"] = pd.to_datetime(f["date"], errors="coerce")
        sm_frames.append(f[f["_dt"].notna()])
sm = pd.concat(sm_frames, ignore_index=True) if sm_frames else pd.DataFrame()
sm_cats = [c for c in COMMENTARY_CATS if c in sm.columns]            # the kept five
sm["_conf"] = sm[sm_cats].notna().any(axis=1) if (sm_cats and len(sm)) else False

def _sm_bucket(col):
    out = {}
    for gk, g in sm.groupby(col):
        rec = {"total": int(g["_conf"].sum()), "corpus_total": int(len(g))}
        for c in sm_cats:
            rec[c] = int(g[c].notna().sum())
        out[gk] = rec
    return out

sm["_yr"] = sm["_dt"].dt.year
sm_year = [dict(year=int(y), **rec) for y, rec in sorted(_sm_bucket("_yr").items())]
with open(os.path.join(OUT_DIR, "sm_threats_by_year.json"), "w") as fh:
    json.dump(sm_year, fh)
sm["_mo"] = sm["_dt"].dt.month
sm_month = [dict(month=int(m), **rec) for m, rec in sorted(_sm_bucket("_mo").items())]
with open(os.path.join(OUT_DIR, "sm_threats_by_month.json"), "w") as fh:
    json.dump(sm_month, fh)
# by actual year-month (continuous, 0-filled — for the rolling 3-monthly view)
sm_yearmonth = []
if len(sm):
    sm["_ymp"] = sm["_dt"].dt.to_period("M")
    for p in pd.period_range(sm["_ymp"].min(), sm["_ymp"].max(), freq="M"):
        g = sm[sm["_ymp"] == p]
        entry = {"ym": str(p), "year": int(p.year),
                 "total": int(g["_conf"].sum()) if len(g) else 0, "corpus_total": int(len(g))}
        for c in sm_cats:
            entry[c] = int(g[c].notna().sum()) if len(g) else 0
        sm_yearmonth.append(entry)
with open(os.path.join(OUT_DIR, "sm_threats_by_yearmonth.json"), "w") as fh:
    json.dump(sm_yearmonth, fh)
print(f"  wrote sm_threats_by_yearmonth.json ({len(sm_yearmonth)} months)")
# by target country (confrontational pieces only) → ISO3 via the MFA CONF_ISO_MAP
sm_country = {}
for _, r in sm[sm["_conf"]].iterrows():
    iso = CONF_ISO_MAP.get(str(r.get("subject_country", "") or "").strip())
    if not iso:
        continue
    rec = sm_country.setdefault(iso, {"total": 0, **{c: 0 for c in sm_cats}})
    rec["total"] += 1
    for c in sm_cats:
        if pd.notna(r[c]):
            rec[c] += 1
with open(os.path.join(OUT_DIR, "sm_threats_by_country.json"), "w") as fh:
    json.dump(sm_country, fh)
# also by target country by year (for year slider on confrontation map)
sm_country_year = {}
for _, r in sm[sm["_conf"]].iterrows():
    iso = CONF_ISO_MAP.get(str(r.get("subject_country", "") or "").strip())
    if not iso:
        continue
    yr = r["_yr"]
    if iso not in sm_country_year:
        sm_country_year[iso] = {}
    if yr not in sm_country_year[iso]:
        sm_country_year[iso][yr] = {"total": 0, **{c: 0 for c in sm_cats}}
    sm_country_year[iso][yr]["total"] += 1
    for c in sm_cats:
        if pd.notna(r[c]):
            sm_country_year[iso][yr][c] += 1
with open(os.path.join(OUT_DIR, "sm_threats_by_country_year.json"), "w") as fh:
    json.dump(sm_country_year, fh)
print(f"  wrote sm_threats_by_year ({len(sm_year)} yrs) / by_month / by_country ({len(sm_country)} countries) / by_country_year ({len(sm_country_year)} countries)")

# ── I: Recent daily deep-dive feed (last ~5 weeks, MFA + PD + CD) ───────────────
# Powers the "Last month" tab on the Confrontation-over-time chart: one bar per day,
# per outlet, coloured by the strongest confrontational act. MFA shows EVERY Q&A in the
# window (confrontational categories + an "Other Q&A" residual for non-confrontational
# exchanges); the commentary outlets (People's Daily — 钟声 flagged; China Daily) show the
# pieces flagged confrontational. Rolling window, regenerated weekly by the pipeline.
print("Building recent_daily (last-month deep-dive) …")
RECENT_DAYS = 34
SEV_ORDER = ["Active threat", "Threat", "Demand", "Representations", "Urge", "Request"]  # strongest→weakest
MFA_LM_CATS = ["Active threat", "Threat", "Demand", "Urge", "Request", "Representations"]  # display/stack order
SM_LM_CATS  = ["Active threat", "Threat", "Demand", "Urge", "Representations"]             # Request excluded

def _excl_cat(row, cats):
    for c in SEV_ORDER:
        if c in cats and pd.notna(row.get(c)):
            return c
    return None

def _clip(v, n):
    s = "" if v is None else str(v).strip()
    if s in ("-", "nan"):
        s = ""
    return s[:n]

# Common rolling window = last RECENT_DAYS ending at the latest date across all sources
_lm_masters = {}
_lm_dates = [df["_date_p"].max()]
for _k, _rel in [("pd_zh", "pd_zh/pd_commentary_master.csv"),
                 ("pd_en", "pd_en/pd_en_commentary_master.csv"),
                 ("cd",    "cd/cd_commentary_master.csv")]:
    _p = os.path.join(COMMENTARY_DIR, _rel)
    if os.path.exists(_p):
        _cm = pd.read_csv(_p, low_memory=False)
        _cm["_dt"] = pd.to_datetime(_cm["date"], errors="coerce")
        _lm_masters[_k] = _cm[_cm["_dt"].notna()]
        _lm_dates.append(_lm_masters[_k]["_dt"].max())
_lm_latest = max(d for d in _lm_dates if pd.notna(d))
_lm_cut = _lm_latest - pd.Timedelta(days=RECENT_DAYS)
_lm_latest_s, _lm_cut_s = _lm_latest.strftime("%Y-%m-%d"), _lm_cut.strftime("%Y-%m-%d")

# MFA — every Q&A in the window (confrontational categorised, else "Other Q&A")
_mw = df[(df["_date_p"] >= _lm_cut) & (df["_date_p"] <= _lm_latest)].copy()

def _mfa_lm_item(r):
    cat = _excl_cat(r, MFA_LM_CATS)
    return {
        "date": r["_date_p"].strftime("%Y-%m-%d"),
        "cat":  cat or "Other Q&A",
        "codes": [c for c in MFA_LM_CATS if pd.notna(r.get(c))],
        "sp":   _clip(r.get("spokesperson", ""), 40),
        "who":  _clip(r.get("who_asked", ""), 60),
        "loc":  _clip(r.get("q_loc", ""), 60),
        "q":    _clip(r.get("question", ""), 280),
        "a":    _clip(r.get("answer", ""), 540),
        "link": _clip(r.get("link", ""), 200),
    }

_mw_sorted = _mw.sort_values("_date_p")
mfa_lm_items = [_mfa_lm_item(r) for _, r in _mw_sorted.iterrows()]
# MFA — domestic-question subset (question from a domestic Chinese outlet; who_asked populated 2020+)
mfa_dom_lm_items = [_mfa_lm_item(r) for _, r in _mw_sorted[_mw_sorted["press_cat"].isin(DOMESTIC_PRESS)].iterrows()]

def _commentary_lm_items(keys):
    out = []
    for key in keys:
        cm = _lm_masters.get(key)
        if cm is None:
            continue
        present = [c for c in SM_LM_CATS if c in cm.columns]
        w = cm[(cm["_dt"] >= _lm_cut) & (cm["_dt"] <= _lm_latest)]
        for _, r in w.iterrows():                                # EVERY piece in the window (not just confrontational)
            col = _clip(r.get("column", ""), 40)
            cat = _excl_cat(r, present)
            out.append({
                "date": r["_dt"].strftime("%Y-%m-%d"),
                "cat":  cat or "Other commentary",              # non-confrontational → grey residual
                "codes": [c for c in present if pd.notna(r.get(c))],
                "src":  key,
                "column": col,
                "zs":   ("钟声" in col) or (key == "pd_en"),   # flagship column (English pd_en is the translated 钟声)
                "title": _clip(r.get("title", ""), 200),
                "country": _clip(r.get("subject_country", ""), 60),
                "zh":   key == "pd_zh",                          # Chinese → render a Google-Translate link
                "url":  _clip(r.get("url", ""), 200),
            })
    out.sort(key=lambda x: x["date"])
    return out

recent_daily = {
    "latest": _lm_latest_s,
    "cutoff": _lm_cut_s,
    # keys match the Confrontation-over-time source toggle (mfa / mfa_dom / sm) so the same
    # selector drives every tab, including "Last month". State media = PD (Chinese + English) + CD.
    "outlets": [
        {"key": "mfa", "label": "MFA press conference", "unit": "Q&A exchanges",
         "residual": True, "cats": MFA_LM_CATS + ["Other Q&A"], "items": mfa_lm_items},
        {"key": "mfa_dom", "label": "MFA — domestic questions", "unit": "Q&A exchanges",
         "residual": True, "cats": MFA_LM_CATS + ["Other Q&A"], "items": mfa_dom_lm_items},
        {"key": "sm",  "label": "State media commentary", "unit": "commentary pieces",
         "residual": True, "highlight": "钟声", "cats": SM_LM_CATS + ["Other commentary"],
         "items": _commentary_lm_items(["pd_zh", "pd_en", "cd"])},
    ],
}
with open(os.path.join(OUT_DIR, "recent_daily.json"), "w") as fh:
    json.dump(recent_daily, fh, ensure_ascii=False)
print(f"  wrote recent_daily.json (window {_lm_cut_s} … {_lm_latest_s}; "
      f"MFA {len(mfa_lm_items)} exchanges, "
      f"MFA-domestic {len(mfa_dom_lm_items)}, "
      f"state-media {len(recent_daily['outlets'][2]['items'])} pieces)")

print("\nDone.")
