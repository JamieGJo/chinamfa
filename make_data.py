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
        "post": "Ambassador to France (2008–14); Ambassador to EU (2014–)",
        "note": "MFA spokesperson 2000–2006. Prior posts include Director-General, MFA Information Dept.",
    },
    "Zhang Qiyue": {
        "era": "Hu Jintao",
        "post": "Ambassador to Switzerland",
        "note": "MFA spokesperson 2001–2006. First female MFA spokesperson in the modern era.",
    },
    "Liu Jianchao": {
        "era": "Hu Jintao",
        "post": "Minister, CPC International Liaison Dept (2013–15); Ambassador to Philippines (2009–12)",
        "note": "MFA spokesperson 2003–2006. Joined MFA 1991.",
    },
    "Qin Gang": {
        "era": "Xi Jinping",
        "post": "Ambassador to US (2021–22); Foreign Minister Dec 2022 – Jul 2023",
        "note": "Born 1966. MFA spokesperson 2005–2010. Removed from office July 2023.",
    },
    "Jiang Yu": {
        "era": "Hu Jintao",
        "post": "Senior MFA official",
        "note": "MFA spokesperson 2006–2012.",
    },
    "Ma Zhaoxu": {
        "era": "Hu Jintao / Xi Jinping",
        "post": "Ambassador to US (2022–)",
        "note": "Born 1963. MFA spokesperson 2009–2012. Ambassador to UN Geneva (2015–2022). Ambassador to UN New York (2010–2015).",
    },
    "Liu Weimin": {
        "era": "Hu Jintao / Xi Jinping",
        "post": "Director-General, MFA Information Dept",
        "note": "MFA spokesperson 2011–2013.",
    },
    "Hong Lei": {
        "era": "Hu Jintao / Xi Jinping",
        "post": "Delegate to UNESCO, Paris",
        "note": "MFA spokesperson 2010–2016.",
    },
    "Geng Shuang": {
        "era": "Xi Jinping",
        "post": "Deputy Permanent Representative to UN, New York (2020–)",
        "note": "Born 1973. MFA spokesperson 2015–2020.",
    },
    "Lu Kang": {
        "era": "Xi Jinping",
        "post": "Ambassador to Algeria (2019–)",
        "note": "Born 1969. MFA spokesperson 2015–2019. Prior posts include Consul-General, San Francisco.",
    },
    "Hua Chunying": {
        "era": "Xi Jinping",
        "post": "Vice Foreign Minister (2022–)",
        "note": "Born 1970. MFA spokesperson 2012–2022. Assistant Foreign Minister 2019–2022. Joined MFA 1994.",
    },
    "Zhao Lijian": {
        "era": "Xi Jinping",
        "post": "Director-General, MFA Oceanic Affairs (2023–)",
        "note": "Born 1972. MFA spokesperson 2019–2023. Prior posts include Deputy Chief of Mission, Pakistan (2015–19).",
    },
    "Wang Wenbin": {
        "era": "Xi Jinping",
        "post": "Active spokesperson (2020–)",
        "note": "MFA spokesperson from August 2020.",
    },
    "Mao Ning": {
        "era": "Xi Jinping",
        "post": "Active spokesperson (2022–)",
        "note": "MFA spokesperson from September 2022.",
    },
    "Lin Jian": {
        "era": "Xi Jinping",
        "post": "Active spokesperson (2023–)",
        "note": "MFA spokesperson from February 2023.",
    },
    "Guo Jiakun": {
        "era": "Xi Jinping",
        "post": "Active spokesperson (2024–)",
        "note": "MFA spokesperson from March 2024.",
    },
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
    cards.append({
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
    })

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
        tags.append("Attacked/criticised")
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

articles = []
for _, row in explorer_df.iterrows():
    tags = build_tags(row)
    is_io = bool(
        "international order" in str(row.get("answer", "")).lower()
    )
    if is_io and "IO mention" not in tags:
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
    entry = {"year": int(yr), "total": int(conf_mask.sum())}
    for col in ALL_CONF_COLS:
        entry[col] = int(rows[col].notna().sum()) if col in rows.columns else 0
    threats_by_year.append(entry)

with open(os.path.join(OUT_DIR, "threats_by_year.json"), "w") as f:
    json.dump(threats_by_year, f)
print(f"  wrote threats_by_year.json ({len(threats_by_year)} years)")

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

threats_by_country = {}
conf_rows = df[df[ALL_CONF_COLS].notna().any(axis=1)]
for _, row in conf_rows.iterrows():
    q_raw = str(row.get("q_loc", ""))
    if q_raw in ("nan", ""):
        continue
    for loc in [l.strip() for l in q_raw.split(";") if l.strip() not in ("-", "nan", "")]:
        iso = CONF_ISO_MAP.get(loc)
        if not iso:
            continue
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

print("\nDone.")
