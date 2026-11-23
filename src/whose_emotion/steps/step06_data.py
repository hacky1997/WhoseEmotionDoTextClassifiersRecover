# Step file generated from the archived notebook by tools/export_notebook.py.
# Run it through run.py, which executes every step in order in one shared namespace,
# as "Run all" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md

# =====================================================================================
# CELL 6 — DATA LOADING AND VALIDATION
# All corpora are downloaded from their public sources, checksummed, and validated against the counts
# reported in the paper. The first session writes shared/data_manifest.json; later sessions abort if any
# file checksum differs (a silent dataset change would invalidate comparisons across sessions).
# =====================================================================================
import re, tarfile, urllib.request
from collections import Counter, defaultdict
import pandas as pd

DATA_DIR = Path(WORK_DIR) / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
HF_PARQUET = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
MELD_CSV = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/{split}_sent_emo.csv"
ED_TGZ = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"

SETFIT = ["sadness", "joy", "love", "anger", "fear", "surprise"]
SETFIT2ID = {n: i for i, n in enumerate(SETFIT)}
GOEMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
              "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
              "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
GOEMOTIONS_TO_SETFIT = {n: SETFIT2ID[n] for n in SETFIT}
MELD_TO_SETFIT = {"sadness": 0, "joy": 1, "anger": 3, "fear": 4, "surprise": 5}
ISEAR_TO_SETFIT = {"sadness": 0, "joy": 1, "anger": 3, "fear": 4}
ED_TO_SETFIT = {**{k: 1 for k in ["joyful", "excited", "content", "hopeful", "grateful", "proud", "impressed", "anticipating"]},
                **{k: 0 for k in ["sad", "devastated", "ashamed", "guilty", "lonely", "nostalgic", "disappointed", "sentimental"]},
                **{k: 3 for k in ["angry", "furious", "jealous", "disgusted", "annoyed"]},
                **{k: 4 for k in ["afraid", "apprehensive", "terrified", "anxious"]},
                **{k: 5 for k in ["surprised", "astonished"]}}
PRESENT_CLASSES = {"setfit": list(range(6)), "goemotions": list(range(6)), "meld": [0, 1, 3, 4, 5], "isear": [0, 1, 3, 4], "ed": [0, 1, 3, 4, 5]}
NUM_LABELS = {"setfit": 6, "tweeteval": 4, "ed": 6, "meld_target": 6, "isear_target": 6, "goemotions_target": 6}
# ED and the target-trained corpora keep SetFit label ids (absent classes are never seen); TweetEval has its own 4 labels
EXPECTED = {  # counts verified against the public data and reported in the paper
    "setfit": {"train": 16000, "validation": 2000, "test": 2000, "test_class_counts": [581, 695, 159, 275, 224, 66]},
    "tweeteval": {"train": 3257, "validation": 374, "test": 1421},
    "ed": {"train": 16343, "validation": 2284, "test": 2120, "test_class_counts": {0: 614, 1: 676, 3: 410, 4: 293, 5: 127}},
    "goemotions": 638, "goemotions_train": 5272, "goemotions_validation": 625, "meld": 1286, "meld_train": 5008, "meld_validation": 617,
    "isear": 4329, "isear_rows": 7516, "isear_split": (3030, 649, 650),
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(url: str, name: str) -> Path:
    p = DATA_DIR / name
    if p.exists() and p.stat().st_size > 0:
        return p
    try:
        tmp = Path(str(p) + ".part")
        urllib.request.urlretrieve(url, tmp)
        shutil.move(str(tmp), str(p))
    except Exception as e:
        raise RuntimeError(f"Download failed for {url}: {e}") from e
    assert p.stat().st_size > 0, f"empty download: {url}"
    return p

def parquet(repo: str, cfg: str, split: str, name: str, rev: str = "refs%2Fconvert%2Fparquet") -> pd.DataFrame:
    return pd.read_parquet(fetch(HF_PARQUET.format(repo=repo, rev=rev, path=f"{cfg}/{split}/0000.parquet"), name))

def load_data() -> dict:
    D, files = {}, {}
    # SetFit/emotion (source) --------------------------------------------------------------
    D["setfit"] = {}
    for split in ["train", "validation", "test"]:
        df = parquet("SetFit/emotion", "default", split, f"setfit_{split}.parquet")
        for i, t in df.groupby("label").label_text.first().items():          # label order check against the data itself
            assert SETFIT[int(i)] == t, f"SetFit label order mismatch: id {i} is '{t}', expected '{SETFIT[int(i)]}'"
        assert df.text.notna().all() and df.label.notna().all(), f"nulls in SetFit {split}"
        D["setfit"][split] = {"text": df.text.tolist(), "label": df.label.astype(int).tolist()}
        files[f"setfit_{split}.parquet"] = sha256_file(DATA_DIR / f"setfit_{split}.parquet")
        assert len(df) == EXPECTED["setfit"][split], f"SetFit {split}: {len(df)} rows"
    c = Counter(D["setfit"]["test"]["label"])
    assert [c[i] for i in range(6)] == EXPECTED["setfit"]["test_class_counts"], f"SetFit test class counts {c}"
    # TweetEval emotion ------------------------------------------------------------------
    D["tweeteval"] = {}
    for split in ["train", "validation", "test"]:
        df = parquet("cardiffnlp/tweet_eval", "emotion", split, f"tweeteval_{split}.parquet")
        assert len(df) == EXPECTED["tweeteval"][split], f"TweetEval {split}: {len(df)} rows"
        assert set(df.label.unique()) <= {0, 1, 2, 3}
        D["tweeteval"][split] = {"text": df.text.tolist(), "label": df.label.astype(int).tolist()}
        files[f"tweeteval_{split}.parquet"] = sha256_file(DATA_DIR / f"tweeteval_{split}.parquet")
    # GoEmotions: single-label test comments whose label is one of the six source classes ---------
    D["goemotions"] = {}
    for split, key in [("test", "goemotions"), ("train", "goemotions_train"), ("validation", "goemotions_validation")]:
        df = parquet("google-research-datasets/go_emotions", "simplified", split, f"goemo_{split}.parquet")
        keep = df[df.labels.map(len) == 1]
        names = keep.labels.map(lambda l: GOEMOTIONS[int(l[0])])
        keep = keep[names.isin(GOEMOTIONS_TO_SETFIT)]
        D["goemotions"][split] = {"text": keep.text.tolist(), "label": [GOEMOTIONS_TO_SETFIT[GOEMOTIONS[int(l[0])]] for l in keep.labels]}
        assert len(keep) == EXPECTED[key], f"GoEmotions strict {split}: {len(keep)}"
        files[f"goemo_{split}.parquet"] = sha256_file(DATA_DIR / f"goemo_{split}.parquet")
    # MELD: text of utterances with one of the five shared labels --------------------------------
    D["meld"] = {}
    for split, fname in [("train", "train"), ("validation", "dev"), ("test", "test")]:
        p = fetch(MELD_CSV.format(split=fname), f"meld_{fname}.csv")
        m = pd.read_csv(p, encoding_errors="replace")
        m = m[m.Emotion.str.lower().isin(MELD_TO_SETFIT)]
        D["meld"][split] = {"text": m.Utterance.astype(str).tolist(), "label": [MELD_TO_SETFIT[e.lower()] for e in m.Emotion]}
        files[f"meld_{fname}.csv"] = sha256_file(p)
    assert len(D["meld"]["test"]["text"]) == EXPECTED["meld"] and len(D["meld"]["train"]["text"]) == EXPECTED["meld_train"]
    assert len(D["meld"]["validation"]["text"]) == EXPECTED["meld_validation"]
    # ISEAR (Hugging Face copy, pinned revision; no licence statement in that repository) ---------
    # WARNING: the gsri-18 repository provides no dataset card or licence; confirm terms of use before release.
    isr = parquet(ISEAR_HF_REPO, "default", "train", "isear_gsri18.parquet")
    assert len(isr) == EXPECTED["isear_rows"], f"ISEAR rows: {len(isr)}"
    isr = isr[isr.emotion.str.lower().str.strip().isin(ISEAR_TO_SETFIT)]
    D["isear"] = {"test": {"text": isr.content.astype(str).tolist(), "label": [ISEAR_TO_SETFIT[e.lower().strip()] for e in isr.emotion]}}
    assert len(isr) == EXPECTED["isear"], f"ISEAR four-class items: {len(isr)}"
    from sklearn.model_selection import train_test_split          # stratified 70/15/15 split for target training
    tr_df, tmp = train_test_split(isr, test_size=0.3, stratify=isr.emotion, random_state=ISEAR_SPLIT_SEED)
    va_df, te_df = train_test_split(tmp, test_size=0.5, stratify=tmp.emotion, random_state=ISEAR_SPLIT_SEED)
    assert (len(tr_df), len(va_df), len(te_df)) == EXPECTED["isear_split"], f"ISEAR split {(len(tr_df), len(va_df), len(te_df))}"
    D["isear_ft"] = {n: {"text": d.content.astype(str).tolist(), "label": [ISEAR_TO_SETFIT[e.lower().strip()] for e in d.emotion]}
                     for n, d in [("train", tr_df), ("validation", va_df), ("test", te_df)]}
    D["isear_ft"]["test"]["pos_in_full"] = isr.index.get_indexer(te_df.index).tolist()   # rows of the 4,329-item transfer set
    assert min(D["isear_ft"]["test"]["pos_in_full"]) >= 0 and len(set(D["isear_ft"]["test"]["pos_in_full"])) == len(te_df)
    files["isear_gsri18.parquet"] = sha256_file(DATA_DIR / "isear_gsri18.parquet")
    # EmpatheticDialogues: situation description, one item per conversation, official splits -------
    p = fetch(ED_TGZ, "empatheticdialogues.tar.gz")
    files["empatheticdialogues.tar.gz"] = sha256_file(p)
    D["ed"] = {}
    with tarfile.open(p) as tf:
        for split, fname in [("train", "train.csv"), ("validation", "valid.csv"), ("test", "test.csv")]:
            member = next(m for m in tf.getmembers() if m.name.endswith(fname))
            lines = tf.extractfile(member).read().decode("utf-8").splitlines()
            head = lines[0].split(",")
            ic, ie, ip = head.index("conv_id"), head.index("context"), head.index("prompt")
            seen, text, lab = set(), [], []
            for line in lines[1:]:
                f = line.split(",")              # ParlAI format: commas inside fields are encoded as _comma_
                if len(f) <= max(ic, ie, ip) or f[ic] in seen:
                    continue
                seen.add(f[ic])
                if f[ie] in ED_TO_SETFIT:
                    text.append(f[ip].replace("_comma_", ",")); lab.append(ED_TO_SETFIT[f[ie]])
            D["ed"][split] = {"text": text, "label": lab}
            assert len(text) == EXPECTED["ed"][split], f"ED {split}: {len(text)} items"
    c = Counter(D["ed"]["test"]["label"])
    assert dict(c) == EXPECTED["ed"]["test_class_counts"], f"ED test class counts {dict(c)}"
    D["meld_target"] = D["meld"]
    D["goemotions_target"] = D["goemotions"]
    D["isear_target"] = D["isear_ft"]
    return D, files

def check_manifest(files: dict) -> None:
    path = f"{SHARED_PREFIX}/data_manifest.json"
    p = hub_download(path)
    if p is None:
        lp = local_path(path); write_json(lp, {"files": files, "created": datetime.now().isoformat()})
        hub_upload_file(lp, path)
        logger.info("Data manifest created on the Hub")
        return
    with open(p) as f:
        old = json.load(f)["files"]
    diff = {k: (old.get(k), v) for k, v in files.items() if k in old and old[k] != v}
    if diff:
        raise RuntimeError(f"Dataset files differ from the manifest written by an earlier session: {list(diff)}. Aborting.")
    new = {k: v for k, v in files.items() if k not in old}
    if new:                                                     # files added by a newer notebook version: append, never overwrite
        lp = local_path(path); write_json(lp, {"files": {**old, **new}, "created": datetime.now().isoformat()})
        hub_upload_file(lp, path); logger.info(f"Data manifest extended with {sorted(new)}")
    logger.info("Data checksums match the manifest")

# ---- lexical cue helpers (pre-registered definitions) ------------------------------------------
FEEL_RE = re.compile(r"\b(?:feel|feels|feeling|felt)\b", re.I)
NEXT_SKIP_RE = re.compile(r"\b(?:feel|feeling|feels|felt)\s+(?:(?:" + "|".join(INTENSIFIERS) + r")\s+)*([a-z]+)", re.I)
WORD_RE = re.compile(r"[a-z']+")
STRIP = ".,!?;:\"()"

def next_word(text: str) -> str:
    m = NEXT_SKIP_RE.search(text.lower())
    return m.group(1) if m else "<none>"

def cue_vocabulary(texts, labels) -> set:
    by = defaultdict(Counter)
    for t, y in zip(texts, labels):
        w = next_word(t)
        if w != "<none>":
            by[w][int(y)] += 1
    return {w for w, c in by.items() if sum(c.values()) >= CUE_MIN_COUNT and c.most_common(1)[0][1] / sum(c.values()) >= CUE_MIN_PURITY}

def is_explicit(text: str) -> bool:
    return any(w in CUE_VOCAB for w in WORD_RE.findall(text.lower()))

def mask_cues(text: str, mask: str):
    words, n = text.split(), 0
    for i, w in enumerate(words):
        if w.lower().strip(STRIP) in CUE_VOCAB:
            words[i], n = mask, n + 1
    return " ".join(words), n

def mask_random_words(text: str, mask: str, n: int, seed: int) -> str:
    words = text.split()
    if len(words) <= n or n <= 0:
        return text
    rng = random.Random(f"{seed}|{text}")
    for i in rng.sample(range(len(words)), n):
        words[i] = mask
    return " ".join(words)

def mask_random_noncue(text: str, mask: str, n: int, seed: int) -> str:
    """CRI control: mask n randomly chosen words that are NOT cue words (so the control never removes a cue)."""
    words = text.split()
    cand = [i for i, w in enumerate(words) if w.lower().strip(STRIP) not in CUE_VOCAB]
    if len(cand) < n or n <= 0:
        return text
    rng = random.Random(f"{seed}|{text}")
    for i in rng.sample(cand, n):
        words[i] = mask
    return " ".join(words)

def mask_feel_phrase(text: str, mask: str) -> str:
    return NEXT_SKIP_RE.sub(lambda m: f"{mask} {mask}", text)

def load_setfit_large(base: dict, n_train: int, seed: int) -> dict:
    p = fetch(HF_PARQUET.format(repo="dair-ai/emotion", rev="refs%2Fconvert%2Fparquet", path="unsplit/train/0000.parquet"),
              "dairai_unsplit.parquet")
    df = pd.read_parquet(p).drop_duplicates("text")
    held_out = set(base["validation"]["text"]) | set(base["test"]["text"])
    df = df[~df.text.isin(held_out)]
    assert n_train <= len(df), f"requested {n_train} > pool {len(df)}"
    rng = np.random.default_rng(seed)
    frac = n_train / len(df)
    s = pd.concat([g.iloc[rng.permutation(len(g))[: int(round(frac * len(g)))]] for _, g in df.groupby("label")])
    assert not set(s.text) & held_out, "leakage: sampled training text occurs in validation or test"
    return {"train": {"text": s.text.tolist(), "label": s.label.astype(int).tolist()}, "validation": base["validation"], "test": base["test"]}

DATA, DATA_FILES = load_data()
check_manifest(DATA_FILES)
CUE_VOCAB = cue_vocabulary(DATA["setfit"]["train"]["text"], DATA["setfit"]["train"]["label"])
assert len(CUE_VOCAB) == EXPECTED_CUE_VOCAB_SIZE, (f"Cue vocabulary has {len(CUE_VOCAB)} words, the pre-registered value is "
                                                   f"{EXPECTED_CUE_VOCAB_SIZE}. The data or the definition changed. Aborting.")
EXPLICIT = {name: np.array([is_explicit(t) for t in split["text"]])
            for name, split in [("setfit_test", DATA["setfit"]["test"]), ("setfit_validation", DATA["setfit"]["validation"]),
                                ("goemotions", DATA["goemotions"]["test"]), ("meld", DATA["meld"]["test"]),
                                ("isear", DATA["isear"]["test"]), ("ed_test", DATA["ed"]["test"]), ("tweeteval_test", DATA["tweeteval"]["test"]),
                                ("isear_target_test", DATA["isear_ft"]["test"])]}
cue_path = local_path(f"{SHARED_PREFIX}/cue_vocabulary.json")
if f"{SHARED_PREFIX}/cue_vocabulary.json" not in HUB_FILES:
    write_json(cue_path, sorted(CUE_VOCAB)); hub_upload_file(cue_path, f"{SHARED_PREFIX}/cue_vocabulary.json")
for corpus in ["setfit", "tweeteval", "ed"]:
    for split, d in DATA[corpus].items():
        lens = np.array([len(t.split()) for t in d["text"]])
        logger.info(f"{corpus}/{split}: n={len(d['text'])} labels={dict(sorted(Counter(d['label']).items()))} "
                    f"words mean={lens.mean():.1f} p95={np.percentile(lens, 95):.0f}")
for name in ["goemotions", "meld", "isear", "isear_ft"]:
    logger.info(f"{name}/test: n={len(DATA[name]['test']['text'])} labels={dict(sorted(Counter(DATA[name]['test']['label']).items()))}")
logger.info(f"Cue vocabulary: {len(CUE_VOCAB)} words | explicit share: " +
            ", ".join(f"{k}={v.mean():.3f}" for k, v in EXPLICIT.items()))

# ---- crowd-enVENT (used only by the cev_source, cev_masked and cev_target stages) ------------------------------
# WARNING: the crowd-enVENT release has no explicit licence (only a citation request); confirm terms before publication.
def load_cev():
    import io, zipfile
    from sklearn.model_selection import train_test_split
    p = fetch(CEV_URL, "crowd-enVent2023.zip")
    with zipfile.ZipFile(p) as zf:
        names = zf.namelist()
        def read(member):
            return pd.read_csv(io.BytesIO(zf.read(next(n for n in names if n.endswith(member)))), sep="\t")
        gen, val = read("crowd-enVent_generation.tsv"), read("crowd-enVent_validation.tsv")
    assert len(gen) == 6600 and len(val) == 6000, f"crowd-enVENT sizes changed: {len(gen)}, {len(val)}"
    g = gen[gen.emotion.isin(CEV_TO_SETFIT)].sort_values("text_id").reset_index(drop=True)
    assert len(g) == 2750 and not g.text_id.duplicated().any(), f"five-class texts: {len(g)}"
    assert g.generated_text.notna().all() and g.hidden_emo_text.notna().all(), "missing text in crowd-enVENT"
    assert not g[CEV_APPRAISALS].isna().any().any(), "missing appraisal ratings"
    orig = g.generated_text.astype(str).str.strip().tolist()
    masked = g.hidden_emo_text.astype(str).str.strip().tolist()
    changed = np.array([a != b for a, b in zip(orig, masked)])
    assert changed.sum() == 863, f"author-masked pairs: {changed.sum()}"
    y = np.array([CEV_TO_SETFIT[e] for e in g.emotion])
    in_reader = g.text_id.isin(set(val.text_id)).to_numpy()
    reader_pos = np.flatnonzero(in_reader)
    assert len(reader_pos) == 500 and changed[reader_pos].sum() == 163, "reader-annotated subset changed"
    vv = val[val.text_id.isin(set(g.text_id[in_reader]))]
    by = {tid: grp.emotion.tolist() for tid, grp in vv.groupby("text_id")}
    assert all(len(v) == 5 for v in by.values()), "expected five readers per text"
    # maj, item_acc, guesses = [], [], []
    maj, item_acc, guesses, cons = [], [], [], []
    for tid, ytrue in zip(g.text_id[in_reader], y[in_reader]):
        c = Counter(by[tid]).most_common()
        cons.append(c[0][1] / len(by[tid]))                 # share of readers choosing the modal emotion (13 classes)
        # -1: top choice tied between labels; -2: majority label outside the five shared classes
        maj.append(-1 if len(c) > 1 and c[0][1] == c[1][1] else CEV_TO_SETFIT.get(c[0][0], -2))
        gl = [CEV_TO_SETFIT.get(e, -2) for e in by[tid]]
        guesses.append(gl); item_acc.append(float(np.mean([x == ytrue for x in gl])))
    maj = np.array(maj)
    assert (maj == -1).sum() == 56, f"reader-majority ties: {(maj == -1).sum()}"
    rest = g[~in_reader]                                   # upper-bound model: texts the readers did not see
    tr, va = train_test_split(rest, test_size=CEV_VAL_FRACTION, stratify=rest.emotion, random_state=CEV_SPLIT_SEED)
    assert (len(tr), len(va)) == (2025, 225), f"upper-bound split {(len(tr), len(va))}"
    split = lambda d: {"text": d.generated_text.astype(str).str.strip().tolist(), "label": [CEV_TO_SETFIT[e] for e in d.emotion]}
    data = {"train": split(tr), "validation": split(va), "test": split(g.iloc[reader_pos])}
    cev = {"text_orig": orig, "text_masked": masked, "label": y, "text_id": g.text_id.to_numpy(), "changed": changed,
           "reader_pos": reader_pos, "reader_majority": maj, "reader_item_acc": np.array(item_acc),
           "reader_guesses": np.array(guesses), "reader_consensus": np.array(cons), "appraisal": g[CEV_APPRAISALS].to_numpy(float)}
    return data, cev, sha256_file(p)

CEV_DATA, CEV, _cev_sha = load_cev()
DATA["cev_target"] = CEV_DATA
NUM_LABELS["cev_target"] = 6                       # SetFit label ids; love is absent
PRESENT_CLASSES["cev"] = list(CEV_TO_SETFIT.values())
check_manifest({"crowd-enVent2023.zip": _cev_sha})
logger.info(f"crowd-enVENT: {len(CEV['label'])} five-class texts, {int(CEV['changed'].sum())} author-masked pairs, "
            f"{len(CEV['reader_pos'])} reader-annotated ({int((CEV['reader_majority'] >= 0).sum())} with a reader majority), "
            f"upper-bound split {len(CEV_DATA['train']['text'])}/{len(CEV_DATA['validation']['text'])}/{len(CEV_DATA['test']['text'])}")

# ---- enISEAR (v8; evaluation only: eni_* stages, llm_transfer task enisear, Cell 9e) ------------------------------------
def load_enisear() -> tuple:
    import io, zipfile
    p = fetch(ENI_URL, "deISEARenISEAR.zip")
    sha = sha256_file(p)
    assert sha == ENI_SHA256, f"enISEAR archive changed: SHA-256 {sha}, expected {ENI_SHA256}"
    with zipfile.ZipFile(p) as zf:
        e = pd.read_csv(io.BytesIO(zf.read("deISEARenISEAR/enISEAR.tsv")), sep="\t")
        ev = pd.read_csv(io.BytesIO(zf.read("deISEARenISEAR/enISEAR_validation.tsv")), sep="\t")
    assert len(e) == 1001 and e.Sentence_id.is_unique, f"enISEAR descriptions: {len(e)}"
    assert set(e.Prior_Emotion) == set(ENI_EMOTIONS) and e.Prior_Emotion.value_counts().eq(143).all(), "143 descriptions per emotion expected"
    assert (e[ENI_EMOTIONS].sum(axis=1) == 5).all(), "every description should have five reader votes"
    raw = {sid: Counter(str(a).strip().capitalize() for a in grp.Annotation) for sid, grp in ev.groupby("Sentence_id")}
    agg = e.set_index("Sentence_id")[ENI_EMOTIONS]
    assert set(raw) == set(agg.index) and all(int(agg.loc[s, k]) == c.get(k, 0) for s, c in raw.items() for k in ENI_EMOTIONS), \
        "aggregated reader votes in enISEAR.tsv differ from enISEAR_validation.tsv"
    e4 = e[e.Prior_Emotion.isin(ENI_TO_SETFIT)].sort_values("Sentence_id").reset_index(drop=True)
    raw_text = e4.Sentence.astype(str).str.strip()
    n_unicode = int(raw_text.str.contains("\u2026", regex=False).sum())
    text = raw_text.str.replace("\u2026", ENI_MARKER, regex=False).tolist()
    y = e4.Prior_Emotion.map(ENI_TO_SETFIT).to_numpy()
    votes = e4[ENI_EMOTIONS].to_numpy(int)
    maj = []
    for row in votes:
        order = np.argsort(-row, kind="stable")
        # -1: top choice tied between labels; -2: majority label outside the four shared classes (as for crowd-enVENT)
        maj.append(-1 if row[order[0]] == row[order[1]] else ENI_TO_SETFIT.get(ENI_EMOTIONS[order[0]], -2))
    maj = np.array(maj)
    item_acc = votes[np.arange(len(e4)), [ENI_EMOTIONS.index(lab) for lab in e4.Prior_Emotion]] / 5.0
    assert len(e4) == 572 and n_unicode == 166 and (maj == -1).sum() == 17 and (maj == -2).sum() == 11, \
        f"enISEAR four-class set changed: n={len(e4)}, marker={n_unicode}, ties={(maj == -1).sum()}, outside={(maj == -2).sum()}"
    eni = {"text": text, "label": y, "sentence_id": e4.Sentence_id.to_numpy(), "reader_majority": maj,
           "reader_item_acc": item_acc, "reader_votes": votes, "n_unicode_marker": n_unicode}
    return eni, sha

ENI, _eni_sha = load_enisear()
PRESENT_CLASSES["eni"] = list(ENI_TO_SETFIT.values())
check_manifest({"deISEARenISEAR.zip": _eni_sha})
_ok = ENI["reader_majority"] >= 0
logger.info(f"enISEAR: {len(ENI['label'])} four-class descriptions ({ENI['n_unicode_marker']} markers normalized to '{ENI_MARKER}'), "
            f"{int(_ok.sum())} with a unique reader majority in the four classes, "
            f"{int((ENI['reader_majority'][_ok] == ENI['label'][_ok]).sum())} of them matching the writer's label")
