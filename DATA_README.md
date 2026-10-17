# NICU project - shared data extracts

Exported 2026-09-27 from the PostgreSQL database `natality2024` (source of truth).
Built by `project/sql/feature_view.sql` from the NCHS 2024 Natality Public Use File.
If a rule or the cohort changes, the SQL is updated and these files are re-exported - please do not hand-edit them.

## Files

| File | Rows | What it is |
|---|---|---|
| `nicu_sample_300k.csv` | 300,000 | Stratified random sample of the cohort (same NICU rate). Use this for EDA, KNN, neural nets, tuning. |
| `nicu_features.parquet` | 3,537,490.0 | Full cohort. Use for final Decision Tree / Naive Bayes runs. `pd.read_parquet()` |
| `codebook.csv` | - | `field, code, label` - decode any coded column |
| `field_dictionary.csv` | - | every column: original NCHS field name, tier, type, description |

## Cohort

U.S. (50 states + DC) births in 2024 that occurred in a **hospital**, in a state that reports the NICU item,
with the item answered Y or N. 3,537,490.0 births, **10.04% admitted to NICU** (`nicu = True`).

## Columns and tiers (leakage policy)

* `birth_id` - row id (not a feature)
* `nicu` - **TARGET**: True = admitted to NICU (`AB_NICU = 'Y'`)
* **Tier A - known during pregnancy (54 cols)**: mager, mager9, mrace6, mhisp_r, mracehisp, meduc, dmar, mbstate_rec, pay_rec, wic, bmi, bmi_r, pwgt_r, m_ht_in, wtgain, wtgain_rec, cig_0, cig_1, cig_2, cig_3, smoker, priorlive, priordead, priorterm, lbo_rec, tbo_rec, illb_r11, ilp_r11, precare, precare5, previs, previs_rec, rf_pdiab, rf_gdiab, rf_phype, rf_ghype, rf_ehype, rf_ppterm, rf_inftr, rf_fedrg, rf_artec, rf_cesar, rf_cesarn, no_risks, ip_gon, ip_syph, ip_chlam, ip_hepb, ip_hepc, no_infec, dplural, fagecomb, frace6, feduc
* **Tier B - known at delivery, before NICU decision (21 cols)**: combgest, oegest_comb, gestrec3, dbwt, bwtr4, sex, apgar5, me_pres, me_rout, me_trial, rdmeth_rec, ld_indl, ld_augm, ld_ster, ld_antb, ld_chor, ld_anes, attend, dob_mm, dob_wk, dob_tt
* **Derived outcomes (Tier B)**: `preterm` (gestation < 37 wk), `lbw` (birth weight < 2500 g)
* **Tier C - leakage, deliberately NOT included**: other newborn conditions (AB_AVEN1/6, AB_SURF, AB_ANTI, AB_SEIZ),
  congenital anomalies (CA_*), infant transferred/living/breastfed, maternal morbidity (MM_*), reporting flags (F_*).

Prenatal-only model -> use Tier A. At-birth model -> Tier A + Tier B. Never use Tier C.

## Loading in pandas / Colab

```python
import pandas as pd

# codes must stay strings (leading zeros, e.g. mager9 = '01'); numerics load normally
STRING_COLS = ['mager9', 'mrace6', 'mhisp_r', 'mracehisp', 'meduc', 'dmar', 'mbstate_rec', 'pay_rec', 'wic', 'bmi_r', 'wtgain_rec', 'smoker', 'lbo_rec', 'tbo_rec', 'illb_r11', 'ilp_r11', 'precare5', 'previs_rec', 'rf_pdiab', 'rf_gdiab', 'rf_phype', 'rf_ghype', 'rf_ehype', 'rf_ppterm', 'rf_inftr', 'rf_fedrg', 'rf_artec', 'rf_cesar', 'no_risks', 'ip_gon', 'ip_syph', 'ip_chlam', 'ip_hepb', 'ip_hepc', 'no_infec', 'dplural', 'frace6', 'feduc', 'gestrec3', 'bwtr4', 'sex', 'me_pres', 'me_rout', 'me_trial', 'rdmeth_rec', 'ld_indl', 'ld_augm', 'ld_ster', 'ld_antb', 'ld_chor', 'ld_anes', 'attend', 'dob_mm', 'dob_wk']
df = pd.read_csv("nicu_sample_300k.csv", dtype={c: "string" for c in STRING_COLS},
                 true_values=["t", "True"], false_values=["f", "False"], low_memory=False)
codebook = pd.read_csv("codebook.csv", dtype="string")

def decode(col, field=None):
    """Map a coded column to its labels, e.g. decode('meduc') or decode('smoker', 'CIG_REC')."""
    field = (field or col).upper()
    m = codebook[codebook.field == field].set_index("code")["label"]
    return df[col].map(m)

df["meduc_label"] = decode("meduc")
```

## Gotchas

* **Missing values are empty cells** - they were "not stated" codes (99, 999, 9999, 99.9) in the raw file and have
  been converted to NULL already. Do not treat 99 as a real value anywhere in these files.
* **Y/N flags** (`rf_*`, `ip_*`, `ld_*`, `wic`, `smoker`, ...) also contain `U` = unknown and sometimes `X` = not applicable.
  Decide explicitly how to treat them (drop, separate category, or impute).
* `smoker` is the NCHS field `CIG_REC`; all other columns keep the NCHS field name in lowercase.
* `nicu`, `preterm`, `lbw` are booleans (`t`/`f` in the CSV; `preterm`/`lbw` can be empty when gestation/weight unknown).
* Class imbalance: ~10% positive. Report precision/recall/F1/PR-AUC for the NICU class, not just accuracy.
* The sample was drawn with a fixed seed, so everyone's numbers on `nicu_sample_300k.csv` should match.
