import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

CAT_ONEHOT = ["client_type", "acquisition_purpose", "referral_channel", "loan_applied", "country"]
NUM_BASE = ["age", "satisfaction_score"]
NUM_PURCHASE = ["units_purchased", "total_spend", "avg_price", "office_share"]


def _parse_dob(s: pd.Series) -> pd.Series:
    """Mixed formats (M/D/YYYY and MM-DD-YYYY). Only the year matters for age."""
    out = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    alt = pd.to_datetime(s, format="%m-%d-%Y", errors="coerce")
    out = out.fillna(alt)
    alt2 = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
    return out.fillna(alt2)


def load_clean(clients_src, props_src):
    log = {}
    c = pd.read_csv(clients_src)
    p = pd.read_csv(props_src)
    log["raw_rows"] = len(c)

    # --- Step 1: cleaning -------------------------------------------------
    c.columns = c.columns.str.strip().str.lower()
    for col in c.select_dtypes("object"):
        c[col] = c[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    n0 = len(c)
    c = c.drop_duplicates(subset="client_id").drop_duplicates(
        subset=["first_name", "last_name", "date_of_birth", "country"])
    log["duplicates_removed"] = n0 - len(c)

    c["client_type"] = c["client_type"].str.title().replace({"Company": "Corporate", "Business": "Corporate"})
    c["acquisition_purpose"] = c["acquisition_purpose"].str.title().replace({"Home": "Personal Use"})
    c["loan_applied"] = c["loan_applied"].str.title()
    c["referral_channel"] = c["referral_channel"].str.title()
    c["gender"] = c["gender"].str.upper().replace({"MALE": "M", "FEMALE": "F"})
    c["country"] = c["country"].str.strip()
    c["region"] = c["region"].str.strip()

    c["dob"] = _parse_dob(c["date_of_birth"])
    ref = pd.Timestamp("2025-12-31")
    c["age"] = ((ref - c["dob"]).dt.days / 365.25).round(1)
    c.loc[(c["age"] < 18) | (c["age"] > 100), "age"] = np.nan
    c["satisfaction_score"] = pd.to_numeric(c["satisfaction_score"], errors="coerce").clip(1, 5)

    miss = c[["client_type", "gender", "country", "region", "acquisition_purpose", "loan_applied",
              "referral_channel", "age", "satisfaction_score"]].isna().sum()
    log["missing_filled"] = int(miss.sum())
    for col in ["client_type", "gender", "country", "region", "acquisition_purpose", "loan_applied", "referral_channel"]:
        c[col] = c[col].fillna(c[col].mode().iloc[0])
    c["age"] = c["age"].fillna(c["age"].median())
    c["satisfaction_score"] = c["satisfaction_score"].fillna(c["satisfaction_score"].median())

    # --- purchase behaviour from properties.csv ---------------------------
    p["price"] = pd.to_numeric(p["sale_price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    sold = p[p["listing_status"].str.title() == "Sold"].dropna(subset=["client_ref", "price"])
    g = sold.groupby("client_ref").agg(
        units_purchased=("listing_id", "count"), total_spend=("price", "sum"),
        avg_price=("price", "mean"), avg_area=("floor_area_sqft", "mean"),
        office_share=("unit_category", lambda s: (s == "Office").mean()))
    c = c.merge(g, left_on="client_id", right_index=True, how="left")
    for col in NUM_PURCHASE + ["avg_area"]:
        c[col] = c[col].fillna(0)
    c["age_group"] = pd.cut(c["age"], [0, 30, 40, 50, 60, 120], labels=["<30", "30-39", "40-49", "50-59", "60+"])
    c["is_international"] = np.where(c["country"] != "USA", "International", "Domestic (USA)")

    # --- Step 2: encoding (label-encoded versions kept for reference) -----
    for col in ["client_type", "region", "acquisition_purpose", "referral_channel", "country"]:
        c[col + "_le"] = LabelEncoder().fit_transform(c[col])
    return c.reset_index(drop=True), log


def build_matrix(df, scaler="StandardScaler", use_purchase=True):
    num = NUM_BASE + (NUM_PURCHASE if use_purchase else [])
    sc = StandardScaler() if scaler == "StandardScaler" else MinMaxScaler()   # Step 3
    Xn = pd.DataFrame(sc.fit_transform(df[num]), columns=num, index=df.index)
    Xc = pd.get_dummies(df[CAT_ONEHOT], dtype=float)                           # Step 2 one-hot
    return pd.concat([Xn, Xc], axis=1)


def k_scan(X, k_min=2, k_max=10, seed=42):
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_, sample_size=min(2000, len(X)), random_state=seed)})
    return pd.DataFrame(rows)


def fit_clusters(X, k, method="K-Means", seed=42):
    if method == "K-Means":
        return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


def name_segments(df, labels):
    """Name clusters. A PRD archetype is only used when the cluster's profile really
    matches it (relative to the whole base); otherwise a descriptive label is used."""
    d = df.assign(cluster=labels)
    prof = d.groupby("cluster").agg(
        corp=("client_type", lambda s: (s == "Corporate").mean()),
        inv=("acquisition_purpose", lambda s: (s == "Investment").mean()),
        loan=("loan_applied", lambda s: (s == "Yes").mean()),
        intl=("country", lambda s: (s != "USA").mean()),
        age=("age", "mean"), sat=("satisfaction_score", "mean"),
        spend=("total_spend", "mean"), units=("units_purchased", "mean"), n=("client_id", "count"))
    base = dict(corp=(df.client_type == "Corporate").mean(), inv=(df.acquisition_purpose == "Investment").mean(),
                loan=(df.loan_applied == "Yes").mean(), intl=(df.country != "USA").mean(),
                age=df.age.mean(), sat=df.satisfaction_score.mean(), spend=df.total_spend.mean())
    cand = {}
    for i, r in prof.iterrows():
        if r.corp >= 1.5 * base["corp"]:
            cand[(i, "Corporate Buyers")] = r.corp / base["corp"]
        if r.intl >= 1.2 * base["intl"] and r.inv >= base["inv"]:
            cand[(i, "Global Investors")] = r.intl / base["intl"] + r.inv / base["inv"]
        if r.loan >= 1.1 * base["loan"] and r.age <= base["age"] - 2:
            cand[(i, "First-Time Buyers")] = r.loan / base["loan"] + (base["age"] - r.age) / 10
        if r.spend >= 1.25 * base["spend"] and r.sat >= base["sat"]:
            cand[(i, "Luxury Investors")] = r.spend / base["spend"] + r.sat / base["sat"]
    names, used = {}, set()
    for (i, s), _ in sorted(cand.items(), key=lambda kv: -kv[1]):
        if i not in names and s not in used:
            names[i] = s; used.add(s)
    z = (prof[["age", "sat", "spend", "loan", "inv", "intl", "corp"]] - prof[["age", "sat", "spend", "loan", "inv", "intl", "corp"]].mean()) / \
        prof[["age", "sat", "spend", "loan", "inv", "intl", "corp"]].std(ddof=0).replace(0, 1)
    words = {"age": ("Older Buyers", "Younger Buyers"), "sat": ("High-Satisfaction Buyers", "Low-Satisfaction Buyers"),
             "spend": ("High-Spend Buyers", "Lower-Spend Buyers"), "loan": ("Loan-Financed Buyers", "Cash Buyers"),
             "inv": ("Investment-Led Buyers", "Home-Led Buyers"), "intl": ("International Buyers", "Domestic Buyers"),
             "corp": ("Company-Heavy Buyers", "Individual-Only Buyers")}
    seen = {}
    for i in prof.index:
        if i in names:
            continue
        k = z.loc[i].abs().idxmax()
        lab = words[k][0] if z.loc[i, k] > 0 else words[k][1]
        seen[lab] = seen.get(lab, 0) + 1
        names[i] = lab if seen[lab] == 1 else f"{lab} ({seen[lab]})"
    return {i: f"C{i+1} · {n}" for i, n in names.items()}, prof


def pca_2d(X, seed=42):
    return PCA(n_components=2, random_state=seed).fit_transform(X)
