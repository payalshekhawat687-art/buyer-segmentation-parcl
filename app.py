import inspect

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import adjusted_rand_score, silhouette_score

import pipeline as pl

st.set_page_config(page_title="Parcl Buyer Intelligence", page_icon="🏢", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.4rem}
[data-testid="stMetric"]{background:#f6f8fb;border:1px solid #e3e8ef;border-radius:10px;padding:10px 14px}
[data-testid="stMetricLabel"]{color:#5b6677}
</style>""", unsafe_allow_html=True)

PALETTE = px.colors.qualitative.Bold
# Works on both older Streamlit (use_container_width) and newer (width="stretch")
FULL = ({"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters
        else {"use_container_width": True})


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Cleaning data & engineering features...")
def get_data(clients_src, props_src):
    return pl.load_clean(clients_src, props_src)


@st.cache_data(show_spinner="Clustering...")
def run_model(df, scaler, use_purchase, method, k):
    X = pl.build_matrix(df, scaler, use_purchase)
    labels = pl.fit_clusters(X, k, method)
    names, prof = pl.name_segments(df, labels)
    sil = silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=42)
    other = "Hierarchical" if method == "K-Means" else "K-Means"
    ari = adjusted_rand_score(labels, pl.fit_clusters(X, k, other))
    return X, labels, names, sil, ari, other


@st.cache_data(show_spinner="Scanning k...")
def get_k_scan(df, scaler, use_purchase):
    return pl.k_scan(pl.build_matrix(df, scaler, use_purchase))


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🏢 Parcl Buyer Intelligence")
with st.sidebar.expander("📂 Data source", expanded=False):
    up_c = st.file_uploader("clients.csv", type="csv")
    up_p = st.file_uploader("properties.csv", type="csv")
df, log = get_data(up_c or "clients.csv", up_p or "properties.csv")

st.sidebar.subheader("⚙️ Model settings")
method = st.sidebar.radio("Algorithm", ["K-Means", "Hierarchical"], horizontal=True)
k = st.sidebar.slider("Number of clusters (k)", 2, 8, 4)
scaler = st.sidebar.selectbox("Scaler", ["StandardScaler", "MinMaxScaler"])
use_purchase = st.sidebar.checkbox("Include purchase behaviour (units, spend, office share)", True,
                                   help="Built from properties.csv via client_ref. Needed to see 'multiple units' and 'large investment' buyers.")

X, labels, names, sil, ari, other = run_model(df, scaler, use_purchase, method, k)
df = df.assign(cluster=labels)
df["segment"] = df["cluster"].map(names)
seg_order = [names[i] for i in sorted(names)]
cmap = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(seg_order)}

st.sidebar.subheader("🔎 Filters")
sel_country = st.sidebar.multiselect("Country", sorted(df.country.unique()))
reg_pool = df[df.country.isin(sel_country)] if sel_country else df
sel_region = st.sidebar.multiselect("Region", sorted(reg_pool.region.unique()))
sel_purpose = st.sidebar.multiselect("Acquisition purpose", sorted(df.acquisition_purpose.unique()))
sel_type = st.sidebar.multiselect("Client type", sorted(df.client_type.unique()))

f = df.copy()
if sel_country: f = f[f.country.isin(sel_country)]
if sel_region: f = f[f.region.isin(sel_region)]
if sel_purpose: f = f[f.acquisition_purpose.isin(sel_purpose)]
if sel_type: f = f[f.client_type.isin(sel_type)]
st.sidebar.caption(f"Showing **{len(f):,}** of {len(df):,} clients. Segments are fitted on all clients; filters only change the view.")

# ------------------------------------------------------------------ header
st.title("Buyer Segmentation & Investment Profiling")
st.caption("ML-based clustering for Parcl real estate market intelligence")

if f.empty:
    st.warning("No clients match the current filters.")
    st.stop()

tabs = st.tabs(["📊 Segmentation Overview", "💼 Investor Behavior", "🌍 Geographic Analysis",
                "🧠 Segment Insights", "🧪 Model & Methodology"])

# ================================================================== 1. overview
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Clients", f"{len(f):,}")
    c2.metric("Segments", f.segment.nunique())
    c3.metric("Investment buyers", f"{(f.acquisition_purpose == 'Investment').mean():.0%}")
    c4.metric("Loan applied", f"{(f.loan_applied == 'Yes').mean():.0%}")
    c5.metric("Avg satisfaction", f"{f.satisfaction_score.mean():.2f} / 5")

    left, right = st.columns([1, 1.2])
    cnt = f.segment.value_counts().reindex(seg_order).dropna().reset_index()
    cnt.columns = ["segment", "clients"]
    with left:
        fig = px.pie(cnt, names="segment", values="clients", hole=0.55, color="segment",
                     color_discrete_map=cmap, title="Cluster distribution")
        fig.update_traces(textinfo="percent", sort=False)
        fig.update_layout(legend=dict(orientation="h", y=-0.15), margin=dict(t=50, b=10))
        st.plotly_chart(fig, **FULL)
    with right:
        xy = pl.pca_2d(X)
        pdf = pd.DataFrame(xy, columns=["PC1", "PC2"], index=df.index).loc[f.index]
        pdf["segment"] = f.segment
        fig = px.scatter(pdf, x="PC1", y="PC2", color="segment", color_discrete_map=cmap, opacity=0.65,
                         category_orders={"segment": seg_order}, title="Clients in 2-D feature space (PCA)")
        fig.update_layout(margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, **FULL)
    st.info(f"Silhouette score for this model: **{sil:.3f}**. "
            "Values near 0.1–0.2 mean segments overlap noticeably, so treat them as tendencies, not hard boundaries "
            "(see *Model & Methodology*).")

# ================================================================== 2. investor behaviour
with tabs[1]:
    st.subheader("Investment patterns by cluster")
    a, b = st.columns(2)
    pur = (f.groupby(["segment", "acquisition_purpose"]).size().reset_index(name="n"))
    pur["share"] = pur.n / pur.groupby("segment").n.transform("sum")
    fig = px.bar(pur, x="segment", y="share", color="acquisition_purpose", barmode="stack",
                 category_orders={"segment": seg_order}, title="Acquisition purpose mix",
                 color_discrete_sequence=["#4c78a8", "#f58518"])
    fig.update_yaxes(tickformat=".0%"); fig.update_xaxes(title=None)
    a.plotly_chart(fig, **FULL)

    loan = f.groupby("segment").loan_applied.apply(lambda s: (s == "Yes").mean()).reindex(seg_order).dropna().reset_index()
    loan.columns = ["segment", "loan_rate"]
    fig = px.bar(loan, x="segment", y="loan_rate", color="segment", color_discrete_map=cmap,
                 title="Loan dependence (share applying for a loan)", text=loan.loan_rate.map("{:.0%}".format))
    fig.update_yaxes(tickformat=".0%", rangemode="tozero"); fig.update_layout(showlegend=False); fig.update_xaxes(title=None)
    b.plotly_chart(fig, **FULL)

    a, b = st.columns(2)
    fig = px.box(f, x="segment", y="total_spend", color="segment", color_discrete_map=cmap,
                 category_orders={"segment": seg_order}, title="Total spend per client (USD)", points=False)
    fig.update_layout(showlegend=False); fig.update_xaxes(title=None)
    a.plotly_chart(fig, **FULL)
    fig = px.box(f, x="segment", y="units_purchased", color="segment", color_discrete_map=cmap,
                 category_orders={"segment": seg_order}, title="Units purchased per client", points=False)
    fig.update_layout(showlegend=False); fig.update_xaxes(title=None)
    b.plotly_chart(fig, **FULL)

    a, b = st.columns(2)
    ref = f.groupby(["segment", "referral_channel"]).size().reset_index(name="n")
    ref["share"] = ref.n / ref.groupby("segment").n.transform("sum")
    fig = px.bar(ref, x="segment", y="share", color="referral_channel", barmode="group",
                 category_orders={"segment": seg_order}, title="Referral channel mix")
    fig.update_yaxes(tickformat=".0%"); fig.update_xaxes(title=None)
    a.plotly_chart(fig, **FULL)
    ag = f.groupby(["segment", "age_group"], observed=True).size().reset_index(name="n")
    ag["share"] = ag.n / ag.groupby("segment").n.transform("sum")
    fig = px.bar(ag, x="segment", y="share", color="age_group", barmode="stack",
                 category_orders={"segment": seg_order}, title="Age profile")
    fig.update_yaxes(tickformat=".0%"); fig.update_xaxes(title=None)
    b.plotly_chart(fig, **FULL)

# ================================================================== 3. geography
with tabs[2]:
    st.subheader("Where each segment buys from")
    ctry = f.groupby("country").agg(clients=("client_id", "count"), avg_spend=("total_spend", "mean"),
                                    invest=("acquisition_purpose", lambda s: (s == "Investment").mean())).reset_index()
    ctry["dominant_segment"] = ctry.country.map(f.groupby("country").segment.agg(lambda s: s.value_counts().index[0]))
    metric = st.radio("Colour map by", ["clients", "avg_spend", "invest"], horizontal=True,
                      format_func={"clients": "Client count", "avg_spend": "Avg spend", "invest": "Investment share"}.get)
    iso = {"USA": "USA", "UK": "GBR", "Canada": "CAN", "Germany": "DEU", "France": "FRA", "Belgium": "BEL",
           "Mexico": "MEX", "Australia": "AUS", "Russia": "RUS", "Denmark": "DNK"}
    ctry["iso"] = ctry.country.map(iso)
    fig = px.choropleth(ctry, locations="iso", color=metric, hover_name="country",
                        hover_data={"iso": False, "clients": True, "dominant_segment": True},
                        color_continuous_scale="Blues")
    fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth")
    fig.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=420)
    st.plotly_chart(fig, **FULL)

    a, b = st.columns(2)
    cs = f.groupby(["country", "segment"]).size().reset_index(name="n")
    cs["share"] = cs.n / cs.groupby("country").n.transform("sum")
    fig = px.bar(cs, x="country", y="share", color="segment", color_discrete_map=cmap,
                 category_orders={"segment": seg_order}, title="Segment mix by country")
    fig.update_yaxes(tickformat=".0%"); fig.update_xaxes(title=None)
    a.plotly_chart(fig, **FULL)
    top = f.region.value_counts().head(12).index
    rs = f[f.region.isin(top)].groupby(["region", "segment"]).size().reset_index(name="n")
    fig = px.bar(rs, y="region", x="n", color="segment", color_discrete_map=cmap, orientation="h",
                 category_orders={"segment": seg_order}, title="Top 12 regions by clients")
    fig.update_yaxes(categoryorder="total ascending", title=None)
    b.plotly_chart(fig, **FULL)

    st.markdown("**Region × segment heatmap (client counts)**")
    heat = f.pivot_table(index="region", columns="segment", values="client_id", aggfunc="count", fill_value=0)
    heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).head(20).index]
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="Blues", text_auto=True)
    fig.update_layout(height=520, margin=dict(t=10))
    st.plotly_chart(fig, **FULL)

# ================================================================== 4. insights
with tabs[3]:
    st.subheader("Segment insights")
    g = f.groupby("segment")
    summ = pd.DataFrame({
        "Clients": g.size(),
        "Share of base": g.size() / len(f),
        "Avg age": g.age.mean(),
        "Avg satisfaction": g.satisfaction_score.mean(),
        "Investment %": g.acquisition_purpose.apply(lambda s: (s == "Investment").mean()),
        "Loan %": g.loan_applied.apply(lambda s: (s == "Yes").mean()),
        "Corporate %": g.client_type.apply(lambda s: (s == "Corporate").mean()),
        "International %": g.country.apply(lambda s: (s != "USA").mean()),
        "Avg units": g.units_purchased.mean(),
        "Avg spend (USD)": g.total_spend.mean(),
        "Total revenue (USD)": g.total_spend.sum(),
        "Top country": g.country.agg(lambda s: s.value_counts().index[0]),
        "Top channel": g.referral_channel.agg(lambda s: s.value_counts().index[0]),
    }).reindex([s for s in seg_order if s in g.groups])
    st.dataframe(summ.style.format({
        "Share of base": "{:.1%}", "Avg age": "{:.1f}", "Avg satisfaction": "{:.2f}", "Investment %": "{:.0%}",
        "Loan %": "{:.0%}", "Corporate %": "{:.0%}", "International %": "{:.0%}", "Avg units": "{:.1f}",
        "Avg spend (USD)": "${:,.0f}", "Total revenue (USD)": "${:,.0f}"}), **FULL)

    st.markdown("#### Segment profiles")
    base = df
    cols = st.columns(min(3, len(summ)))
    for i, (seg, r) in enumerate(summ.iterrows()):
        pts = []
        d = lambda a, b: (a - b) / b if b else 0
        if abs(d(r["Avg spend (USD)"], base.total_spend.mean())) > .15:
            pts.append(f"Spends {d(r['Avg spend (USD)'], base.total_spend.mean()):+.0%} vs average")
        if abs(r["Avg satisfaction"] - base.satisfaction_score.mean()) > .3:
            pts.append(f"Satisfaction {r['Avg satisfaction']:.1f} vs {base.satisfaction_score.mean():.1f} overall")
        if abs(r["Avg age"] - base.age.mean()) > 3:
            pts.append(f"Avg age {r['Avg age']:.0f} vs {base.age.mean():.0f} overall")
        if abs(r["Loan %"] - (base.loan_applied == 'Yes').mean()) > .08:
            pts.append(f"Loan use {r['Loan %']:.0%} vs {(base.loan_applied == 'Yes').mean():.0%} overall")
        if abs(r["Investment %"] - (base.acquisition_purpose == 'Investment').mean()) > .07:
            pts.append(f"Investment share {r['Investment %']:.0%} vs {(base.acquisition_purpose == 'Investment').mean():.0%}")
        if r["Avg units"] > base.units_purchased.mean() * 1.3:
            pts.append(f"Buys {r['Avg units']:.1f} units on average")
        with cols[i % len(cols)]:
            st.markdown(f"**{seg}** — {int(r['Clients']):,} clients")
            for p_ in (pts or ["Close to the overall average on every measure"]):
                st.markdown(f"- {p_}")

    st.markdown("#### Segment explorer")
    pick = st.selectbox("Segment", list(summ.index))
    sub = f[f.segment == pick]
    show = ["client_id", "client_type", "gender", "age", "country", "region", "acquisition_purpose", "loan_applied",
            "referral_channel", "satisfaction_score", "units_purchased", "total_spend"]
    st.dataframe(sub[show].round(1), **FULL, height=280)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download this segment", sub[show].to_csv(index=False).encode(), f"{pick[:2]}_segment.csv", "text/csv")
    d2.download_button("⬇️ Download all clients with segment", f[show + ["segment"]].to_csv(index=False).encode(),
                       "clients_segmented.csv", "text/csv")

# ================================================================== 5. methodology
with tabs[4]:
    st.subheader("Optimal k and model validation")
    scan = get_k_scan(df, scaler, use_purchase)
    a, b = st.columns(2)
    fig = px.line(scan, x="k", y="inertia", markers=True, title="Elbow method (K-Means inertia)")
    fig.add_vline(x=k, line_dash="dash", line_color="grey", annotation_text=f"k={k}")
    a.plotly_chart(fig, **FULL)
    fig = px.line(scan, x="k", y="silhouette", markers=True, title="Silhouette score by k")
    fig.add_vline(x=k, line_dash="dash", line_color="grey", annotation_text=f"k={k}")
    b.plotly_chart(fig, **FULL)
    best = scan.loc[scan.silhouette.idxmax()]
    st.write(f"Best silhouette: **k = {int(best.k)}** ({best.silhouette:.3f}). Current model ({method}, k = {k}): "
             f"silhouette **{sil:.3f}**. Agreement with {other} on the same k (Adjusted Rand Index): **{ari:.2f}**.")
    st.caption("The silhouette is low at every k, so the data has no sharply separated groups. "
               "The default k = 4 follows the four buyer segments in the project brief.")

    st.markdown("#### Hierarchical clustering dendrogram (random sample of 250 clients)")
    samp = X.sample(250, random_state=1)
    fig = ff.create_dendrogram(samp.values, linkagefun=lambda x: linkage(x, "ward"))
    fig.update_layout(height=380, margin=dict(t=10), xaxis=dict(showticklabels=False))
    st.plotly_chart(fig, **FULL)

    st.markdown("#### Pipeline")
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows loaded", f"{log['raw_rows']:,}")
    m2.metric("Duplicates removed", log["duplicates_removed"])
    m3.metric("Missing values filled", log["missing_filled"])
    st.markdown(f"""
1. **Cleaning** – trimmed text, merged duplicate clients, standardised labels (`Company` → `Corporate`, `Home` → `Personal Use`),
   parsed mixed date formats into age, filled any gaps (mode / median).
2. **Encoding** – one-hot for client type, purpose, loan, referral channel and country; label-encoded copies of
   client type, region, purpose, channel and country are kept in the data. Region (55 values) is not used for clustering
   because label codes would imply a fake order; it is analysed on the Geographic tab.
3. **Scaling** – `{scaler}` on age, satisfaction{', units, spend, average price, office share' if use_purchase else ''}.
4. **Models** – K-Means and Ward hierarchical clustering; choose k with elbow + silhouette.
5. **Interpretation** – a cluster gets a brief archetype (Global Investors, First-Time Buyers, Corporate Buyers,
   Luxury Investors) only when its profile clearly stands out from the full client base; otherwise it gets a descriptive name.
""")
    with st.expander("Raw cluster centroids (feature means)"):
        cen = X.assign(cluster=df.segment.values).groupby("cluster").mean().round(2)
        st.dataframe(cen.T, **FULL)
