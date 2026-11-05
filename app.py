import io

import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# IN SILICO PCR ANALYZER (v2)
# Girdi: Forward ve Reverse primerin BLAST hit tabloları (outfmt 6 / NCBI "Hit table")
# Yenilikler:
#  - Primer uzunluğuna göre hizalanmayan 5'/3' uçlar hesaba katılır
#  - 3' uç ve toplam uyumsuzluk filtresi (Taq 3' uyumsuzlukta uzatamaz)
#  - Amplikon sınırları primerin tam ayak izine göre hesaplanır
#  - Hedef bölge (ROI) bayrağı ve tek-primer ROI hitleri (tanı için)
# =========================================================

BLAST_COLS = [
    "query", "subject", "identity", "align_length", "mismatches",
    "gap_opens", "q_start", "q_end", "s_start", "s_end", "evalue", "bit_score",
]


def read_blast(raw_text: str) -> pd.DataFrame:
    """Sekme veya virgülle ayrılmış BLAST hit tablosunu okur, # satırlarını atar."""
    lines = [l for l in raw_text.splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        return pd.DataFrame(columns=BLAST_COLS)
    delim = "\t" if "\t" in lines[0] else ","
    df = pd.read_csv(
        io.StringIO("\n".join(lines)), sep=delim, header=None,
        names=BLAST_COLS, usecols=range(12),
    )
    return df


def prepare_hits(df: pd.DataFrame, plen: int) -> pd.DataFrame:
    """Strand, koordinat ve primerin tam ayak izini (footprint) hesaplar."""
    d = df.copy()
    d["strand"] = np.where(d["s_start"] <= d["s_end"], "+", "-")
    d["low"] = d[["s_start", "s_end"]].min(axis=1)
    d["high"] = d[["s_start", "s_end"]].max(axis=1)
    d["unal_5p"] = d["q_start"] - 1          # hizalanmayan 5' bazlar
    d["unal_3p"] = plen - d["q_end"]         # hizalanmayan 3' bazlar
    d["total_mm"] = d["mismatches"] + d["unal_5p"] + d["unal_3p"]
    plus = d["strand"] == "+"
    # + strand: q_start düşük koordinatta, 3' uç yüksek koordinatta
    # - strand: q_end düşük koordinatta (3' uç aşağı), q_start yüksek koordinatta
    d["fp_low"] = np.where(plus, d["low"] - d["unal_5p"], d["low"] - d["unal_3p"])
    d["fp_high"] = np.where(plus, d["high"] + d["unal_3p"], d["high"] + d["unal_5p"])
    return d


def filter_hits(d: pd.DataFrame, max_mm: int, max_3p: int) -> pd.DataFrame:
    keep = (d["gap_opens"] == 0) & (d["total_mm"] <= max_mm) & (d["unal_3p"] <= max_3p)
    return d[keep].copy()


def pair_hits(f: pd.DataFrame, r: pd.DataFrame, min_size: int, max_size: int) -> pd.DataFrame:
    """Aynı contigte, karşılıklı bakan F ve R hitlerinden amplikon üretir."""
    m = f.merge(r, on="subject", suffixes=("_F", "_R"))
    if m.empty:
        return m
    fwd = ((m["strand_F"] == "+") & (m["strand_R"] == "-")
           & (m["fp_low_F"] < m["fp_low_R"]) & (m["fp_high_F"] < m["fp_high_R"]))
    rev = ((m["strand_F"] == "-") & (m["strand_R"] == "+")
           & (m["fp_low_R"] < m["fp_low_F"]) & (m["fp_high_R"] < m["fp_high_F"]))
    m = m[fwd | rev].copy()
    if m.empty:
        return m
    is_fwd = m["strand_F"] == "+"
    m["Amp_Start"] = np.where(is_fwd, m["fp_low_F"], m["fp_low_R"])
    m["Amp_End"] = np.where(is_fwd, m["fp_high_R"], m["fp_high_F"])
    m["Size_bp"] = m["Amp_End"] - m["Amp_Start"] + 1
    m = m[(m["Size_bp"] >= min_size) & (m["Size_bp"] <= max_size)].copy()
    m["Orientation"] = np.where(m["strand_F"] == "+", "F+ / R-", "F- / R+")
    m["Worst_mm"] = m[["total_mm_F", "total_mm_R"]].max(axis=1)
    m["Worst_3p"] = m[["unal_3p_F", "unal_3p_R"]].max(axis=1)
    m["Class"] = np.select(
        [m["Worst_mm"] == 0, (m["Worst_3p"] == 0) & (m["Worst_mm"] <= 2)],
        ["Mükemmel", "Olası"], default="Zayıf",
    )
    return m.sort_values(["Size_bp", "Worst_mm"])


def flag_roi(df: pd.DataFrame, subject_text: str, roi_start: int, roi_end: int,
             start_col: str, end_col: str, subject_col: str = "subject") -> pd.Series:
    if not subject_text or roi_end <= roi_start:
        return pd.Series(False, index=df.index)
    return (df[subject_col].astype(str).str.contains(subject_text, regex=False)
            & (df[end_col] >= roi_start) & (df[start_col] <= roi_end))


# ----------------------------- UI -----------------------------
def main():
    st.set_page_config(page_title="In Silico PCR Analyzer", layout="wide")
    st.title("🧬 In Silico PCR Analyzer v2")
    st.markdown(
        "Forward ve Reverse primerin BLAST hit tablolarını birleştirir, yönelimi ve 3' ucu denetler, "
        "amplikon boyutunu **primerin tam ayak izine** göre hesaplar."
    )

    with st.expander("BLAST ayarları (önemli)"):
        st.markdown(
            "- Program: **blastn**, *Search for short nearly exact matches* (blastn-short)\n"
            "- Expect threshold: **1000** veya daha yüksek, word size: **7**\n"
            "- **Low complexity filter KAPALI**, mask kapalı\n"
            "- Max target sequences: mümkün olan en yüksek değer\n"
            "- Reverse primeri **olduğu gibi (5'→3')** girin, ters-tamamlayıcısını almayın"
        )

    st.sidebar.header("Primerler")
    f_seq = st.sidebar.text_input("Forward primer dizisi (opsiyonel)", "")
    r_seq = st.sidebar.text_input("Reverse primer dizisi (opsiyonel)", "")
    f_len_in = st.sidebar.number_input("Forward uzunluğu (dizi girilmediyse)", value=20, min_value=10)
    r_len_in = st.sidebar.number_input("Reverse uzunluğu (dizi girilmediyse)", value=20, min_value=10)
    clean = lambda s: "".join(c for c in s.upper() if c in "ACGTRYSWKMBDHVN")
    f_len = len(clean(f_seq)) if clean(f_seq) else int(f_len_in)
    r_len = len(clean(r_seq)) if clean(r_seq) else int(r_len_in)

    st.sidebar.header("Filtreler")
    max_mm = st.sidebar.number_input("Primer başına maks. uyumsuzluk (hizalanmayan uçlar dahil)", value=3, min_value=0)
    max_3p = st.sidebar.number_input("Maks. hizalanmayan 3' baz", value=0, min_value=0,
                                     help="0 = 3' uç tam oturmalı. 1-2 yaparsanız 'şüpheli' ürünleri de görürsünüz.")
    min_size = st.sidebar.number_input("Minimum amplikon (bp)", value=30, step=10)
    max_size = st.sidebar.number_input("Maksimum amplikon (bp)", value=3000, step=100)

    st.sidebar.header("Hedef bölge (ROI)")
    roi_subject = st.sidebar.text_input("Contig/kromozom adı (içerir)", "")
    roi_start = st.sidebar.number_input("ROI başlangıç (bp)", value=0, step=1000)
    roi_end = st.sidebar.number_input("ROI bitiş (bp)", value=0, step=1000)

    c1, c2, c3 = st.columns(3)
    f_file = c1.file_uploader("Forward BLAST hit tablosu", type=["txt", "tsv", "csv"])
    r_file = c2.file_uploader("Reverse BLAST hit tablosu", type=["txt", "tsv", "csv"])
    map_file = c3.file_uploader("Opsiyonel: accession → kromozom adı (2 sütun, TSV)", type=["txt", "tsv"])

    if not (f_file and r_file):
        st.info("İki hit tablosunu yükleyin.")
        return

    f_raw = read_blast(f_file.getvalue().decode("utf-8", errors="ignore"))
    r_raw = read_blast(r_file.getvalue().decode("utf-8", errors="ignore"))
    f_all = prepare_hits(f_raw, f_len)
    r_all = prepare_hits(r_raw, r_len)
    f_hits = filter_hits(f_all, int(max_mm), int(max_3p))
    r_hits = filter_hits(r_all, int(max_mm), int(max_3p))

    name_map = {}
    if map_file:
        for line in map_file.getvalue().decode("utf-8", errors="ignore").splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                name_map[parts[0].strip()] = parts[1].strip()

    st.write(f"**Forward:** {len(f_raw)} ham hit → {len(f_hits)} hit (filtre sonrası) | "
             f"**Reverse:** {len(r_raw)} ham hit → {len(r_hits)} hit")

    pairs = pair_hits(f_hits, r_hits, int(min_size), int(max_size))

    if pairs.empty:
        st.warning("Filtrelere uyan amplikon bulunamadı. 3' ve uyumsuzluk filtresini gevşetmeyi deneyin.")
    else:
        pairs["Chromosome"] = pairs["subject"].map(name_map).fillna(pairs["subject"])
        pairs["In_ROI"] = flag_roi(pairs, roi_subject, int(roi_start), int(roi_end), "Amp_Start", "Amp_End")
        show = pairs[[
            "Chromosome", "subject", "Amp_Start", "Amp_End", "Size_bp", "Orientation", "Class",
            "In_ROI", "total_mm_F", "unal_3p_F", "total_mm_R", "unal_3p_R", "identity_F", "identity_R",
        ]].rename(columns={"subject": "Contig", "total_mm_F": "F_mm", "unal_3p_F": "F_3p_unal",
                           "total_mm_R": "R_mm", "unal_3p_R": "R_3p_unal",
                           "identity_F": "F_ident", "identity_R": "R_ident"})
        st.success(f"{len(show)} olası amplikon bulundu "
                   f"({(show['Class'] == 'Mükemmel').sum()} mükemmel, {(show['Class'] == 'Olası').sum()} olası).")
        st.dataframe(show, use_container_width=True)
        st.bar_chart(show.groupby("Chromosome").size())
        st.download_button("📥 CSV indir", show.to_csv(index=False).encode("utf-8"),
                           "epcr_results.csv", "text/csv")

    # Tanı: ROI içinde tek primer hitleri (çift oluşmasa bile)
    if roi_subject and roi_end > roi_start:
        st.subheader("ROI içindeki tek-primer hitleri (filtre ÖNCESİ)")
        for label, d in (("Forward", f_all), ("Reverse", r_all)):
            inroi = d[flag_roi(d, roi_subject, int(roi_start), int(roi_end), "low", "high")]
            st.markdown(f"**{label}:** {len(inroi)} hit")
            if len(inroi):
                st.dataframe(inroi[["subject", "strand", "low", "high", "identity", "mismatches",
                                    "unal_5p", "unal_3p", "total_mm", "gap_opens"]],
                             use_container_width=True)

    with st.expander("Filtre sonrası tüm hitler"):
        st.write("Forward"); st.dataframe(f_hits, use_container_width=True)
        st.write("Reverse"); st.dataframe(r_hits, use_container_width=True)


if __name__ == "__main__":
    main()
