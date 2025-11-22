import os
import csv
import json
from io import BytesIO

import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import mapclassify
from matplotlib.patches import Rectangle
from matplotlib.colors import to_hex
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib import cm
import contextily as cx
from PIL import Image
import numpy as np
import re

# Base dirs
base_dir = os.path.dirname(__file__)
languages_dir = os.path.join(base_dir, "languages")
layer_dir = os.path.join(base_dir, "layers")

arial_font = FontProperties(family="DejaVu Sans", size=7)


def load_language(lang_code: str) -> dict:
    path = os.path.join(languages_dir, f"{lang_code}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_eurostat(df):
    df2 = df.copy()
    cols = df2.columns.tolist()

    for col in cols:
        colname = str(col).lower()   # <-- FIX QUI

        # non toccare la colonna GEO
        if colname == "geo":
            continue

        # salta colonne descrittive
        if colname in ["freq", "unit", "obs_flag", "conf_status"]:
            continue

        # se già numerica non toccare
        if df2[col].dtype != object:
            continue

        # pulizia valori
        def _clean(v):
            if pd.isna(v):
                return np.nan
            v = str(v).strip()
            if v.startswith(":"):
                return np.nan
            m = re.match(r"^([0-9\.,\-]+)", v)
            if m:
                return m.group(1)
            return np.nan

        df2[col] = df2[col].apply(_clean)
        df2[col] = df2[col].astype(str).str.replace(",", ".", regex=False)
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

    return df2



def clean_generic(df):
    df = df.copy()

    # rimuove valori tipo ":", "-", "", " "
    df = df.replace({":": np.nan, "-": np.nan, "": np.nan, " ": np.nan})

    for col in df.columns:
        # prova a convertire colonne che sembrano numeri
        if df[col].dtype == object:
            # normalizza virgole
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)

            # remove trailing flags (lettere dopo numeri)
            df[col] = df[col].str.extract(r"([0-9.\-]+)", expand=False)

            # tenta conversione a float
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df
    
def clean_auto(df):
    eurostat_like = False

    # Riconoscimento colonne Eurostat classiche
    for col in df.columns:
        if "TIME" in col.upper() or "GEO" in col.upper() or ":" in str(df[col].iloc[0]):
            eurostat_like = True
            break

    # Riconoscimento valori Eurostat (":" e flag)
    for col in df.columns:
        if df[col].astype(str).str.startswith(":").any():
            eurostat_like = True
            break
        if df[col].astype(str).str.contains(r"[0-9][ ]?[a-z]$").any():
            eurostat_like = True
            break

    if eurostat_like:
        return clean_eurostat(df)
    else:
        return clean_generic(df)
    
def read_csv_with_sniffing(uploaded_file):
    try:
        sample = uploaded_file.read(2048).decode("utf-8")
        uploaded_file.seek(0)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=";,|\t")
        df = pd.read_csv(uploaded_file, sep=dialect.delimiter)
        return df
    except Exception as e:
        st.error(T["error_read_csv"].format(error=e))
        return None


def format_value(val):
    if abs(val) >= 100:
        return f"{int(round(val)):,}".replace(",", ".")
    else:
        return (
            f"{val:,.2f}"
            .replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )

# ======================================================================
# Sidebar language selector
selected_lang = st.sidebar.selectbox("Language / Lingua", [ "en", "it"])
T = load_language(selected_lang)

# Color palettes from language file
colorbrewer_palettes = T["colorbrewer_labels"]

st.title(T["title"])
st.markdown(T["intro_html"], unsafe_allow_html=True)

file = st.file_uploader(T["file_uploader_label"], type=["csv", "xlsx"])

if file:
    if file.name.endswith(".csv"):
        df = read_csv_with_sniffing(file)
        df = clean_auto(df)
    elif file.name.endswith(".xlsx"):
        xls = pd.ExcelFile(file)
        valid_sheets = [s for s in xls.sheet_names if not xls.parse(s).empty]

        if len(valid_sheets) == 0:
            st.error(T["excel_no_data"])
            st.stop()
        elif len(valid_sheets) == 1:
            df = xls.parse(valid_sheets[0])
            df = clean_auto(df)
        else:
            selected_sheet = st.selectbox(T["sheet_select"], valid_sheets)
            df = xls.parse(selected_sheet)
            df = clean_auto(df)
    else:
        st.error(T["unsupported_format"])
        st.stop()

    if df is not None:
        st.write(T["data_preview"], df.head())

    cod_col = st.selectbox(T["geo_code_column"], df.columns)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if cod_col in numeric_cols:
        numeric_cols.remove(cod_col)
    default_val = numeric_cols[0] if numeric_cols else None

    val_col = st.selectbox(
        T["value_column"],
        numeric_cols,
        index=0 if default_val else None,
    )

    nuts_files = [
        f
        for f in os.listdir(layer_dir)
        if f.startswith("nuts3_") and f.endswith(".topojson")
    ]

    if not nuts_files:
        st.error("No NUTS TopoJSON files found in /layers.")
        st.stop()

    available_years = sorted(
        [
            int(f.split("_")[1].replace(".topojson", ""))
            for f in nuts_files
        ]
    )

    year = st.selectbox(T["year_label"], available_years)
    nuts_path = os.path.join(layer_dir, f"nuts3_{year}.topojson")

    inset_grid_path = os.path.join(layer_dir, "inset_grid.topojson")

    nuts_code_col = "NUTS_ID"

    if not os.path.exists(nuts_path):
        st.error(f"NUTS file not found: {nuts_path}")
        st.stop()

    gdf = gpd.read_file(nuts_path)

    inset_gdf = None
    if os.path.exists(inset_grid_path):
        inset_gdf = gpd.read_file(inset_grid_path)

    gdf[nuts_code_col] = gdf[nuts_code_col].astype(str)
    df[cod_col] = df[cod_col].astype(str)

    gdf = gdf.merge(
        df[[cod_col, val_col]],
        left_on=nuts_code_col,
        right_on=cod_col,
        how="left",
    )

    gdf_valid = gdf[gdf[val_col].notna()].copy()
    gdf_nodata = gdf[gdf[val_col].isna()].copy()

    st.subheader(T["style_header"])

    palette_key = st.selectbox(
        T["color_scale_label"],
        options=list(colorbrewer_palettes.keys()),
        format_func=lambda x: f"{colorbrewer_palettes[x]} ({x})",
        index=list(colorbrewer_palettes.keys()).index("YlOrRd")
        if "YlOrRd" in colorbrewer_palettes
        else 0,
    )

    # Dizionario metodi: {"quantiles": "Quantili", ...}
    methods_dict = T["classification_methods"]
    
    # Selectbox: mostrano le etichette tradotte, ritornano solo la chiave fissa
    method_key = st.selectbox(
        T["classification_method_label"],
        options=list(methods_dict.keys()),
        format_func=lambda k: methods_dict[k]
    )
    
    # slider disabilitato se il metodo è "manual"
    k = st.slider(
        T["num_classes_label"],
        3,
        9,
        5,
        disabled=(method_key == "manual")
    )
    
    if gdf_valid.empty:
        st.error("No valid data available.")
        st.stop()
    
    # --- CLASSIFICAZIONE ---
    if method_key == "manual":
        raw_limits = st.text_input(T["manual_limits_label"])
        try:
            bounds = sorted(
                float(x.strip())
                for x in raw_limits.split(",")
                if x.strip() != ""
            )
        except Exception as e:
            st.error(T["error_define_classes"].format(error=e))
            st.stop()
    
        if len(bounds) < 2:
            st.error(T["error_min_limits"])
            st.stop()
    
        if len(bounds) != len(set(bounds)):
            st.error(T["error_duplicates"])
            st.stop()
    
        gdf_valid["classe"] = pd.cut(
            gdf_valid[val_col],
            bins=bounds,
            labels=False,
            include_lowest=True
        )
        k = len(bounds) - 1
    
    else:
        # metodo automatico
        if method_key == "quantiles":
            classifier = mapclassify.Quantiles(gdf_valid[val_col], k=k)
    
        elif method_key == "equal":
            classifier = mapclassify.EqualInterval(gdf_valid[val_col], k=k)
    
        elif method_key == "std":
            classifier = mapclassify.StdMean(gdf_valid[val_col])
            k = len(classifier.bins)
    
        gdf_valid["classe"] = classifier.yb
        bounds = [round(gdf_valid[val_col].min(), 2)] + [
            round(b, 2) for b in classifier.bins
        ]
    
    # applica classi a tutto il gdf
    gdf["classe"] = -1
    gdf.loc[gdf_valid.index, "classe"] = gdf_valid["classe"]
    color_nodata = "#D9D9D9"


    col_legend, col_basemap = st.columns(2)
    show_legend = col_legend.checkbox(T["legend_checkbox_label"], value=True)
    show_basemap = col_basemap.checkbox(T["basemap_checkbox_label"], value=True)

    fig, ax = plt.subplots(figsize=(8.27, 11.69))

    # Titolo della mappa
    # Titolo della mappa dentro il riquadro
    titolo = st.text_input(T["map_title_label"], value="")
    
    if titolo.strip():
        prop = fm.FontProperties(family='DejaVu Sans', size=18)
        ax.text(
            0.5,                       # centro orizzontale
            0.96,                      # posizione sopra la mappa ma dentro il frame
            titolo,
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontproperties=prop,
            color='black',
            bbox=dict(                 # BOX NON arrotondato
                facecolor='white',
                alpha=0.8,
                edgecolor='black',
                linewidth=1
            ),
            zorder=200
        )

    
    gdf_valid = gdf[gdf["classe"] != -1]
    gdf_nodata = gdf[gdf["classe"] == -1]

    if inset_gdf is not None and not inset_gdf.empty:
        inset_gdf.plot(
            ax=ax,
            color="black",
            linewidth=1,
            alpha=1.0,
        )

    gdf_valid.plot(
        column="classe",
        cmap=palette_key,
        legend=False,
        edgecolor="black",
        linewidth=0.0,
        alpha=0.8,
        ax=ax,
    )

    if not gdf_nodata.empty:
        gdf_nodata.plot(
            color=color_nodata,
            edgecolor="black",
            linewidth=0.0,
            alpha=0.6,
            ax=ax,
        )

    if show_basemap:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        except Exception:
            st.warning(T["warning_basemap"])
            cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    shift = 50000
    ax.set_ylim(ymin + shift, ymax + shift)

    ax.add_patch(
        Rectangle(
            (xmin, ymin + shift),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
            zorder=100,
        )
    )

    if method_key == "manual":
        # bounds già impostati manualmente
        bins = bounds[1:]
        min_val = bounds[0]
    else:
        # calcoli automatici della classificazione
        bins = list(classifier.bins)
        min_val = round(gdf[val_col].min(), 2)
        bounds = [min_val] + [round(b, 2) for b in bins]

    if show_legend:
        legend_height = 0.022 * (k + 1)
        top_y = 0.9
        legend_ax = ax.inset_axes(
            [0.75, top_y - legend_height, 0.22, legend_height]
        )
        legend_ax.axis("off")
        legend_ax.add_patch(
            Rectangle(
                (0, 0),
                3,
                len(bounds) - 1,
                facecolor="white",
                edgecolor="black",
                linewidth=1,
                zorder=0,
            )
        )

        for i in range(len(bounds) - 1):
            color = to_hex(cm.get_cmap(palette_key, k)(i))
            y = len(bounds) - i - 2 + 0.2
            label_text = (
                f"{format_value(bounds[i])} – {format_value(bounds[i+1])}"
            )
            legend_ax.add_patch(
                Rectangle(
                    (0.2, y),
                    0.7,
                    0.6,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.3,
                    zorder=1,
                )
            )
            legend_ax.text(
                1.3,
                y + 0.3,
                label_text,
                va="center",
                fontsize=7,
                fontproperties=arial_font,
                zorder=2,
            )

        legend_ax.set_xlim(0, 3)
        legend_ax.set_ylim(0, len(bounds) - 1)

    ax.axis("off")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.92, bottom=0.08)

    st.pyplot(fig)

    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        pdf.savefig(fig)
    pdf_data = pdf_buffer.getvalue()

    png_buffer = BytesIO()
    fig.savefig(png_buffer, format="png", dpi=300)
    png_data = png_buffer.getvalue()

    jpg_buffer = BytesIO()
    img = Image.open(BytesIO(png_data)).convert("RGB")
    img.save(jpg_buffer, format="JPEG", quality=95)
    jpg_data = jpg_buffer.getvalue()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.download_button(
            T["download_pdf_label"],
            data=pdf_data,
            file_name="map.pdf",
            mime="application/pdf",
        ):
            st.success(T["download_pdf_success"])

    with col2:
        if st.download_button(
            T["download_png_label"],
            data=png_data,
            file_name="map.png",
            mime="image/png",
        ):
            st.success(T["download_png_success"])

    with col3:
        if st.download_button(
            T["download_jpg_label"],
            data=jpg_data,
            file_name="map.jpg",
            mime="image/jpeg",
        ):
            st.success(T["download_jpg_success"])
