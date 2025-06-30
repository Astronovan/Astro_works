import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Dashboard Galaxias", layout="wide")

# Clasificar en grupos de redshift
def clasificar_redshift(z):
    if 0.0069 <= z < 0.057:
        return "Grupo 1 (0.0069–0.057)"
    elif 0.057 <= z < 0.164:
        return "Grupo 2 (0.057–0.164)"
    elif 0.164 <= z <= 0.348:
        return "Grupo 3 (0.164–0.348)"
    else:
        return "Fuera de rango"

# Cargar CSV
@st.cache_data
def cargar_datos():
    df = pd.read_csv("analisis_galaxias.csv")
    df["Grupo_z"] = df["redshift"].apply(clasificar_redshift)
    return df

# Cargar y mostrar datos
data = cargar_datos()
st.title("📊 Dashboard de Parámetros Físicos de Galaxias")

# Sidebar: filtros
st.sidebar.header("🎛️ Filtros")

z_min, z_max = st.sidebar.slider("Redshift", float(data["redshift"].min()), float(data["redshift"].max()), (float(data["redshift"].min()), float(data["redshift"].max())))
m_min, m_max = st.sidebar.slider("log(M*)", float(np.log10(data["Mstar"].min())), float(np.log10(data["Mstar"].max())), (8.0, 11.5))
sfr_min, sfr_max = st.sidebar.slider("SFR (M☉/año)", float(data["SFR"].min()), float(data["SFR"].max()), (float(data["SFR"].min()), float(data["SFR"].max())))

# Aplicar filtros
df_filtrado = data[
    (data["redshift"] >= z_min) &
    (np.log10(data["Mstar"]) >= m_min) &
    (np.log10(data["Mstar"]) <= m_max) &
    (data["SFR"] >= sfr_min) &
    (data["SFR"] <= sfr_max)
]

st.markdown(f"**Galaxias seleccionadas:** {df_filtrado.shape[0]}")

# Descargar datos
st.download_button("📥 Descargar CSV filtrado", df_filtrado.to_csv(index=False), file_name="galaxias_filtradas.csv")

# Estadísticas resumidas
st.subheader("📈 Estadísticas Globales")
st.dataframe(df_filtrado.describe().T)

# Histograma
st.subheader("🔍 Histograma")
param = st.selectbox("Selecciona parámetro", df_filtrado.columns)
fig1, ax1 = plt.subplots()
sns.histplot(df_filtrado[param], bins=30, kde=True, ax=ax1)
st.pyplot(fig1)

# Diagrama de dispersión con ajuste
st.subheader("🌌 Dispersión + Ajuste lineal")
x_param = st.selectbox("🧭 Eje X", df_filtrado.columns, index=df_filtrado.columns.get_loc("Mstar"))
log_x = st.checkbox("🔢 Escala logarítmica para X")

y_param = st.selectbox("🧭 Eje Y", df_filtrado.columns, index=df_filtrado.columns.get_loc("sSFR"))
log_y = st.checkbox("🔢 Escala logarítmica para Y")


x_vals = df_filtrado[x_param]
y_vals = df_filtrado[y_param]

# Aplicar logaritmo si se solicita (y si no hay valores negativos o cero)
if log_x:
    x_vals = np.log10(x_vals.replace(0, np.nan))
if log_y:
    y_vals = np.log10(y_vals.replace(0, np.nan))

fig2, ax2 = plt.subplots()
sns.scatterplot(x=x_vals, y=y_vals, hue=df_filtrado["Grupo_z"], ax=ax2)
sns.regplot(x=x_vals, y=y_vals, scatter=False, ax=ax2, color="black", line_kws={"label": "Ajuste global"}, ci=None)

ax2.set_xlabel(f"log({x_param})" if log_x else x_param)
ax2.set_ylabel(f"log({y_param})" if log_y else y_param)
ax2.legend()

# Correlaciones
try:
    pearson_r, pearson_p = stats.pearsonr(x_vals.dropna(), y_vals.dropna())
    spearman_r, spearman_p = stats.spearmanr(x_vals.dropna(), y_vals.dropna())
    st.markdown(f"**Pearson r = {pearson_r:.3f}, p = {pearson_p:.3g}**")
    st.markdown(f"**Spearman ρ = {spearman_r:.3f}, p = {spearman_p:.3g}**")
except:
    st.warning("⚠️ No se pudo calcular la correlación con estos ejes.")



fig2, ax2 = plt.subplots()
sns.scatterplot(data=df_filtrado, x=x_param, y=y_param, hue="Grupo_z", ax=ax2)
sns.regplot(data=df_filtrado, x=x_param, y=y_param, scatter=False, ax=ax2, color="black", line_kws={"label": "Ajuste global"}, ci=None)
ax2.legend()

try:
    pearson_r, pearson_p = stats.pearsonr(df_filtrado[x_param], df_filtrado[y_param])
    spearman_r, spearman_p = stats.spearmanr(df_filtrado[x_param], df_filtrado[y_param])
    st.markdown(f"**Pearson r = {pearson_r:.3f}, p = {pearson_p:.3g}**")
    st.markdown(f"**Spearman ρ = {spearman_r:.3f}, p = {spearman_p:.3g}**")
except Exception as e:
    st.warning(f"⚠️ No se pudo calcular la correlación: {e}")

st.pyplot(fig2)

# Análisis por grupo
st.subheader("🔎 Análisis por grupo de redshift")

for grupo in sorted(df_filtrado["Grupo_z"].unique()):
    sub_df = df_filtrado[df_filtrado["Grupo_z"] == grupo]
    st.markdown(f"### {grupo} ({len(sub_df)} galaxias)")
    if len(sub_df) < 3:
        st.markdown("🔹 Muy pocos datos para análisis.")
        continue
    try:
        r, p = stats.pearsonr(sub_df[x_param], sub_df[y_param])
        rho, p_s = stats.spearmanr(sub_df[x_param], sub_df[y_param])
        st.markdown(f"- **Pearson r = {r:.3f}, p = {p:.3g}**")
        st.markdown(f"- **Spearman ρ = {rho:.3f}, p = {p_s:.3g}**")
    except:
        st.markdown("❌ Error en el cálculo estadístico.")
