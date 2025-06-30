import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Dashboard Galaxias", layout="wide")

# Cargar datos
st.title("📊 Dashboard de Parámetros Físicos de Galaxias")
st.markdown("Este dashboard permite explorar las propiedades físicas obtenidas de los archivos `.fit`")

# Cargar CSV
@st.cache_data
def cargar_datos():
    return pd.read_csv("analisis_galaxias.csv")

data = cargar_datos()

# Sidebar
st.sidebar.header("🎛️ Filtros")
z_min, z_max = st.sidebar.slider("Redshift", float(data["redshift"].min()), float(data["redshift"].max()), (float(data["redshift"].min()), float(data["redshift"].max())))
mstar_min, mstar_max = st.sidebar.slider("log(M*)", float(np.log10(data["Mstar"].min())), float(np.log10(data["Mstar"].max())), (8.0, 11.5))

# Aplicar filtros
df_filtrado = data[
    (data["redshift"] >= z_min) &
    (np.log10(data["Mstar"]) >= mstar_min) &
    (np.log10(data["Mstar"]) <= mstar_max)
]

st.markdown(f"**Galaxias seleccionadas:** {df_filtrado.shape[0]}")

# Estadísticas
st.subheader("📈 Estadísticas resumidas")
st.dataframe(df_filtrado.describe().T)

# Histograma
st.subheader("🔍 Histograma de un parámetro")
param = st.selectbox("Selecciona parámetro para histograma", df_filtrado.columns)
fig1, ax1 = plt.subplots()
sns.histplot(df_filtrado[param], bins=30, kde=True, ax=ax1)
st.pyplot(fig1)

# Diagrama de dispersión
st.subheader("🌌 Diagrama de dispersión")
x_param = st.selectbox("Eje X", df_filtrado.columns, index=df_filtrado.columns.get_loc("Mstar"))
y_param = st.selectbox("Eje Y", df_filtrado.columns, index=df_filtrado.columns.get_loc("sSFR"))
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df_filtrado, x=x_param, y=y_param, ax=ax2)
st.pyplot(fig2)

# Mapa de correlación
if st.checkbox("📉 Mostrar mapa de correlaciones"):
    st.subheader("Mapa de correlaciones")
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    sns.heatmap(df_filtrado.corr(numeric_only=True), cmap='coolwarm', annot=True, fmt=".2f", ax=ax3)
    st.pyplot(fig3)
