import pandas as pd
import plotly.express as px
import plotly.io as pio

data = pd.read_csv('./data/train.csv')
numeric_columns = data.select_dtypes(include=['number']).columns

histograms_html = ""
for col in numeric_columns:
    fig = px.histogram(data, x=col, nbins=30, title=f"Distribution de {col}", marginal="box")
    histograms_html += pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

# Sauvegarde des histogrammes en HTML
with open("./plots/histogrammes.html", "w", encoding="utf-8") as f:
    f.write(histograms_html)

boxplots_html = ""
for col in numeric_columns:
    fig = px.box(data, y=col, title=f"Boxplot de {col}")
    boxplots_html += pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

# Sauvegarde des boxplots en HTML
with open("./plots/boxplots.html", "w", encoding="utf-8") as f:
    f.write(boxplots_html)

# Calcul des corrélations avec CO2
correlation_matrix = data[numeric_columns].corr()
correlation_matrix = correlation_matrix.reset_index().melt(id_vars='index')

# Générer la heatmap (un peu) interactive
fig = px.imshow(
    data[numeric_columns].corr(),
    text_auto=True,
    color_continuous_scale="RdBu",
    title="Matrice de Corrélation"
)

# Sauvegarde de la heatmap en HTML
with open("./plots/correlation_matrix.html", "w", encoding="utf-8") as f:
    f.write(pio.to_html(fig, full_html=True, include_plotlyjs='cdn'))
