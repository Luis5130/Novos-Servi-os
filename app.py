import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- Carregar dados ---
@st.cache_data
def carregar_dados():
    """
    Carrega os dados para a aplicação a partir de um arquivo CSV.
    Certifique-se de que o arquivo 'Preços Herois.csv' esteja na mesma pasta do 'app.py'.
    """
    try:
        df = pd.read_csv("Preços Herois.csv")
        # Renomear colunas para compatibilidade com o código do histograma
        # 'nm_servico' -> 'servico' e 'vl_preco' -> 'price'
        # Assumindo que o CSV já tem as colunas 'nm_servico', 'tipo_compra', 'quantidade_captada', 'vl_preco'
        df.rename(columns={'nm_servico': 'servico', 'vl_preco': 'price'}, inplace=True)
        return df
    except FileNotFoundError:
        st.error("Arquivo 'Preços Herois.csv' não encontrado. Por favor, certifique-se de que o arquivo está na mesma pasta do 'app.py'.")
        st.stop() # Para a execução do Streamlit se o arquivo não for encontrado
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o CSV: {e}")
        st.stop()


# Carrega os dados uma vez e armazena em cache
df = carregar_dados()

st.title("📊 Histograma Interativo de Preços por Serviço") # Título principal ajustado

# --- Filtros na barra lateral ---
st.sidebar.header("Filtros")

# Filtro por Serviço
todos_servicos = sorted(df["servico"].unique())
selecionar_todos_servicos = st.sidebar.checkbox("Selecionar todos os serviços", value=True, key="serv_all")
if selecionar_todos_servicos:
    servicos_selecionados = st.sidebar.multiselect("Serviço", todos_servicos, default=todos_servicos, key="serv_select")
else:
    servicos_selecionados = st.sidebar.multiselect("Serviço", todos_servicos, key="serv_select_manual")

# Filtro por Tipo de Compra (mantido para filtrar os dados, mas não para gerar um histograma separado)
todos_tipos_compra = sorted(df["tipo_compra"].unique())
selecionar_todos_tipos_compra = st.sidebar.checkbox("Selecionar todos os tipos de compra", value=True, key="tipo_all")
if selecionar_todos_tipos_compra:
    tipos_compra_selecionados = st.sidebar.multiselect("Tipo de Compra", todos_tipos_compra, default=todos_tipos_compra, key="tipo_select")
else:
    tipos_compra_selecionados = st.sidebar.multiselect("Tipo de Compra", todos_tipos_compra, key="tipo_select_manual")

# Novo filtro para Quantidade Captada
todos_quantidades_captada = sorted(df["quantidade_captada"].unique())
selecionar_todas_quantidades_captada = st.sidebar.checkbox("Selecionar todas as quantidades captadas", value=True, key="qtd_all")
if selecionar_todas_quantidades_captada:
    quantidades_captada_selecionadas = st.sidebar.multiselect("Quantidade Captada", todos_quantidades_captada, default=todos_quantidades_captada, key="qtd_select")
else:
    quantidades_captada_selecionadas = st.sidebar.multiselect("Quantidade Captada", todos_quantidades_captada, key="qtd_select_manual")


# --- Filtrar dados com base nas seleções do usuário ---
df_filtrado = df[
    (df["servico"].isin(servicos_selecionados)) &
    (df["tipo_compra"].isin(tipos_compra_selecionados)) &
    (df["quantidade_captada"].isin(quantidades_captada_selecionadas)) # Adicionado o novo filtro
]

# --- Função auxiliar para plotar histogramas ---
def plotar_histograma(df_data, group_col, title_prefix):
    """
    Plota um histograma interativo usando Plotly.

    Args:
        df_data (pd.DataFrame): DataFrame com os dados a serem plotados.
        group_col (str): Nome da coluna para agrupar (ex: 'servico', 'tipo_compra').
        title_prefix (str): Prefixo para o título do subcabeçalho.
    """
    st.subheader(title_prefix)

    if df_data.empty:
        st.warning("Nenhum dado encontrado com os filtros selecionados.")
        return

    bin_size = 20 # Tamanho de cada "barra" do histograma (faixa de preço)
    max_price = df_data["price"].max()

    # Garante que os 'bins' sejam criados corretamente, mesmo se o preço máximo for 0
    if max_price == 0:
        bins = np.arange(0, bin_size * 2, bin_size)
    else:
        bins = np.arange(0, max_price + bin_size, bin_size)

    hist_data = []
    all_counts_per_bin = np.zeros(len(bins) - 1) # Para somar as contagens de cada bin (para a altura da linha da média)

    # Itera sobre os valores únicos da coluna de agrupamento para criar as barras empilhadas
    for group_val in df_data[group_col].unique():
        df_group = df_data[df_data[group_col] == group_val]
        counts, _ = np.histogram(df_group["price"], bins=bins)

        # Acumula as contagens para calcular a altura máxima para a linha da média
        all_counts_per_bin += counts

        # Calcula o percentual de cada faixa de preço dentro do grupo
        percent = (counts / counts.sum()) * 100 if counts.sum() > 0 else np.zeros_like(counts)
        # Cria os rótulos para as faixas de preço
        labels = [f"R${bins[i]:.0f} - R${bins[i+1]-1:.0f}" for i in range(len(counts))]

        hist_data.append(go.Bar(
            x=labels,
            y=counts,
            name=str(group_val), # Converte para string para exibição no Plotly
            hovertemplate="<br>".join([
                f"{group_col.replace('_', ' ').title()}: " + str(group_val),
                "Faixa de preço: %{x}",
                "Quantidade: %{y}",
                "Percentual: %{customdata:.1f}%",
            ]),
            customdata=percent
        ))

    media = df_data["price"].mean()
    # Altura máxima para a linha da média no gráfico empilhado
    max_y_value = max(all_counts_per_bin) if len(all_counts_per_bin) > 0 else 0

    # Determina a posição X da linha da média no histograma
    mean_bin_index = np.digitize(media, bins) - 1
    if len(labels) > 0 and 0 <= mean_bin_index < len(labels):
        mean_x_position = labels[mean_bin_index]
    else:
        mean_x_position = None # Não há bin válido para a média

    layout_shapes = []
    layout_annotations = []

    # Adiciona a linha da média e sua anotação se houver uma posição X válida
    if mean_x_position:
        layout_shapes.append(
            dict(
                type="line",
                x0=mean_x_position,
                x1=mean_x_position,
                y0=0,
                y1=max_y_value,
                line=dict(color="black", dash="dash"),
            )
        )
        layout_annotations.append(
            dict(
                x=mean_x_position,
                y=max_y_value,
                text=f"Média: R${media:.2f}",
                showarrow=True,
                arrowhead=1,
                yshift=10 # Desloca a anotação um pouco acima da linha
            )
        )

    layout = go.Layout(
        title=f"Distribuição de Preços por {group_col.replace('_', ' ').title()}",
        xaxis_title="Faixa de Preço (R$)",
        yaxis_title="Quantidade",
        barmode="stack", # Empilha as barras para cada grupo
        bargap=0.05,
        shapes=layout_shapes,
        annotations=layout_annotations
    )

    fig = go.Figure(data=hist_data, layout=layout)
    st.plotly_chart(fig, use_container_width=True)

    # Expansor para ver os dados filtrados em tabela
    with st.expander("🔍 Ver dados filtrados"):
        st.dataframe(df_data.reset_index(drop=True))

# --- Plotar Histograma de Serviços ---
plotar_histograma(df_filtrado, "servico", "Distribuição de Preços por Serviço")

# As linhas abaixo foram removidas para atender à sua solicitação:
# st.markdown("---")
# plotar_histograma(df_filtrado, "tipo_compra", "Distribuição de Preços por Tipo de Compra")
