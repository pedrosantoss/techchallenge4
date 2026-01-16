import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import ta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import datetime
from sklearn.metrics import (
    accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error,
    f1_score, confusion_matrix, roc_curve
)

# -------------------------------
# Configuração da página
# -------------------------------
st.set_page_config(page_title="PAINEL IBOVESPA", layout="wide")
st.header("**PAINEL IBOVESPA**")

# -------------------------------
# Funções auxiliares
# -------------------------------
def formatar_reais(valor):
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def flatten_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))
        ohlc_set = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if "Close" in lvl1 or (ohlc_set & lvl1):
            df.columns = df.columns.get_level_values(1)
        elif "Close" in lvl0 or (ohlc_set & lvl0):
            df.columns = df.columns.get_level_values(0)
        else:
            if len(df.columns) == 6:
                df.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            else:
                raise ValueError(f"Não foi possível achatar colunas: {list(df.columns)}")
    return df

def ensure_close(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            raise KeyError(f"Close não encontrado nas colunas: {list(df.columns)}")
    return df

def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(df["Close"], errors="coerce")

    df["retorno"] = close.pct_change()
    df["close_diff"] = close.diff()
    df["log_close"] = np.log(close)
    df["log_diff"] = df["log_close"].diff()

    df["rsi"] = ta.momentum.RSIIndicator(close=close).rsi()
    df["macd"] = ta.trend.MACD(close=close).macd()
    df["sma"] = ta.trend.SMAIndicator(close=close, window=22).sma_indicator()
    df["ema"] = ta.trend.EMAIndicator(close=close, window=22).ema_indicator()

    df["volatilidade"] = df["retorno"].rolling(window=10).std()
    df["sma_ratio"] = df["sma"] / df["ema"]

    for col in ["retorno", "log_diff", "rsi", "macd"]:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        for win in [5, 10]:
            df[f"{col}_roll{win}"] = df[col].rolling(win).mean()

    df["target"] = (close.shift(-1) > close).astype(int)
    df["ds"] = df.index
    df = df.dropna()
    return df

def estatisticas_basicas(df):
    return {
        "Média": float(df["Close"].mean()),
        "Mediana": float(df["Close"].median()),
        "Mínimo": float(df["Close"].min()),
        "Máximo": float(df["Close"].max())
    }

# -------------------------------
# 1) Selecionar número de anos
# -------------------------------
anos = st.slider("Selecione o número de anos de histórico", 1, 20, 5)

# Calcular datas de início e fim (da data atual para trás)
fim = datetime.datetime.today()
inicio = fim - datetime.timedelta(days=anos*365)

df = yf.download("^BVSP", start=inicio, end=fim, interval="1d", auto_adjust=True, group_by="column")
if df.empty:
    st.error("⚠️ Nenhum dado foi retornado. Tente outro período.")
    st.stop()

df = flatten_ohlc(df)
df = ensure_close(df)
df.index = pd.to_datetime(df.index)
df = df.dropna(subset=["Open", "High", "Low", "Close"])
df = criar_features(df)

# Informações rápidas
col1, col2, col3 = st.columns(3)
with col1: st.write("**Índice:** Ibovespa")
with col2: st.write("**Tipo:** Índice de ações")
with col3: st.write(f"**Último Fechamento:** {formatar_reais(float(df['Close'].iloc[-1]))}")

# -------------------------------
# 2) Gráfico de fechamento (Candlestick + SMA/EMA)
# -------------------------------
st.subheader("Evolução do Fechamento")
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="Candlestick")])

# Adicionar médias móveis
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['sma'],
    line=dict(color='blue', width=1),
    name="SMA 22"))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df['ema'],
    line=dict(color='orange', width=1),
    name="EMA 22"))

fig.update_layout(
    title="Evolução do Ibovespa",
    xaxis_title="Data",
    yaxis_title="Fechamento",
    template="plotly_dark",
    xaxis_rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 3) Análise exploratória
# -------------------------------
stats = estatisticas_basicas(df)
colA, colB, colC, colD = st.columns(4)
colA.metric("Média de Fechamento", formatar_reais(stats["Média"]))
colB.metric("Mediana de Fechamento", formatar_reais(stats["Mediana"]))
colC.metric("Máximo de Fechamento", formatar_reais(stats["Máximo"]))
colD.metric("Mínimo de Fechamento", formatar_reais(stats["Mínimo"]))

# -------------------------------
# 4) Gráfico do volume negociado
# -------------------------------
st.subheader("Volume Negociado")
if "Volume" in df.columns:
    if anos > 1:
        anos_volume = st.slider(
            "Selecione o número de anos para o gráfico de volume",
            min_value=1,
            max_value=anos,
            value=min(anos, 3)
        )
    else:
        anos_volume = 1  # fixa em 1 ano quando o período inicial é 1

    # calcular intervalo de datas para o volume
    fim_vol = datetime.datetime.today()
    inicio_vol = fim_vol - datetime.timedelta(days=anos_volume*365)

    df_filtrado = df.loc[(df.index >= inicio_vol) & (df.index <= fim_vol)]

    fig = px.bar(
        df_filtrado, x=df_filtrado.index, y="Volume",
        title=f"Volume Negociado - Últimos {anos_volume} anos",
        labels={"Volume": "Volume", "index": "Data"},
        color="Volume", color_continuous_scale="Blues"
    )
    fig.update_layout(template="plotly_dark", bargap=0.1,
                      xaxis_title="Data", yaxis_title="Volume")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 5,6,7) Previsão, métricas e tendência
# -------------------------------
st.subheader("Previsão com Modelo Híbrido - ARIMA + XGBoost")
serie_prev = None
rmse = mae = acc = auc = f1 = None
direcao = None

try:
    with open("modelo_ibovespahibrido.pkl", "rb") as f:
        modelo_hibrido = pickle.load(f)

    modelo_arima = modelo_hibrido.get("modelo_arima")
    modelo_xgb   = modelo_hibrido.get("modelo_xgb")
    scaler_xgb   = modelo_hibrido.get("scaler")
    features     = modelo_hibrido.get("features", [])

    def selecionar_features_numericas(df_local, features_lista):
        feats_existentes = [f for f in features_lista if f in df_local.columns]
        return df_local[feats_existentes].select_dtypes(include=[np.number])

    def previsao_hibrida(df_full, modelo_arima, modelo_xgb, scaler, features_lista, passos):
        if len(df_full) < passos + 1:
            raise ValueError(f"Dados insuficientes: necessário ao menos {passos + 1} linhas após features.")
        # ARIMA -> vetor 1D
        previsao_log = modelo_arima.forecast(steps=passos)
        previsao_log = np.asarray(previsao_log).reshape(-1)
        previsao_arima = np.exp(previsao_log)
        # XGB -> features consistentes e vetor 1D
        df_tail = df_full.tail(passos)
        X = selecionar_features_numericas(df_tail, features_lista)
        X_scaled = scaler.transform(X.values)
        previsao_residuo = np.asarray(modelo_xgb.predict(X_scaled)).reshape(-1)
        # Soma e retorna 1D
        return (previsao_arima + previsao_residuo).reshape(-1)

    # --- 6) Tendência do próximo pregão ---
    st.subheader("Tendência do próximo pregão")
    previsao_1d = previsao_hibrida(df, modelo_arima, modelo_xgb, scaler_xgb, features, 1)
    previsao_proximo = float(previsao_1d[0])
    ultimo_fechamento = df["Close"].iloc[-1]
    direcao = "O Índice Ibovespa vai subir" if previsao_proximo > ultimo_fechamento else "📉 O Índice Ibovespa vai descer"
    st.write(direcao)

    # --- 5) Métricas de validação ---
    st.subheader("Métricas de validação - teste com últimos 30 dias do período selecionado")
    n_dias_metricas = 30
    y_real = df["Close"].tail(n_dias_metricas).values
    y_pred = previsao_hibrida(df, modelo_arima, modelo_xgb, scaler_xgb, features, n_dias_metricas)
    fechamento_base = df["Close"].iloc[-n_dias_metricas-1:-1].values
    y_bin_real = (y_real > fechamento_base).astype(int)
    y_bin_pred = (y_pred > fechamento_base).astype(int)

    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    acc = accuracy_score(y_bin_real, y_bin_pred)
    auc = roc_auc_score(y_bin_real, y_bin_pred)
    f1 = f1_score(y_bin_real, y_bin_pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:.2f}")
    c2.metric("MAE", f"{mae:.2f}")
    c3.metric("Acurácia", f"{acc:.2%}")
    c4, c5 = st.columns(2)
    c4.metric("AUC", f"{auc:.2f}")
    c5.metric("F1-score", f"{f1:.2f}")

    # --- Avaliação Gráfica lado a lado ---
   
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        cm = confusion_matrix(y_bin_real, y_bin_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel("Previsto")
        ax_cm.set_ylabel("Real")
        ax_cm.set_title("Matriz de Confusão")
        st.pyplot(fig_cm)

    with col_g2:
        fpr, tpr, _ = roc_curve(y_bin_real, y_bin_pred)
        fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
        ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color="red")
        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_title("Curva ROC")
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, alpha=0.3)
        st.pyplot(fig_roc)


    # --- 7) Slider e previsão para mais dias ---
    dias = st.slider("Dias à frente para prever", 2, 30, 5)
    previsao = previsao_hibrida(df, modelo_arima, modelo_xgb, scaler_xgb, features, dias)
    inicio = df.index[-1] + pd.Timedelta(days=1)
    idx_prev = pd.date_range(start=inicio, periods=len(previsao), freq="B")
    serie_prev = pd.Series(previsao, index=idx_prev, name="Previsão")

    st.subheader("Previsão para os próximos dias")
    fig_prev = px.line(
        x=serie_prev.index, y=serie_prev.values,
        title="Previsão Ibovespa - Próximos Dias",
        labels={"x": "Data", "y": "Fechamento previsto"},
        markers=True
    )
    fig_prev.update_traces(line=dict(color="orange"))
    fig_prev.update_layout(template="plotly_dark", xaxis_title="Data", yaxis_title="Fechamento Previsto")
    st.plotly_chart(fig_prev, use_container_width=True)

    
except FileNotFoundError:
    st.error("⚠️ Arquivo 'modelo_ibovespahibrido.pkl' não encontrado. Salve o modelo antes de rodar o app.")
except Exception as e:
    st.error(f"Erro na previsão: {e}")

# -------------------------------
# 8) Salvar resultados
# -------------------------------
st.subheader("Exportar Resultados da Previsão")
formato_saida = st.selectbox("Escolha o formato de exportação", ["CSV", "JSON"])
nome_saida = st.text_input("Nome do arquivo (sem extensão)", value="resultados_previsao")

botao = st.button("Salvar resultados")

if botao:
    if serie_prev is None or any(v is None for v in [rmse, mae, acc, auc, f1, direcao]):
        st.error("Não há resultados de previsão/métricas para salvar. Verifique se o modelo foi carregado com sucesso.")
    else:
        df_resultado = pd.DataFrame({
            "Data": serie_prev.index,
            "Previsão": serie_prev.values,
            "RMSE": [rmse] * len(serie_prev),
            "MAE": [mae] * len(serie_prev),
            "Acurácia": [acc] * len(serie_prev),
            "AUC": [auc] * len(serie_prev),
            "F1-score": [f1] * len(serie_prev),
            "Tendência": [direcao] * len(serie_prev)
        })
        if formato_saida == "CSV":
            df_resultado.to_csv(f"{nome_saida}.csv", index=False)
            st.success(f"Arquivo {nome_saida}.csv salvo com sucesso!")
        else:
            df_resultado.to_json(f"{nome_saida}.json", orient="records", date_format="iso")
            st.success(f"Arquivo {nome_saida}.json salvo com sucesso!")

