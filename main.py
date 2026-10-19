import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import math

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 영화 유형 나누기")
st.caption("영화의 흥행 특성을 분석하여 3개 유형으로 분류합니다.")

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

FEATURES = ["스크린 수(로그)", "누적 관객(로그)", "10위권 일수", "롱런 지수"]
LABELS = {
    "스크린 수(로그)": "스크린 수 (log10)",
    "누적 관객(로그)": "누적 관객 (log10)",
    "10위권 일수": "10위권 일수",
    "롱런 지수": "롱런 지수",
}
COLORS = {"㉮": "#E45756", "㉯": "#4C78A8", "㉰": "#54A24B"}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    total_movies = len(df)

    numeric_cols = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "movieNm",
            "first_scrn",
            "first_week_audi",
            "total_audi",
            "days_in_top10",
        ]
    ).copy()

    df = df[df["first_week_audi"] > 0].copy()
    df = df[(df["first_scrn"] >= 0) & (df["total_audi"] >= 0)].copy()

    df["스크린 수(로그)"] = df["first_scrn"].apply(
        lambda x: math.log10(x + 1)
    )
    df["누적 관객(로그)"] = df["total_audi"].apply(
        lambda x: math.log10(x + 1)
    )
    df["롱런 지수"] = (
        df["total_audi"] / df["first_week_audi"]
    ).clip(upper=20)

    df = df.dropna(subset=FEATURES).copy()
    return df, total_movies


st.subheader("1. 데이터 및 속성 선택")

try:
    df, total_movies = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(df) < 3:
    st.error("군집화를 하려면 유효한 영화가 최소 3편 필요합니다.")
    st.stop()

selected_features = st.multiselect(
    "군집화에 사용할 속성 (두 개 이상 선택)",
    FEATURES,
    default=FEATURES,
    format_func=lambda x: LABELS[x],
)

if len(selected_features) < 2:
    st.warning("군집화를 위해 속성을 두 개 이상 선택하세요.")
    st.stop()

# 표준화 및 K-means 군집화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[selected_features])

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)
df["cluster_raw"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰ 부여
cluster_order = (
    df.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
cluster_names = {
    cluster_id: ["㉮", "㉯", "㉰"][i]
    for i, cluster_id in enumerate(cluster_order)
}
df["군집"] = df["cluster_raw"].map(cluster_names)

# 요약 정보
st.subheader("2. 분석 결과 요약")
col1, col2 = st.columns(2)
col1.metric("전체 영화 편수", f"{total_movies:,}편")
col2.metric("군집 분석에 사용한 편수", f"{len(df):,}편")
st.caption(f"제외된 영화: {total_movies - len(df):,}편")

# 2차원 산점도
st.subheader("3. 2차원 산점도")
c1, c2 = st.columns(2)

with c1:
    x_feature = st.selectbox(
        "가로축 속성",
        selected_features,
        index=0,
        format_func=lambda x: LABELS[x],
        key="x2",
    )

with c2:
    y_feature = st.selectbox(
        "세로축 속성",
        selected_features,
        index=1,
        format_func=lambda x: LABELS[x],
        key="y2",
    )

fig2 = px.scatter(
    df,
    x=x_feature,
    y=y_feature,
    color="군집",
    hover_name="movieNm",
    hover_data={
        "first_scrn": ":,",
        "total_audi": ":,",
        "days_in_top10": True,
        "군집": True,
    },
    category_orders={"군집": ["㉮", "㉯", "㉰"]},
    color_discrete_map=COLORS,
    labels=LABELS,
    title="영화 유형별 분포",
)
fig2.update_traces(marker={"size": 7, "opacity": 0.75})
fig2.update_layout(legend_title_text="영화 유형")
st.plotly_chart(fig2, use_container_width=True)

# 3차원 산점도
st.subheader("4. 3차원 산점도")

if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 군집화 속성을 세 개 이상 선택하세요.")
else:
    c1, c2, c3 = st.columns(3)

    with c1:
        x3 = st.selectbox(
            "X축", selected_features, index=0,
            format_func=lambda x: LABELS[x], key="x3"
        )
    with c2:
        y3 = st.selectbox(
            "Y축", selected_features, index=1,
            format_func=lambda x: LABELS[x], key="y3"
        )
    with c3:
        z3 = st.selectbox(
            "Z축", selected_features, index=2,
            format_func=lambda x: LABELS[x], key="z3"
        )

    fig3 = px.scatter_3d(
        df,
        x=x3,
        y=y3,
        z=z3,
        color="군집",
        hover_name="movieNm",
        hover_data={
            "first_scrn": ":,",
            "total_audi": ":,",
            "days_in_top10": True,
            "군집": True,
        },
        category_orders={"군집": ["㉮", "㉯", "㉰"]},
        color_discrete_map=COLORS,
        labels=LABELS,
        title="영화 유형별 3차원 분포",
    )
    fig3.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3.update_layout(legend_title_text="영화 유형")
    st.plotly_chart(fig3, use_container_width=True)

# 군집별 평균 통계
st.subheader("5. 군집별 통계")

stats = (
    df.groupby("군집")
    .agg(
        영화_편수=("movieNm", "size"),
        평균_스크린_수=("first_scrn", "mean"),
        평균_누적_관객=("total_audi", "mean"),
        평균_10위권_일수=("days_in_top10", "mean"),
        평균_롱런_지수=("롱런 지수", "mean"),
    )
    .reindex(["㉮", "㉯", "㉰"])
    .reset_index()
    .rename(columns={"군집": "영화 유형"})
)

stats = stats.rename(
    columns={
        "영화_편수": "영화 편수",
        "평균_스크린_수": "평균 스크린 수",
        "평균_누적_관객": "평균 누적 관객",
        "평균_10위권_일수": "평균 10위권 일수",
        "평균_롱런_지수": "평균 롱런 지수",
    }
)

formatted_stats = stats.copy()
formatted_stats["영화 편수"] = formatted_stats["영화 편수"].map(
    lambda x: f"{x:,}"
)
formatted_stats["평균 스크린 수"] = formatted_stats["평균 스크린 수"].map(
    lambda x: f"{x:,.1f}"
)
formatted_stats["평균 누적 관객"] = formatted_stats["평균 누적 관객"].map(
    lambda x: f"{x:,.1f}"
)
formatted_stats["평균 10위권 일수"] = formatted_stats["평균 10위권 일수"].map(
    lambda x: f"{x:,.2f}"
)
formatted_stats["평균 롱런 지수"] = formatted_stats["평균 롱런 지수"].map(
    lambda x: f"{x:.2f}"
)

st.dataframe(formatted_stats, use_container_width=True, hide_index=True)

# 군집별 누적 관객 상위 5편
st.subheader("6. 유형별 누적 관객 상위 영화")

for group in ["㉮", "㉯", "㉰"]:
    st.markdown(f"#### {group} 유형")

    top5 = (
        df[df["군집"] == group]
        .sort_values("total_audi", ascending=False)
        .head(5)[["movieNm", "total_audi"]]
        .rename(columns={
            "movieNm": "영화 제목",
            "total_audi": "누적 관객",
        })
        .copy()
    )

    if top5.empty:
        st.write("해당 유형에 영화가 없습니다.")
    else:
        top5["누적 관객"] = top5["누적 관객"].map(
            lambda x: f"{x:,.0f}명"
        )
        st.dataframe(top5, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "K-means 3개 군집 · random_state=42 · "
    "통계는 원래 단위의 평균 · 군집 표시는 누적 관객 평균 순"
)
