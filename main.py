import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 영화 유형 나누기")
st.caption("영화의 흥행·상영 특성을 분석해 비슷한 영화끼리 세 묶음으로 나눕니다.")

FEATURES = {
    "스크린 수 (로그)": "log_first_scrn",
    "누적 관객 (로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "longrun_index",
}

CLUSTER_NAMES = ["㉮", "㉯", "㉰"]


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    required = [
        "movieNm",
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    missing = [
        col for col in required if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "필수 열이 없습니다: " + ", ".join(missing)
        )

    numeric_cols = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["movieNm"] = df["movieNm"].fillna("제목 없음").astype(str)

    valid = (
        df["first_scrn"].gt(0)
        & df["first_week_audi"].gt(0)
        & df["total_audi"].gt(0)
        & df["days_in_top10"].ge(0)
    )

    df["log_first_scrn"] = np.nan
    df["log_total_audi"] = np.nan
    df["longrun_index"] = np.nan

    df.loc[valid, "log_first_scrn"] = np.log10(
        df.loc[valid, "first_scrn"]
    )

    df.loc[valid, "log_total_audi"] = np.log10(
        df.loc[valid, "total_audi"]
    )

    df.loc[valid, "longrun_index"] = (
        df.loc[valid, "total_audi"]
        / df.loc[valid, "first_week_audi"]
    ).clip(upper=20)

    feature_cols = list(FEATURES.values())

    usable = (
        valid
        & df[feature_cols].notna().all(axis=1)
        & np.isfinite(df[feature_cols]).all(axis=1)
    )

    return df, df.loc[usable].copy()


try:
    all_data, usable_data = load_data()
except Exception as error:
    st.error("데이터를 불러오거나 처리하지 못했습니다.")
    st.exception(error)
    st.stop()

st.write(
    f"전체 영화: **{len(all_data):,}편** "
    f"| 분석 가능 영화: **{len(usable_data):,}편**"
)

selected_features = st.multiselect(
    "군집 분석에 사용할 속성을 두 개 이상 선택하세요.",
    options=list(FEATURES.keys()),
    default=list(FEATURES.keys()),
)

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 선택해 주세요.")
    st.stop()

if len(usable_data) < 3:
    st.error("분석 가능한 영화가 3편 미만이라 군집 분석을 할 수 없습니다.")
    st.stop()

selected_cols = [
    FEATURES[name] for name in selected_features
]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(
    usable_data[selected_cols].astype(float)
)

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

usable_data["cluster"] = model.fit_predict(scaled_data)
usable_data["영화 유형"] = usable_data["cluster"].map(
    dict(enumerate(CLUSTER_NAMES))
)

st.subheader("📊 영화 유형 분포")

col1, col2 = st.columns(2)

with col1:
    x_feature = st.selectbox(
        "가로축",
        selected_features,
        index=0,
    )

with col2:
    y_feature = st.selectbox(
        "세로축",
        selected_features,
        index=1,
    )

fig = px.scatter(
    usable_data,
    x=FEATURES[x_feature],
    y=FEATURES[y_feature],
    color="영화 유형",
    hover_name="movieNm",
    hover_data=[
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ],
    labels={
        FEATURES[x_feature]: x_feature,
        FEATURES[y_feature]: y_feature,
    },
    title="비슷한 특성을 가진 영화 세 묶음",
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🎞️ 유형별 영화 분석")

for cluster_id, cluster_name in enumerate(CLUSTER_NAMES):
    group = usable_data[
        usable_data["cluster"] == cluster_id
    ].copy()

    with st.expander(
        f"{cluster_name} 유형 · {len(group)}편",
        expanded=True,
    ):
        if group.empty:
            st.write("해당 유형의 영화가 없습니다.")
            continue

        st.write("**주요 특성 (중앙값)**")

        st.write(
            f"- 스크린 수: {group['first_scrn'].median():,.0f}개"
        )
        st.write(
            f"- 누적 관객: {group['total_audi'].median():,.0f}명"
        )
        st.write(
            f"- 10위권 일수: {group['days_in_top10'].median():.1f}일"
        )
        st.write(
            f"- 롱런 지수: {group['longrun_index'].median():.2f}"
        )

        table = group[
            [
                "movieNm",
                "first_scrn",
                "first_week_audi",
                "total_audi",
                "days_in_top10",
                "longrun_index",
            ]
        ].sort_values(
            "total_audi",
            ascending=False,
        ).rename(
            columns={
                "movieNm": "영화 제목",
                "first_scrn": "스크린 수",
                "first_week_audi": "첫 주 관객 수",
                "total_audi": "누적 관객 수",
                "days_in_top10": "10위권 일수",
                "longrun_index": "롱런 지수",
            }
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

st.info(
    "분석 안내: K-means 알고리즘이 선택한 속성을 표준화한 뒤 "
    "영화들을 세 그룹으로 나눕니다. 유형 기호는 순위가 아닙니다. "
    "롱런 지수는 누적 관객 수를 첫 주 관객 수로 나눈 값이며, "
    "계산 결과가 지나치게 커지는 것을 막기 위해 최대 20으로 제한했습니다."
)
