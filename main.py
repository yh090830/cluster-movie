import streamlit as st
import pandas as pd
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
st.caption("영화의 흥행·상영 특성을 바탕으로 비슷한 영화끼리 세 묶음으로 나눕니다.")

FEATURES = {
    "스크린 수 (로그)": "log_first_scrn",
    "누적 관객 (로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "longrun_index",
}
DISPLAY_FEATURES = list(FEATURES.keys())
ORIGINAL_MEAN_COLUMNS = {
    "스크린 수": "first_scrn",
    "누적 관객": "total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "longrun_index",
}
CLUSTER_NAMES = ["㉮", "㉯", "㉰"]


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    required = [
        "movieCd", "movieNm", "first_scrn", "first_week_audi",
        "total_audi", "days_in_top10",
    ]
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError("필수 열을 찾을 수 없습니다: " + ", ".join(missing_columns))

    for column in ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["movieNm"] = df["movieNm"].fillna("제목 없음").astype(str)
    df["log_first_scrn"] = pd.NA
    df["log_total_audi"] = pd.NA

    # 상용로그는 양수에서만 정의되므로 스크린 수와 누적 관객도 양수인지 확인합니다.
    valid = (
        df["first_scrn"].notna()
        & df["total_audi"].notna()
        & df["first_week_audi"].notna()
        & df["days_in_top10"].notna()
        & (df["first_scrn"] > 0)
        & (df["total_audi"] > 0)
        & (df["first_week_audi"] > 0)
    )

    df.loc[valid, "log_first_scrn"] = __import__("numpy").log10(
        df.loc[valid, "first_scrn"].astype(float)
    )
    df.loc[valid, "log_total_audi"] = __import__("numpy").log10(
        df.loc[valid, "total_audi"].astype(float)
    )
    df["longrun_index"] = pd.NA
    df.loc[valid, "longrun_index"] = (
        df.loc[valid, "total_audi"] / df.loc[valid, "first_week_audi"]
    ).clip(upper=20)

    feature_columns = list(FEATURES.values())
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    usable = valid & df[feature_columns].notna().all(axis=1)
    usable_df = df.loc[usable].copy()
    return df, usable_df


try:
    all_data, usable_data = load_data()
except Exception as error:
    st.error("데이터를 불러오거나 처리하지 못했습니다.")
    st.exception(error)
    st.stop()

total_count = len(all_data)
usable_count = len(usable_data)

st.markdown(
    f"**전체 영화: {total_count:,}편　|　묶기에 사용한 영화: {usable_count:,}편**"
)

selected_features = st.multiselect(
    "묶는 데 사용할 속성을 선택하세요. (두 개 이상 선택)",
