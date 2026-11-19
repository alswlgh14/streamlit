"""타이타닉 승객 데이터 탐색용 Streamlit 대시보드."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

HERE = Path(__file__).parent
SOURCE = "https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv"
CLASSES = {1: "1등석", 2: "2등석", 3: "3등석"}
SEXES = {"female": "여성", "male": "남성"}
OUTCOMES = {1: "생존", 0: "미생존"}
COLORS = {"생존": "#007F83", "미생존": "#A46449"}


@st.cache_data
def load_passengers(path):
    data = pd.read_csv(path, usecols=["survived", "pclass", "sex", "age"])
    return data.assign(
        객실=data["pclass"].map(CLASSES),
        성별=data["sex"].map(SEXES),
        결과=data["survived"].map(OUTCOMES),
    )


def survival_chart(data, group, order):
    rates = (
        data.groupby(group, as_index=False, sort=False)
        .agg(생존율=("survived", "mean"), 인원=("survived", "size"), 생존=("survived", "sum"))
    )
    figure = px.bar(
        rates, x=group, y="생존율", text="생존율", custom_data=["생존", "인원"],
        category_orders={group: order}, color_discrete_sequence=["#007F83"],
    )
    figure.update_traces(
        texttemplate="%{y:.1%}", textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>생존율 %{y:.1%}<br>%{customdata[0]}명 생존 / %{customdata[1]}명<extra></extra>",
    )
    figure.update_yaxes(range=[0, 1.05], tickformat=".0%", title_text="생존율")
    figure.update_xaxes(title_text=None)
    figure.update_layout(height=265, margin=dict(l=8, r=8, t=18, b=8))
    return figure


st.set_page_config(page_title="타이타닉 승객 생존 분석", layout="wide")
st.title("타이타닉 승객 생존 분석")
st.caption(f"객실 등급·성별·나이에 따라 생존율이 어떻게 달랐는지 살펴보세요. · [데이터 출처: seaborn-data]({SOURCE})")

try:
    passengers = load_passengers(HERE / "titanic.csv")
except FileNotFoundError:
    st.error("app.py와 같은 폴더에 titanic.csv를 넣어 주세요.")
    st.stop()

with st.container(border=True):
    class_col, sex_col, age_col = st.columns([1.2, 1, 1.4])
    classes = class_col.multiselect("객실 등급", list(CLASSES.values()), default=list(CLASSES.values()))
    sexes = sex_col.multiselect("성별", list(SEXES.values()), default=list(SEXES.values()))
    ages = age_col.slider("나이 범위", 0, 80, (0, 80), format="%d세")
    include_unknown = age_col.checkbox("나이 미기록 승객도 포함", value=True)

age_match = passengers["age"].between(*ages)
if include_unknown:
    age_match |= passengers["age"].isna()
view = passengers.loc[passengers["객실"].isin(classes) & passengers["성별"].isin(sexes) & age_match]
if view.empty:
    st.warning("조건에 맞는 승객이 없어요. 객실 등급, 성별 또는 나이 범위를 바꿔 보세요.")
    st.stop()

survived = int(view["survived"].sum())
count_col, survived_col, rate_col = st.columns(3)
count_col.metric("선택한 승객", f"{len(view):,}명", border=True)
survived_col.metric("생존 승객", f"{survived:,}명", border=True)
rate_col.metric("생존율", f"{survived / len(view):.1%}", border=True)

left, right = st.columns(2)
with left.container(border=True):
    st.markdown("**객실 등급별 생존율**")
    st.plotly_chart(survival_chart(view, "객실", list(CLASSES.values())), width="stretch")
with right.container(border=True):
    st.markdown("**성별 생존율**")
    st.plotly_chart(survival_chart(view, "성별", list(SEXES.values())), width="stretch")

with st.container(border=True):
    st.markdown("**나이대별 승객 수**")
    known_age = view.dropna(subset=["age"])
    if known_age.empty:
        st.info("현재 선택에는 나이가 기록된 승객이 없어요.")
    else:
        age_histogram = px.histogram(
            known_age, x="age", color="결과", color_discrete_map=COLORS,
            category_orders={"결과": list(OUTCOMES.values())},
            labels={"age": "나이 (세)"},
        )
        age_histogram.update_traces(xbins=dict(start=0, end=90, size=10), xhoverformat=".0f")
        age_histogram.update_xaxes(dtick=10, range=[0, 90])
        age_histogram.update_yaxes(title_text="승객 수 (명)")
        age_histogram.update_layout(
            height=290, margin=dict(l=8, r=8, t=10, b=8),
            legend=dict(title_text="", orientation="h", x=0, y=1.10),
        )
        st.plotly_chart(age_histogram, width="stretch")
    if view["age"].isna().any():
        st.caption(f"나이가 기록되지 않은 {view['age'].isna().sum():,}명은 이 차트에서 제외했어요.")

with st.expander(f"선택한 승객 데이터 · {len(view):,}명"):
    table = view[["객실", "성별", "age", "결과"]].rename(columns={"age": "나이"})
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button(
        "선택한 데이터 CSV 저장",
        data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name="selected_titanic.csv",
        mime="text/csv",
    )

