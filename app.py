"""타이타닉 승객 데이터 탐색용 Streamlit 대시보드."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).parent
SOURCE = "https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv"
CLASSES = {1: "1등석", 2: "2등석", 3: "3등석"}
SEXES = {"female": "여성", "male": "남성"}
OUTCOMES = {1: "생존", 0: "미생존"}
COLORS = {"생존": "#007F83", "미생존": "#A46449"}


@st.cache_data
def load_passengers(path):
    data = pd.read_csv(path, usecols=["survived", "pclass", "sex", "age", "fare"])
    return data.assign(
        객실=data["pclass"].map(CLASSES),
        성별=data["sex"].map(SEXES),
        결과=data["survived"].map(OUTCOMES),
    )


def survival_summary(data, group):
    return data.groupby(group, sort=False)["survived"].agg(
        survivors="sum", total="size", rate="mean"
    )


def selection_insight(data):
    rates = survival_summary(data, "객실")["rate"].sort_values(ascending=False)
    if len(rates) == 1:
        return f"선택한 {rates.index[0]} 승객의 생존율은 **{rates.iloc[0]:.1%}**"
    if rates.iloc[0] == rates.iloc[-1]:
        return f"선택한 객실 등급의 생존율은 모두 **{rates.iloc[0]:.1%}**"
    return (
        f"선택한 승객 · **{rates.index[0]} {rates.iloc[0]:.1%}** · "
        f"**{rates.index[-1]} {rates.iloc[-1]:.1%}** · 생존율 차이 "
        f"**{(rates.iloc[0] - rates.iloc[-1]) * 100:.1f}%p**"
    )


def survival_heatmap(data):
    classes, sexes = list(CLASSES.values()), list(SEXES.values())
    grid = pd.MultiIndex.from_product([classes, sexes], names=["객실", "성별"])
    summary = survival_summary(data, ["객실", "성별"]).reindex(grid)
    rates = summary["rate"].to_numpy().reshape(len(classes), len(sexes))
    labels = [
        f"<b>{rate:.1%}</b><br>{survivors:.0f}/{total:.0f}명" if pd.notna(rate) else ""
        for survivors, total, rate in summary.itertuples(index=False, name=None)
    ]
    figure = go.Figure(go.Heatmap(
        x=sexes, y=classes, z=rates, zmin=0, zmax=1, xgap=5, ygap=5,
        text=[labels[i:i + len(sexes)] for i in range(0, len(labels), len(sexes))],
        texttemplate="%{text}", textfont=dict(color="#FFFFFF", size=14),
        colorscale=[[0, "#26344A"], [1, "#107A7D"]], showscale=False,
        hoverongaps=False, hovertemplate="%{y} · %{x}<br>%{text}<extra></extra>",
    ))
    figure.update_xaxes(side="top", fixedrange=True, showgrid=False)
    figure.update_yaxes(autorange="reversed", fixedrange=True, showgrid=False)
    figure.update_layout(height=219, margin=dict(l=8, r=8, t=27, b=0))
    return figure


def age_survival_chart(data):
    starts = (data["age"] // 10).astype(int) * 10
    rates = survival_summary(data.assign(나이대=starts), "나이대").sort_index().reset_index()
    rates["나이대"] = rates["나이대"].map(lambda start: f"{start}~{start + 9}세")
    figure = px.bar(
        rates, x="나이대", y="rate", text="rate", custom_data=["survivors", "total"],
        color_discrete_sequence=[COLORS["생존"]],
        category_orders={"나이대": [f"{start}~{start + 9}세" for start in range(0, 90, 10)]},
    )
    figure.update_traces(
        texttemplate="%{y:.1%}", textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>생존율 %{y:.1%}<br>%{customdata[0]}명 생존 / %{customdata[1]}명<extra></extra>",
    )
    figure.update_yaxes(range=[0, 1.08], tickformat=".0%", title_text="생존율")
    figure.update_xaxes(title_text=None)
    figure.update_layout(height=290, margin=dict(l=8, r=8, t=18, b=8))
    return figure


def show_survival_by_sex(data):
    groups = data.groupby("성별")["survived"].agg(["sum", "size"])
    shares = {}
    rows = []
    for label in SEXES.values():
        if label not in groups.index:
            continue
        survivors, total = groups.loc[label]
        share = survivors / total
        shares[label] = share
        rows.append(f"""
        <div class="sex-share-row">
          <div class="sex-share-label">
            <span><strong>{label}</strong><small>{survivors:,}명 생존 / {total:,}명</small></span>
            <strong class="sex-share-value">{share:.1%}</strong>
          </div>
          <div class="sex-share-track" aria-hidden="true">
            <div style="width:{share:.6%}"></div>
          </div>
        </div>
        """)

    difference = ""
    if len(shares) == 2:
        gap = abs(shares["여성"] - shares["남성"]) * 100
        difference = f'<div class="sex-share-gap">생존율 차이 <strong>{gap:.1f}%p</strong></div>'

    st.html(f"""
    <style>
      .sex-share {{ padding: .4rem 0 .15rem; }}
      .sex-share-row + .sex-share-row {{ margin-top: 1.2rem; }}
      .sex-share-label {{
        display: flex; align-items: baseline; justify-content: space-between;
        gap: .75rem; margin-bottom: .5rem;
      }}
      .sex-share-label small {{
        display: block; font-size: .79rem; line-height: 1.45; opacity: .72;
      }}
      .sex-share-value {{
        font-size: 1.65rem; font-weight: 650; line-height: 1;
        letter-spacing: -.035em; font-variant-numeric: tabular-nums;
      }}
      .sex-share-track {{
        height: .625rem; overflow: hidden; border-radius: 3px;
        background: color-mix(in srgb, currentColor 15%, transparent);
      }}
      .sex-share-track div {{ height: 100%; background: #168A8D; }}
      .sex-share-gap {{
        margin-top: 1.1rem; padding-top: .75rem; font-size: .82rem;
        border-top: 1px solid color-mix(in srgb, currentColor 16%, transparent);
      }}
      .sex-share-gap strong {{ margin-left: .35rem; font-variant-numeric: tabular-nums; }}
    </style>
    <div class="sex-share">{''.join(rows)}{difference}</div>
    """)


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
st.markdown(selection_insight(view))

left, right = st.columns([1.25, 1])
with left.container(border=True):
    st.markdown("**객실 등급 × 성별 생존율**")
    st.plotly_chart(survival_heatmap(view), width="stretch", config={"displayModeBar": False})
    st.caption("생존 인원 / 전체 인원 · 빈 칸은 선택 조건에 해당하는 승객이 없어요.")
with right.container(border=True):
    st.markdown("**성별 생존율**")
    show_survival_by_sex(view)

with st.container(border=True):
    st.markdown("**나이대별 생존 분포**")
    known_age = view.dropna(subset=["age"])
    if known_age.empty:
        st.info("현재 선택에는 나이가 기록된 승객이 없어요.")
    else:
        age_mode = st.segmented_control(
            "나이 그래프 보기", ["인원수", "생존율"], default="인원수", label_visibility="collapsed"
        )
        if age_mode == "생존율":
            st.plotly_chart(age_survival_chart(known_age), width="stretch")
            st.caption("표본이 적은 나이대는 몇 명의 차이로도 생존율이 크게 달라질 수 있어요.")
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

with st.container(border=True):
    st.markdown("**객실 등급과 생존 여부에 따른 운임 분포**")
    fares = px.box(
        view, x="객실", y="fare", color="결과", color_discrete_map=COLORS,
        category_orders={"객실": list(CLASSES.values()), "결과": list(OUTCOMES.values())},
        labels={"fare": "운임"}, points="outliers",
    )
    fares.update_layout(height=320, margin=dict(l=8, r=8, t=16, b=8), legend_title_text="")
    st.plotly_chart(fares, width="stretch")
    st.caption("상자 가운데 선은 중앙값이고, 상자는 가운데 50%의 운임 범위예요.")

with st.expander(f"선택한 승객 데이터 · {len(view):,}명"):
    table = view[["객실", "성별", "age", "fare", "결과"]].rename(columns={"age": "나이", "fare": "운임"})
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button(
        "선택한 데이터 CSV 저장",
        data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name="selected_titanic.csv",
        mime="text/csv",
    )

