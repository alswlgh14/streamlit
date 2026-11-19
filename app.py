"""수업용 가상 데이터를 살펴보는 Streamlit 대시보드."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

HERE = Path(__file__).parent
COLORS = {"A": "#006D77", "B": "#B45122"}
METRICS = {"매출": "sales", "영업이익": "profit"}


@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    return data.sort_values(["branch", "month"])


def format_chart(figure, height):
    figure.update_layout(
        height=height,
        margin=dict(l=10, r=16, t=18, b=12),
        legend=dict(title_text="", orientation="h", y=1.04, x=0),
        hoverlabel=dict(namelength=-1),
    )
    return figure


st.set_page_config(page_title="지점 실적 대시보드", layout="wide")
st.title("지점 실적 대시보드")
st.caption("2026년 월별 실적 · 수업용 가상 데이터 · 금액 단위는 백만원")

try:
    data = load_data(HERE / "sales.csv")
except FileNotFoundError:
    st.error("app.py와 같은 폴더에 sales.csv를 넣어 주세요.")
    st.stop()

branch_options = sorted(data["branch"].unique().tolist())
with st.container(border=True):
    branch_col, month_col, metric_col = st.columns([1.1, 1.4, 1])
    branches = branch_col.multiselect(
        "지점", branch_options, default=branch_options, format_func=lambda branch: f"{branch}지점"
    )
    months = month_col.slider("조회 기간", 1, 12, (1, 12), format="%d월")
    metric_label = metric_col.selectbox("비교 지표", list(METRICS))

metric = METRICS[metric_label]
view = data.loc[data["branch"].isin(branches) & data["month"].between(*months)]

if view.empty:
    st.warning("표시할 데이터가 없습니다. 지점을 하나 이상 선택하거나 조회 기간을 바꿔 주세요.")
    st.stop()

st.subheader(f"{months[0]}~{months[1]}월 실적 요약")
sales_total = view["sales"].sum()
profit_total = view["profit"].sum()
margin = profit_total / sales_total * 100 if sales_total else None
sales_card, profit_card, margin_card = st.columns(3)
sales_card.metric("매출", f"{sales_total:,.0f} 백만원", border=True)
profit_card.metric("영업이익", f"{profit_total:,.0f} 백만원", border=True)
margin_card.metric("영업이익률", f"{margin:.1f}%" if margin is not None else "—", border=True)

labels = {"branch": "지점", "month": "월", metric: f"{metric_label} (백만원)", "margin": "영업이익률 (%)"}
chart_options = dict(color="branch", color_discrete_map=COLORS, labels=labels)
totals = view.groupby("branch", as_index=False)[["sales", "profit"]].sum()
totals["margin"] = totals["profit"].div(totals["sales"].where(totals["sales"].ne(0))).mul(100)

with st.container(border=True):
    st.markdown(f"**월별 {metric_label} 추이**")
    trend = px.line(view, x="month", y=metric, markers=True, **chart_options)
    trend.update_xaxes(dtick=1, ticksuffix="월", range=[months[0] - 0.35, months[1] + 0.35])
    trend.update_yaxes(rangemode="tozero")
    st.plotly_chart(format_chart(trend, 330), width="stretch")

total_col, margin_col = st.columns(2)
with total_col.container(border=True):
    st.markdown(f"**지점별 {metric_label} 합계**")
    comparison = px.bar(totals, x="branch", y=metric, text_auto=",.0f", **chart_options)
    comparison.update_layout(showlegend=False)
    st.plotly_chart(format_chart(comparison, 270), width="stretch")

with margin_col.container(border=True):
    st.markdown("**지점별 영업이익률**")
    rates = px.bar(totals, x="branch", y="margin", text_auto=".1f", **chart_options)
    rates.update_layout(showlegend=False)
    rates.update_yaxes(ticksuffix="%")
    st.plotly_chart(format_chart(rates, 270), width="stretch")
    st.caption("영업이익률은 기간 영업이익 합계를 기간 매출 합계로 나눈 값이에요.")

with st.expander(f"선택한 데이터 보기 · {len(view)}행"):
    st.dataframe(
        view.rename(columns={"month": "월", "branch": "지점", "sales": "매출", "profit": "영업이익"}),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "선택한 데이터 CSV 저장",
        data=view.to_csv(index=False).encode("utf-8-sig"),
        file_name="selected_sales.csv",
        mime="text/csv",
    )

