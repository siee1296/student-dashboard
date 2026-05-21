"""
외국인유학생 모니터링·예측 대시보드 v2
- L3_aggregated/dashboard_data.xlsx 읽기 (PII 없음, 집계만)
- 10개 탭: 국적별/GPA/교육과정·성별/코호트/발생경로/어학급수/위험군/예측모델/시점예측/인증지표
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="외국인유학생 모니터링 대시보드",
    page_icon="🎓",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"

# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dashboard(path: str) -> dict[str, pd.DataFrame]:
    sheets: dict[str, pd.DataFrame] = {}
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
        for name in xl.sheet_names:
            sheets[name] = xl.parse(name)
    except Exception as e:
        st.error(f"파일 로드 실패: {e}")
    return sheets


def _find_local_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.xlsx"), reverse=True)


# ─── 사이드바: 파일 선택 ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎓 외국인유학생 대시보드")
    st.divider()

    local_files = _find_local_files()
    data_path: str | None = None

    if local_files:
        names = [f.name for f in local_files]
        sel = st.selectbox("데이터 파일 선택", names, key="file_sel")
        data_path = str(DATA_DIR / sel)
        st.caption(f"경로: `data/{sel}`")
    else:
        st.info("data/ 폴더에 dashboard_data.xlsx 없음")

    st.divider()
    uploaded = st.file_uploader("또는 파일 직접 업로드", type=["xlsx"])
    if uploaded:
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        tmp.write(uploaded.read())
        tmp.close()
        data_path = tmp.name

    st.divider()
    st.caption("집계 데이터만 표시됩니다. 개별 학생 정보 없음.")

# ─── 데이터 없음 처리 ─────────────────────────────────────────────────────────

if data_path is None:
    st.title("🎓 외국인유학생 모니터링 대시보드 v2")
    st.info(
        "좌측 사이드바에서 데이터 파일을 선택하거나 업로드하세요.\n\n"
        "백오피스에서 L3 산출 후 `data/dashboard_data.xlsx`를 이 리포지토리에 복사하면 "
        "자동으로 표시됩니다."
    )
    st.stop()

sheets = load_dashboard(data_path)

# _info 시트에서 메타 읽기
_info = sheets.get("_info", pd.DataFrame())
snap_date = "—"
if not _info.empty and "key" in _info.columns and "value" in _info.columns:
    meta = dict(zip(_info["key"], _info["value"]))
    snap_date = meta.get("snap_date", "—")
    generated_at = meta.get("generated_at", "—")
else:
    generated_at = "—"


# ─── 헤더 ─────────────────────────────────────────────────────────────────────

st.title("🎓 외국인유학생 모니터링·예측 대시보드")
st.markdown(
    f"**스냅샷**: `{snap_date}`  &nbsp;|&nbsp;  **산출일시**: `{generated_at}`"
)
st.divider()


# ─── 공통 헬퍼 ────────────────────────────────────────────────────────────────

def _is_stub(df: pd.DataFrame) -> bool:
    return df is None or df.empty or (len(df.columns) == 1 and "note" in df.columns)


def _show_stub(msg: str = "데이터 미산출"):
    st.info(msg)


def _bar(df, x, y, title="", color=None, h=350):
    kwargs = dict(x=x, y=y, title=title, text_auto=True)
    if color:
        kwargs["color"] = color
        kwargs["color_continuous_scale"] = "Blues"
    fig = px.bar(df, **kwargs)
    fig.update_layout(height=h, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def _table(df, note=""):
    st.dataframe(df, use_container_width=True, hide_index=True)
    if note:
        st.caption(note)


# ─── 10개 탭 ──────────────────────────────────────────────────────────────────

TAB_LABELS = [
    "🌏 국적별",
    "📊 GPA",
    "🎓 교육과정·성별",
    "📅 코호트",
    "🔍 발생경로",
    "🗣 어학급수",
    "⚠️ 위험군",
    "🤖 예측모델",
    "📈 시점예측",
    "📋 인증지표",
]

tabs = st.tabs(TAB_LABELS)


# ── 탭1: 국적별 ───────────────────────────────────────────────────────────────
with tabs[0]:
    df = sheets.get("agg_by_nationality")
    if _is_stub(df):
        _show_stub()
    else:
        nat_col = df.columns[0]
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 국적별 인원")
            top = df.nlargest(15, "인원") if "인원" in df.columns else df
            _bar(top, x=nat_col, y="인원", title="국적별 인원 (상위 15)", color="인원")
        with col_b:
            if "불체율(%)" in df.columns:
                st.markdown("##### 국적별 불법체류율")
                _bar(top, x=nat_col, y="불체율(%)", title="불체율(%) by 국적", color="불체율(%)")
            elif "불체수" in df.columns:
                st.markdown("##### 국적별 불체 인원")
                _bar(top, x=nat_col, y="불체수", title="불체수 by 국적")
        st.divider()
        _table(df, note="집계 데이터 — 개별 학생 정보 없음")


# ── 탭2: GPA ──────────────────────────────────────────────────────────────────
with tabs[1]:
    df = sheets.get("agg_by_gpa")
    if _is_stub(df):
        _show_stub()
    else:
        gpa_col = df.columns[0]
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### GPA 구간별 인원")
            _bar(df, x=gpa_col, y="인원", title="GPA 구간별 인원 분포", color="인원")
        with col_b:
            if "불체율(%)" in df.columns:
                st.markdown("##### GPA 구간별 불법체류율")
                _bar(df, x=gpa_col, y="불체율(%)", title="불체율(%) by GPA 구간")
        st.divider()
        _table(df)


# ── 탭3: 교육과정·성별 ────────────────────────────────────────────────────────
with tabs[2]:
    df = sheets.get("agg_by_curriculum_gender")
    if _is_stub(df):
        _show_stub()
    else:
        curr_col   = "교육과정구분" if "교육과정구분" in df.columns else df.columns[0]
        gender_col = "성별"         if "성별"         in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 교육과정 × 성별 인원")
            if gender_col and gender_col in df.columns:
                fig = px.bar(
                    df, x=curr_col, y="인원", color=gender_col,
                    barmode="group", title="교육과정 × 성별 인원",
                    text_auto=True,
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                _bar(df, x=curr_col, y="인원", title="교육과정별 인원")
        with col_b:
            if "불체율(%)" in df.columns and gender_col:
                st.markdown("##### 교육과정 × 성별 불체율")
                pivot = df.pivot_table(index=curr_col, columns=gender_col, values="불체율(%)", aggfunc="mean")
                fig = px.imshow(
                    pivot.fillna(0), text_auto=".2f",
                    title="불체율(%) 히트맵",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        st.divider()
        _table(df)


# ── 탭4: 코호트 ───────────────────────────────────────────────────────────────
with tabs[3]:
    df = sheets.get("agg_by_cohort")
    if _is_stub(df):
        _show_stub()
    else:
        cohort_col = df.columns[0]
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 입학코호트별 인원 추이")
            fig = px.line(df, x=cohort_col, y="인원", markers=True, title="코호트별 입학 인원")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            if "불체율(%)" in df.columns:
                st.markdown("##### 입학코호트별 불체율")
                fig = px.line(df, x=cohort_col, y="불체율(%)", markers=True, title="코호트별 불체율(%)")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        st.divider()
        _table(df)


# ── 탭5: 발생경로 ─────────────────────────────────────────────────────────────
with tabs[4]:
    df = sheets.get("agg_by_route")
    if _is_stub(df):
        _show_stub("수기 입력(발생경로) 데이터 없음 — 백오피스에서 입력 후 재산출 필요")
    else:
        route_col = df.columns[0]
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 발생경로별 인원")
            _bar(df, x=route_col, y="인원", title="불체 발생경로별 인원", color="인원")
        with col_b:
            if "불체율(%)" in df.columns:
                st.markdown("##### 발생경로별 불체율")
                _bar(df, x=route_col, y="불체율(%)", title="불체율(%) by 발생경로")
        st.divider()
        _table(df)


# ── 탭6: 어학급수 ─────────────────────────────────────────────────────────────
with tabs[5]:
    df = sheets.get("agg_by_lang")
    if _is_stub(df):
        _show_stub()
    else:
        lang_col = df.columns[0]
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 어학급수별 인원")
            _bar(df, x=lang_col, y="인원", title="어학급수별 인원 분포", color="인원")
        with col_b:
            if "불체율(%)" in df.columns:
                st.markdown("##### 어학급수별 불체율")
                _bar(df, x=lang_col, y="불체율(%)", title="불체율(%) by 어학급수")
        st.divider()
        _table(df)


# ── 탭7: 위험군 v2 ────────────────────────────────────────────────────────────
with tabs[6]:
    df = sheets.get("risk_group_v2")
    if _is_stub(df):
        note_val = df.iloc[0]["note"] if (df is not None and not df.empty and "note" in df.columns) else "—"
        _show_stub(f"위험군 데이터 없음 — {note_val}")
    else:
        tier_col = "risk_tier" if "risk_tier" in df.columns else df.columns[0]

        # 상단 KPI
        k_cols = st.columns(len(df))
        tier_order = {"고위험": 0, "중위험": 1, "저위험": 2, "미산출": 3}
        df_sorted = df.copy()
        if tier_col in df_sorted.columns:
            df_sorted["_ord"] = df_sorted[tier_col].map(tier_order).fillna(99)
            df_sorted = df_sorted.sort_values("_ord").drop(columns=["_ord"])
        for i, row in df_sorted.iterrows():
            with k_cols[list(df_sorted.index).index(i)]:
                tier = str(row.get(tier_col, "—"))
                n = int(row.get("인원", 0))
                avg = row.get("평균위험점수", None)
                delta = f"평균점수 {avg:.4f}" if avg is not None and str(avg) != "nan" else ""
                icon = {"고위험": "🔴", "중위험": "🟡", "저위험": "🟢"}.get(tier, "⚪")
                st.metric(f"{icon} {tier}", f"{n:,}명", delta=delta)

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 위험군 분포")
            if "인원" in df_sorted.columns:
                fig = px.pie(
                    df_sorted, names=tier_col, values="인원",
                    title="위험군 구성 비율",
                    color_discrete_map={"고위험": "#d9534f", "중위험": "#f0ad4e",
                                        "저위험": "#5cb85c", "미산출": "#aaa"},
                )
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)
        with col_b:
            st.markdown("##### 위험군별 평균 위험점수")
            if "평균위험점수" in df_sorted.columns:
                _bar(df_sorted, x=tier_col, y="평균위험점수",
                     title="위험군별 평균 위험점수", color="평균위험점수", h=320)

        if "실제탈락률" in df_sorted.columns:
            st.divider()
            st.markdown("##### 위험군별 실제 중도탈락률")
            df_s2 = df_sorted.copy()
            df_s2["실제탈락률(%)"] = (df_s2["실제탈락률"] * 100).round(2)
            _bar(df_s2, x=tier_col, y="실제탈락률(%)",
                 title="위험군별 실제 탈락률", color="실제탈락률(%)", h=300)

        st.divider()
        _table(df_sorted, note="고위험: 위험점수 상위 20%, 중위험: 20~50%, 저위험: 하위 50%")


# ── 탭8: 예측모델 ─────────────────────────────────────────────────────────────
with tabs[7]:
    df = sheets.get("model_results")
    if _is_stub(df):
        note_val = df.iloc[0]["note"] if (df is not None and not df.empty and "note" in df.columns) else "—"
        _show_stub(f"모델 결과 없음 — {note_val}")
    else:
        st.markdown("##### 5모델 성능 비교")

        def _highlight_max(s):
            try:
                vals = pd.to_numeric(s, errors="coerce")
                is_max = vals == vals.max()
                return ["background-color:#d4edda" if v else "" for v in is_max]
            except Exception:
                return [""] * len(s)

        style = df.style
        for col in ["PR-AUC(holdout)", "ROC-AUC(holdout)", "Recall@Top15%"]:
            if col in df.columns:
                style = style.apply(_highlight_max, subset=[col])
        st.dataframe(style, use_container_width=True, hide_index=True)

        # PR-AUC 막대 차트
        pr_col = "PR-AUC(holdout)"
        model_col = "모델" if "모델" in df.columns else df.columns[0]
        if pr_col in df.columns:
            plot_df = df[df[pr_col] != "-"].copy()
            plot_df["_pr"] = pd.to_numeric(plot_df[pr_col], errors="coerce")
            plot_df = plot_df.dropna(subset=["_pr"])
            if not plot_df.empty:
                st.divider()
                st.markdown("##### PR-AUC 비교 (holdout 30%)")
                fig = px.bar(
                    plot_df.sort_values("_pr", ascending=False),
                    x=model_col, y="_pr",
                    title="모델별 PR-AUC (holdout)",
                    labels={"_pr": "PR-AUC"},
                    color="_pr", color_continuous_scale="Blues",
                    text_auto=".4f",
                )
                fig.update_layout(height=350, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "PR-AUC: 클래스 불균형 환경 핵심 지표. "
            "Recall@Top15%: 전체 탈락자 중 상위 15% 위험군에서 포착되는 비율."
        )


# ── 탭9: 시점예측 (생존곡선) ─────────────────────────────────────────────────
with tabs[8]:
    df = sheets.get("survival_curves")
    if _is_stub(df):
        _show_stub()
    else:
        sem_col = "이수학기" if "이수학기" in df.columns else df.columns[0]
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 이수학기별 생존확률")
            if "생존확률" in df.columns:
                fig = px.line(
                    df, x=sem_col, y="생존확률", markers=True,
                    title="생존곡선 (KM 근사)",
                    labels={"생존확률": "생존확률 (1=전원생존)"},
                )
                fig.update_layout(height=350, yaxis_range=[0, 1.05])
                fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                              annotation_text="50% 생존", annotation_position="right")
                st.plotly_chart(fig, use_container_width=True)
            else:
                _show_stub("생존확률 컬럼 없음")
        with col_b:
            st.markdown("##### 이수학기별 순간위험률")
            if "순간위험률" in df.columns:
                fig = px.bar(
                    df, x=sem_col, y="순간위험률",
                    title="순간위험률 (Hazard Rate)",
                    color="순간위험률", color_continuous_scale="Reds",
                )
                fig.update_layout(height=350, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
        st.divider()
        _table(df, note="KM 근사: 이수학기 = 학번 기준 현재까지 이수한 학기 수")


# ── 탭10: 인증지표 ────────────────────────────────────────────────────────────
with tabs[9]:
    df = sheets.get("indicator_values")
    if _is_stub(df):
        _show_stub()
    else:
        ind_col = "지표" if "지표" in df.columns else df.columns[0]
        val_col = "값"   if "값"   in df.columns else (df.columns[1] if len(df.columns) > 1 else None)

        # 핵심 KPI 강조
        KEY_INDICATORS = [
            "불법체류율(%)", "불체율(%)", "불법체류율(%)_공식",
            "중도탈락률(%)", "중도탈락률(%)_공식",
            "전체학생수", "재학생수", "중도탈락자수", "불체발생수",
        ]

        if val_col:
            kpi_rows = df[df[ind_col].isin(KEY_INDICATORS)].copy()
            if not kpi_rows.empty:
                st.markdown("##### 핵심 인증 지표")
                kpi_cols = st.columns(min(len(kpi_rows), 4))
                for i, (_, row) in enumerate(kpi_rows.iterrows()):
                    with kpi_cols[i % 4]:
                        label = str(row[ind_col])
                        val   = row[val_col]
                        display = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}" if isinstance(val, int) else str(val)
                        st.metric(label, display)
                st.divider()

        st.markdown("##### 전체 집계 지표")
        _table(df, note=f"스냅샷: {snap_date}  |  교육국제화역량 인증제 기준")
