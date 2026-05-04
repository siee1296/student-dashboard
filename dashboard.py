import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import date
import io
import warnings
warnings.filterwarnings("ignore")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

st.set_page_config(page_title="유학생 불법체류 실무 분석", page_icon="📊", layout="wide")

CR = "#E63946"; CW = "#F4A261"; CS = "#2A9D8F"
CI = "#457B9D"; CD = "#1d3557"; CY = "#e9c46a"

st.markdown("""<style>
.block-container{padding:1.5rem 2.5rem 2rem!important}
.kf{background:#1d3557;color:#fff;padding:16px 22px;border-radius:8px;
    font-size:1.05em;line-height:1.75;margin-bottom:14px}
.kf b{color:#F4A261}.kf .r{color:#E63946;font-weight:700}
.kf .g{color:#56CFB2;font-weight:700}
.warn-box{background:#fff3cd;border-left:4px solid #F4A261;
    padding:14px 18px;border-radius:4px;margin:10px 0;font-size:.97em}
.info-box{background:#d1ecf1;border-left:4px solid #457B9D;
    padding:14px 18px;border-radius:4px;margin:10px 0;font-size:.97em}
.policy-box{background:#f8d7da;border-left:4px solid #E63946;
    padding:16px 20px;border-radius:4px;margin:12px 0;line-height:1.8}
</style>""", unsafe_allow_html=True)

FILE = "기초자료v2.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(FILE, sheet_name="통합_원자료")

df_raw = load_data()

# ── 전처리 ─────────────────────────────────────────────────────────
df = df_raw.copy()
df["불체_flag"] = (df["불체여부"] == "Y").astype(int)
df["입학년도"]  = pd.to_numeric(df["입학년도"], errors="coerce")

_VALID = ["베트남","우즈베키스탄","네팔","몽골","미얀마","기타"]
df["국적_그룹"] = df["국적_그룹"].apply(lambda x: x if x in _VALID else "기타")

gpa_bins   = [-0.001, 0.005, 1.5, 2.0, 2.5, 3.0, 3.5, 4.51]
gpa_labels = ["0점","0.01~1.5","1.5~2.0","2.0~2.5","2.5~3.0","3.0~3.5","3.5+"]
df["GPA구간"] = pd.cut(df["평점평균"], bins=gpa_bins, labels=gpa_labels)

_ORD = ["영어트랙","무시험","저급(1~2)","중급(3~4)","고급(5~7)"]
def _lg(x):
    if pd.isna(x): return None
    s = str(x).strip()
    if s == "영어트랙": return "영어트랙"
    if s == "무시험":   return "무시험"
    try:
        n = int(float(s))
        return "저급(1~2)" if n<=2 else ("중급(3~4)" if n<=4 else "고급(5~7)")
    except Exception: return None

df["어학급수그룹"] = df["어학급수"].apply(_lg)
df["어학원출신"]   = df["어학원급수"].apply(lambda x: "어학원출신" if pd.notna(x) else "비출신")
df["현재비자"]     = df["현재비자"].apply(lambda x: "미파악" if pd.isna(x) else str(x).strip())
df["성별"]        = df["성별"].fillna("미상")
df["이수학기"]    = pd.to_numeric(df["이수학기"], errors="coerce")
df["취득학점"]    = pd.to_numeric(df["취득학점"], errors="coerce")
df["분모포함"]    = df["분모포함"].fillna("해당없음").astype(str).str.strip()
df["불체연도_n"]  = pd.to_numeric(df["불체연도"], errors="coerce")
df["체류만료일"]  = pd.to_datetime(df["체류만료일"], errors="coerce")
df["만료_무기한"] = df["체류만료일"].dt.year >= 9000

# ══ 데이터 범위 분리 ═══════════════════════════════════════════════
# 불체자(분자): 2019~2025 전수, 불체여부=Y
df_bu_all     = df[df["불체여부"] == "Y"].copy()
# 분모: 2023~2025 입학생
df_denom_base = df[df["입학년도"].between(2023, 2025)].copy()
# 코호트 분석 전용 (예외): 2019~2025 전체
df_coh_all    = df[df["입학년도"].between(2019, 2025)].copy()

# ── 사이드바 ────────────────────────────────────────────────────────
st.sidebar.header("🔧 분석 필터")
year_sel = st.sidebar.multiselect("분석 연도 (분모 기준)", [2023,2024,2025], default=[2023,2024,2025])
if not year_sel: year_sel = [2023,2024,2025]
과정_sel = st.sidebar.selectbox("교육과정", ["전체","일반과정","전공심화"])
국적_sel = st.sidebar.selectbox("국적 그룹",
    ["전체","베트남","우즈베키스탄","네팔","몽골","미얀마","기타"])

def flt_bu(course, nat):
    d = df_bu_all.copy()
    if course != "전체": d = d[d["교육과정구분"] == course]
    if nat    != "전체": d = d[d["국적_그룹"] == nat]
    return d

def flt_dn(yrs, course, nat):
    d = df_denom_base[df_denom_base["입학년도"].isin(yrs)].copy()
    if course != "전체": d = d[d["교육과정구분"] == course]
    if nat    != "전체": d = d[d["국적_그룹"] == nat]
    return d

df_bu_f    = flt_bu(과정_sel, 국적_sel)
df_dn_f    = flt_dn(year_sel, 과정_sel, 국적_sel)
df_bu_exec = flt_bu(과정_sel, "전체")
df_dn_exec = flt_dn(year_sel, 과정_sel, "전체")

df_coh = df_coh_all.copy()
if 과정_sel != "전체": df_coh = df_coh[df_coh["교육과정구분"] == 과정_sel]

st.sidebar.markdown("---")
st.sidebar.markdown("**현재 필터 대상**")
st.sidebar.markdown(f"- 분모(2023~2025): **{len(df_dn_f):,}명**")
_sb = len(df_bu_f)
_sp = float(_sb)/float(len(df_dn_f))*100 if len(df_dn_f)>0 else 0.0
st.sidebar.markdown(f"- 불체자(전수): **{_sb}명** ({_sp:.1f}%)")

# ── 헬퍼 ────────────────────────────────────────────────────────────
def calc_rate(bu_data, denom_data, grp):
    """불체 분자=bu_data, 분모=denom_data로 분리 집계"""
    grp_l = [grp] if isinstance(grp, str) else list(grp)
    tot_g = denom_data.groupby(grp_l, dropna=True).size().reset_index(name="총원")
    bu_g  = bu_data.groupby(grp_l, dropna=True).size().reset_index(name="불체")
    g = tot_g.merge(bu_g, on=grp_l, how="left").fillna({"불체": 0})
    g["불체"] = g["불체"].astype(int)
    g["불체율(%)"] = g["불체"] / g["총원"] * 100
    return g

def calc_rate_single(d, grp):
    """코호트용: 단일 df, 불체_flag 집계"""
    grp_l = [grp] if isinstance(grp, str) else list(grp)
    g = (d.groupby(grp_l, dropna=True)
          .agg(총원=("불체_flag","count"), 불체=("불체_flag","sum")).reset_index())
    g["불체율(%)"] = g["불체"] / g["총원"] * 100
    return g

def kf(text): st.markdown(f'<div class="kf">{text}</div>', unsafe_allow_html=True)
def p(n, tot): return float(n)/float(tot)*100 if float(tot)>0 else 0.0
def f1(v): return f"{float(v):.1f}"
def f0(v): return f"{float(v):.0f}"

# ── 사전 계산 ──────────────────────────────────────────────────────
tot_f = len(df_dn_f)
bu_f  = len(df_bu_f)
r_f   = p(bu_f, tot_f)

pc_f = df_bu_f["불체발생경로"].value_counts()
미등N=int(pc_f.get("미등록제적",0)); 미등P=p(미등N,bu_f)
학변N=int(pc_f.get("학적변동",0));   학변P=p(학변N,bu_f)
졸업N=int(pc_f.get("졸업",0));       졸업P=p(졸업N,bu_f)
입학N=int(pc_f.get("입학실패",0));   입학P=p(입학N,bu_f)

# 국적 추이: 코호트 전체(2019~2025) 기준 — 탭4와 동일한 예외 적용
yr_nat = (df_coh.groupby(["입학년도","국적_그룹"])
          .agg(총원=("불체_flag","count"), 불체=("불체_flag","sum")).reset_index())
yr_nat["불체율(%)"] = yr_nat["불체"] / yr_nat["총원"] * 100

def ntrd(nat):
    s = yr_nat[yr_nat["국적_그룹"]==nat].sort_values("입학년도")
    return (list(s["불체"].astype(int)), list(s["총원"].astype(int)),
            [round(float(v),1) for v in s["불체율(%)"]])

uz_b,uz_t,uz_r = ntrd("우즈베키스탄")
mn_b,mn_t,mn_r = ntrd("몽골")
vn_b,vn_t,vn_r = ntrd("베트남")
np_b,np_t,np_r = ntrd("네팔")
mm_b,mm_t,mm_r = ntrd("미얀마")
bu_exec = len(df_bu_exec)
vn_비중 = p(sum(vn_b), len(df_bu_all))
nm_max  = max(np_r+mm_r) if (np_r or mm_r) else 0.0

gbt     = df_bu_f[df_bu_f["평점평균"].notna() & df_bu_f["이수학기"].notna()]
gbt_low = len(gbt[gbt["평점평균"]<2.0])
gbt_2   = len(gbt[(gbt["평점평균"]<2.0)&(gbt["이수학기"]==2)])
golden  = p(gbt_2, gbt_low)
uz_str  = "→".join(f"{n}명" for n in uz_b)

# ── ════ 상단 요약 ════ ────────────────────────────────────────────
st.title("📊 유학생 불법체류 실무 분석 대시보드")
st.caption(f"분석 연도: {sorted(year_sel)} | 교육과정: {과정_sel} | 국적: {국적_sel} "
           f"| 불체자: 2019~2025 전수, 분모: 2023~2025")

m1,m2,m3 = st.columns(3)
m1.metric("분모 대상(2023~2025)", f"{tot_f:,}명")
m2.metric("불체 발생(전수)", f"{bu_f:,}명")
m3.metric("전체 불체율", f"{f1(r_f)}%")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

tc1,tc2,tc3,tc4 = st.columns(4)
_cs = "border-radius:10px;color:white;padding:20px 16px;min-height:150px;"
with tc1:
    st.markdown(f"""<div style="background:{CR};{_cs}">
    <div style="font-weight:800">🔴 경제적 이탈</div>
    <div style="font-size:2em;font-weight:900;margin:4px 0">{미등N}명</div>
    <div style="opacity:.9">{f1(미등P)}% · 미등록제적</div>
    <div style="font-size:.82em;margin-top:8px;opacity:.8">등록 전 이탈, 경제적 사유 추정<br>예방 가능 — 등록 전 집중 상담</div>
    </div>""", unsafe_allow_html=True)
with tc2:
    st.markdown(f"""<div style="background:{CW};{_cs}">
    <div style="font-weight:800">🟠 학업 부적응</div>
    <div style="font-size:2em;font-weight:900;margin:4px 0">{학변N}명</div>
    <div style="opacity:.9">{f1(학변P)}% · 학적변동</div>
    <div style="font-size:.82em;margin-top:8px;opacity:.8">저학점 누적 후 이탈, 몽골 집중<br>조기 개입 — 2~3학기 골든타임</div>
    </div>""", unsafe_allow_html=True)
with tc3:
    st.markdown(f"""<div style="background:{CY};color:#333;{_cs}">
    <div style="font-weight:800">🟡 정책 공백</div>
    <div style="font-size:2em;font-weight:900;margin:4px 0">{졸업N}명</div>
    <div style="opacity:.85">{f1(졸업P)}% · 졸업 후 미출국</div>
    <div style="font-size:.82em;margin-top:8px;opacity:.75">우수자 합법 체류경로 부재<br>제도 보완 — D-10 안내</div>
    </div>""", unsafe_allow_html=True)
with tc4:
    st.markdown(f"""<div style="background:#555;{_cs}">
    <div style="font-weight:800">⚫ 유령학생</div>
    <div style="font-size:2em;font-weight:900;margin:4px 0">{입학N}명</div>
    <div style="opacity:.9">{f1(입학P)}% · 입학실패</div>
    <div style="font-size:.82em;margin-top:8px;opacity:.8">입학 후 실질 수학 없이 이탈<br>입학 스크리닝 강화 필요</div>
    </div>""", unsafe_allow_html=True)

st.divider()

st.subheader("⚡ 지금 당장 주목할 3가지")
st.error(f"1️⃣ **우즈베키스탄 불체 동향** — 연도별: {uz_str} (입학년도 기준) | 추이 지속 모니터링, 입학 스크리닝 재검토 필요")
st.warning(f"2️⃣ **GPA 2.0미만 × 2학기 = 골든타임** — GPA 2.0 미만 불체자 중 **{f0(golden)}%**가 2학기에 이탈 → 2학기 등록 전 집중 개입")
st.warning(f"3️⃣ **미등록제적 {f0(미등P)}%** ({미등N}명) — 2학기 등록 전 집중 상담 체계 구축 필요")
st.divider()

# ══════════════════════════════════════════════════════════════════
# 탭
# ══════════════════════════════════════════════════════════════════
tabs = st.tabs(["🌍 탭1 국적별","📉 탭2 GPA","🎓 탭3 교육과정","📅 탭4 코호트",
                "🔀 탭5 발생경로","📚 탭6 어학급수","⚠️ 탭7 위험군",
                "🤖 탭8 예측모델","🎯 탭9 입시시뮬레이터"])
t1,t2,t3,t4,t5,t6,t7,t8,t9 = tabs

# ════════════════════════════════════════════════════════
# 탭1: 국적별
# ════════════════════════════════════════════════════════
with t1:
    st.markdown("#### 📌 국적별 위험 유형")
    nc1,nc2,nc3,nc4 = st.columns(4)
    _nc = "border-radius:8px;color:white;padding:16px;min-height:130px;"
    with nc1:
        uz_t_s = "→".join(f"{n}명" for n in uz_b)
        uz_r_s = "→".join(f"{v}%" for v in uz_r)
        st.markdown(f"""<div style="background:{CR};{_nc}">
        <b>🔴 우즈베키스탄 — 규모 확장형</b><br>
        불체: {uz_t_s}<br>불체율: {uz_r_s}<br>
        <small>입학 스크리닝 재검토 필요</small></div>""", unsafe_allow_html=True)
    with nc2:
        mn_t_s = "→".join(f"{n}명" for n in mn_b)
        mn_r_s = "→".join(f"{v}%" for v in mn_r)
        st.markdown(f"""<div style="background:{CW};{_nc}">
        <b>🟠 몽골 — 구조적 고착형</b><br>
        불체: {mn_t_s}<br>불체율: {mn_r_s}<br>
        <small>학업 부적응 집중 관리 필요</small></div>""", unsafe_allow_html=True)
    with nc3:
        vn_t_s = "→".join(f"{n}명" for n in vn_b)
        vn_r_s = "→".join(f"{v}%" for v in vn_r)
        st.markdown(f"""<div style="background:{CI};{_nc}">
        <b>🔵 베트남 — 절대 다수형</b><br>
        불체: {vn_t_s}<br>(전체의 {f0(vn_비중)}%)<br>
        <small>관리에 따라 개선 가능한 구조</small></div>""", unsafe_allow_html=True)
    with nc4:
        st.markdown(f"""<div style="background:{CS};{_nc}">
        <b>🟢 네팔·미얀마 — 안정형</b><br>
        최고 불체율: {f1(nm_max)}%<br><br>
        <small>이 국적 관리방식을 타 국적 적용 검토</small></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    nat_g  = calc_rate(df_bu_exec, df_dn_exec, "국적_그룹").sort_values("불체율(%)", ascending=True)
    avg_r  = p(bu_exec, len(df_dn_exec))

    col1,col2 = st.columns(2)
    with col1:
        colors = [CR if float(v)>=avg_r*1.5 else (CW if float(v)>=avg_r else CS)
                  for v in nat_g["불체율(%)"]]
        fig = go.Figure(go.Bar(
            x=nat_g["불체율(%)"].astype(float), y=nat_g["국적_그룹"],
            orientation="h", marker_color=colors,
            text=[f"{f1(v)}%  ({int(b)}명)" for v,b in zip(nat_g["불체율(%)"],nat_g["불체"])],
            textposition="outside", textfont_size=13))
        fig.add_vline(x=float(avg_r), line_dash="dash", line_color=CD, line_width=2,
                      annotation_text=f"평균 {f1(avg_r)}%", annotation_font_size=12)
        fig.update_layout(title="국적별 불체율 (분자:전수/분모:2023~2025)", height=400,
                          plot_bgcolor="#fafafa", paper_bgcolor="#fafafa",
                          margin=dict(r=100,t=40,b=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.line(yr_nat, x="입학년도", y="불체율(%)", color="국적_그룹",
                       markers=True, title="연도별 국적별 불체율 추이 (2019~2025 코호트)",
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=400, yaxis_tickformat=".1f")
        st.plotly_chart(fig2, use_container_width=True)

    yr_stack = df_coh.groupby(["입학년도","국적_그룹"]).size().reset_index(name="학생수")
    fig3 = px.bar(yr_stack, x="입학년도", y="학생수", color="국적_그룹",
                  barmode="stack", title="연도별 국적 재적 구성 변화 (2019~2025)",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig3.update_layout(height=360)
    st.plotly_chart(fig3, use_container_width=True)

    top3 = nat_g.nlargest(3,"불체율(%)")
    rising = []
    for nat in _VALID:
        s = yr_nat[yr_nat["국적_그룹"]==nat].sort_values("입학년도")
        r = s["불체율(%)"].values
        if len(r)>=2 and float(r[-1])>float(r[-2]):
            rising.append(nat)
    top3_txt = " | ".join([f"<b>{r['국적_그룹']}</b> {f1(r['불체율(%)'])}% (평균 {(float(r['불체율(%)'])/float(avg_r) if avg_r>0 else 0):.1f}배)"
                           for _,r in top3.iterrows()])
    rising_txt = (f"전년 대비 불체율 상승: <b>{', '.join(rising)}</b>" if rising
                  else "전년 대비 상승 국적 없음")
    kf(f"불체율 상위 3개 국적 — {top3_txt}<br>{rising_txt}")

    with st.expander("▼ 국적별 상세 테이블"):
        tbl = nat_g.sort_values("불체율(%)",ascending=False).copy()
        tbl["불체율(%)"] = tbl["불체율(%)"].apply(lambda v: f"{float(v):.1f}%")
        st.dataframe(tbl.rename(columns={"총원":"학생수","불체":"불체인원"})
                      .reset_index(drop=True), use_container_width=True)


# ════════════════════════════════════════════════════════
# 탭2: GPA × 학업성취
# ════════════════════════════════════════════════════════
with t2:
    df_bu_gpa = df_bu_f.dropna(subset=["GPA구간"])
    df_dn_gpa = df_dn_f.dropna(subset=["GPA구간"])
    gpa_g = calc_rate(df_bu_gpa, df_dn_gpa, "GPA구간")

    low_r = gpa_g[gpa_g["GPA구간"].isin(["0점","0.01~1.5","1.5~2.0"])]["불체율(%)"].mean()
    hi_r  = gpa_g[gpa_g["GPA구간"].isin(["3.0~3.5","3.5+"])]["불체율(%)"].mean()
    mult  = float(low_r)/float(hi_r) if float(hi_r)>0 else 0.0

    dual = df_dn_f[(df_dn_f["평점평균"].notna())&(df_dn_f["평점평균"]<2.0)&
                   (df_dn_f["취득학점"].notna())&(df_dn_f["취득학점"]<15)]
    kf(f"GPA 2.0 미만 불체율 평균 <span class='r'>{f1(low_r)}%</span> — "
       f"GPA 3.0 이상 대비 <b>{f1(mult)}배</b><br>"
       f"취득학점 15학점 미만 + GPA 2.0 미만 동시 해당: <b>{len(dual)}명</b> (조기 개입 대상)")

    col1,col2 = st.columns(2)
    with col1:
        bar_c = [CR if str(x) in ["0점","0.01~1.5","1.5~2.0"] else
                 (CW if str(x) in ["2.0~2.5","2.5~3.0"] else CS)
                 for x in gpa_g["GPA구간"]]
        fig = go.Figure(go.Bar(
            x=gpa_g["GPA구간"].astype(str), y=gpa_g["불체율(%)"].astype(float),
            marker_color=bar_c,
            text=[f"{f1(v)}%" for v in gpa_g["불체율(%)"]],
            textposition="outside", textfont_size=13))
        fig.add_vline(x=2.5, line_dash="dash", line_color=CR, line_width=2,
                      annotation_text="2.0 기준", annotation_font_size=12)
        fig.update_layout(title="GPA 구간별 불체율", height=420,
                          yaxis_tickformat=".1f",
                          plot_bgcolor="#fafafa", paper_bgcolor="#fafafa")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df_2d = df_dn_f.dropna(subset=["GPA구간","취득학점"]).copy()
        df_2d["학점구간"] = pd.cut(df_2d["취득학점"],
            bins=[0,10,20,30,40,500], labels=["0~10","11~20","21~30","31~40","40+"],right=True)
        df_bu_2d = df_bu_f.dropna(subset=["GPA구간","취득학점"]).copy()
        df_bu_2d["학점구간"] = pd.cut(df_bu_2d["취득학점"],
            bins=[0,10,20,30,40,500], labels=["0~10","11~20","21~30","31~40","40+"],right=True)
        ht = calc_rate(df_bu_2d, df_2d, ["GPA구간","학점구간"])
        piv = ht.pivot(index="GPA구간",columns="학점구간",values="불체율(%)").fillna(0)
        fig2 = go.Figure(go.Heatmap(z=piv.values.astype(float),
            x=piv.columns.tolist(), y=piv.index.tolist(),
            colorscale="RdYlGn_r",
            text=[[f"{float(v):.1f}%" for v in row] for row in piv.values],
            texttemplate="%{text}", showscale=True))
        fig2.update_layout(title="GPA × 취득학점 불체율 히트맵", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    # 불체 vs 정상 GPA 박스플롯
    df_box = pd.concat([
        df_bu_f[df_bu_f["평점평균"].notna()].assign(구분="불체"),
        df_dn_f[(df_dn_f["불체여부"]!="Y")&df_dn_f["평점평균"].notna()].assign(구분="정상")
    ])
    fig3 = px.box(df_box, x="구분", y="평점평균", color="구분",
                  color_discrete_map={"불체":CR,"정상":CS},
                  points="outliers", title="불체 여부별 GPA 분포 비교",
                  category_orders={"구분":["정상","불체"]})
    fig3.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("▼ GPA구간별 상세 테이블"):
        t2_tbl = gpa_g.copy()
        t2_tbl["불체율(%)"] = t2_tbl["불체율(%)"].apply(lambda v: f"{float(v):.1f}%")
        st.dataframe(t2_tbl.rename(columns={"총원":"학생수","불체":"불체인원"})
                      .reset_index(drop=True), use_container_width=True)


# ════════════════════════════════════════════════════════
# 탭3: 교육과정 × 성별 × 트랙
# ════════════════════════════════════════════════════════
with t3:
    col1,col2 = st.columns(2)
    with col1:
        bu_cs = df_bu_f[df_bu_f["성별"].isin(["남","여"])]
        dn_cs = df_dn_f[df_dn_f["성별"].isin(["남","여"])]
        if len(dn_cs) > 0:
            crs = calc_rate(bu_cs, dn_cs, ["교육과정구분","성별"])
            piv3 = crs.pivot(index="교육과정구분",columns="성별",values="불체율(%)").fillna(0)
            fig = go.Figure(go.Heatmap(
                z=piv3.values.astype(float), x=piv3.columns.tolist(), y=piv3.index.tolist(),
                colorscale="RdYlGn_r",
                text=[[f"{float(v):.1f}%" for v in row] for row in piv3.values],
                texttemplate="%{text}", showscale=True, zmin=0))
            fig.update_layout(title="교육과정 × 성별 불체율 히트맵", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("해당 조건 데이터 없음")
            crs = pd.DataFrame()

    with col2:
        bu_tr = df_bu_f[df_bu_f["트랙구분"].isin(["영어트랙","한국어트랙"])]
        dn_tr = df_dn_f[df_dn_f["트랙구분"].isin(["영어트랙","한국어트랙"])]
        if len(dn_tr) > 0:
            tr_g = calc_rate(bu_tr, dn_tr, "트랙구분")
            fig2 = px.bar(tr_g, x="트랙구분", y="불체율(%)",
                          color="불체율(%)", color_continuous_scale="RdYlGn_r",
                          text=tr_g["불체율(%)"].apply(lambda v: f"{float(v):.1f}%"),
                          title="트랙별 불체율 비교")
            fig2.update_traces(textposition="outside", textfont_size=14)
            fig2.update_layout(coloraxis_showscale=False, height=400, yaxis_tickformat=".1f")
            st.plotly_chart(fig2, use_container_width=True)

    bu_ct = df_bu_f[df_bu_f["트랙구분"].isin(["영어트랙","한국어트랙"])]
    dn_ct = df_dn_f[df_dn_f["트랙구분"].isin(["영어트랙","한국어트랙"])]
    if len(dn_ct) > 0:
        ct_g = calc_rate(bu_ct, dn_ct, ["교육과정구분","트랙구분"])
        fig3 = px.bar(ct_g, x="교육과정구분", y="불체율(%)", color="트랙구분",
                      barmode="group",
                      color_discrete_map={"한국어트랙":CI,"영어트랙":CS},
                      text=ct_g["불체율(%)"].apply(lambda v: f"{float(v):.1f}%"),
                      title="교육과정 × 트랙별 불체율")
        fig3.update_traces(textposition="outside", textfont_size=13)
        fig3.update_layout(height=380, yaxis_tickformat=".1f")
        st.plotly_chart(fig3, use_container_width=True)

    if len(crs) > 0:
        mx = crs.loc[crs["불체율(%)"].idxmax()]
        kf(f"최고위험 조합: <b>{mx['교육과정구분']} × {mx['성별']}</b> → "
           f"<span class='r'>{f1(mx['불체율(%)'])}%</span>")


# ════════════════════════════════════════════════════════
# 탭4: 입학코호트 (예외: 2019~2025 전체)
# ════════════════════════════════════════════════════════
with t4:
    coh_g = calc_rate_single(df_coh, "입학코호트").sort_values("입학코호트")
    coh_g = coh_g[coh_g["총원"] >= 5]

    if len(coh_g) > 0:
        top_c = coh_g.loc[coh_g["불체율(%)"].idxmax()]
        gbt_coh = df_coh[(df_coh["불체여부"]=="Y")&df_coh["이수학기"].notna()]
        peak_sem = int(gbt_coh["이수학기"].mode()[0]) if len(gbt_coh)>0 else 2
        kf(f"최고위험 코호트: <b>{top_c['입학코호트']}</b> → "
           f"<span class='r'>{f1(top_c['불체율(%)'])}%</span><br>"
           f"불체 집중 발생 시점: <b>입학 후 {peak_sem}학기</b> "
           f"(이수학기 기준 최빈값)")

    col1,col2 = st.columns(2)
    with col1:
        if len(coh_g) > 0:
            max_r4 = float(coh_g["불체율(%)"].max())
            dot_c = [CR if float(v)>=max_r4*0.8 else (CW if float(v)>=max_r4*0.4 else CS)
                     for v in coh_g["불체율(%)"]]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=coh_g["입학코호트"],y=coh_g["총원"].astype(int),
                name="학생수",marker_color="#B0C4DE",opacity=0.5,yaxis="y1"))
            fig.add_trace(go.Scatter(
                x=coh_g["입학코호트"],y=coh_g["불체율(%)"].astype(float),
                name="불체율(%)",mode="lines+markers+text",
                line=dict(color=CR,width=2.5),
                marker=dict(size=11,color=dot_c,line=dict(width=2,color="white")),
                text=[f"{f1(v)}%" for v in coh_g["불체율(%)"]],
                textposition="top center",textfont_size=12,yaxis="y2"))
            fig.update_layout(
                title="코호트별 불체율 추이 (2019~2025)",height=460,
                xaxis=dict(tickangle=-40),
                yaxis=dict(title="학생수",side="left"),
                yaxis2=dict(title="불체율(%)",side="right",overlaying="y",tickformat=".1f"),
                legend=dict(orientation="h",y=1.05,x=1,xanchor="right"),
                plot_bgcolor="#fafafa",paper_bgcolor="#fafafa")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        curves = []
        for yr in range(2019,2026):
            sub = df_coh[df_coh["입학년도"]==yr]
            total = len(sub)
            if total < 5: continue
            bu_sub = sub[(sub["불체여부"]=="Y") & sub["이수학기"].notna()]
            for sem in range(1,8):
                cum = int((bu_sub["이수학기"] <= sem).sum())
                curves.append({"입학년도":str(yr),"이수학기":sem,"누적불체율":float(cum)/float(total)*100})
        if curves:
            df_cv = pd.DataFrame(curves)
            fig2 = px.line(df_cv,x="이수학기",y="누적불체율",color="입학년도",
                           markers=True,title="입학 후 이수학기별 누적 불체율",
                           labels={"이수학기":"입학 후 학기","누적불체율":"누적 불체율(%)"})
            fig2.update_layout(height=460,yaxis_tickformat=".2f")
            st.plotly_chart(fig2, use_container_width=True)

    if len(coh_g) > 0:
        recent = ["2023-1학기","2023-2학기","2024-1학기","2024-2학기","2025-1학기","2025-2학기"]
        dr = coh_g[coh_g["입학코호트"].isin(recent)]
        if len(dr) > 0:
            rc = dr.loc[dr["불체율(%)"].idxmax()]
            st.info(f"⚠️ 현재 재학 중인 위험 코호트: **{rc['입학코호트']}** → 불체율 {f1(rc['불체율(%)'])}% — 집중 모니터링 필요")

    with st.expander("▼ 코호트별 상세 테이블"):
        t4_t = coh_g.nlargest(15,"불체율(%)")[["입학코호트","총원","불체","불체율(%)"]].copy()
        t4_t["불체율(%)"] = t4_t["불체율(%)"].apply(lambda v: f"{float(v):.1f}%")
        st.dataframe(t4_t.rename(columns={"총원":"학생수","불체":"불체인원"})
                      .reset_index(drop=True), use_container_width=True)


# ════════════════════════════════════════════════════════
# 탭5: 발생경로 심화
# ════════════════════════════════════════════════════════
with t5:
    df_bp = df_bu_f[df_bu_f["불체발생경로"].notna()].copy()
    pg = df_bp.groupby("불체발생경로").agg(인원=("불체_flag","count")).reset_index()
    tot_bp = int(pg["인원"].sum())

    def path_stats(path_name):
        sub = df_bp[df_bp["불체발생경로"]==path_name]
        n = len(sub)
        if n == 0: return {}
        gpa_sub = sub[sub["평점평균"].notna()]
        low_pct = p(len(gpa_sub[gpa_sub["평점평균"]<2.0]), len(gpa_sub)) if len(gpa_sub)>0 else 0.0
        hi_pct  = p(len(gpa_sub[gpa_sub["평점평균"]>=3.0]), len(gpa_sub)) if len(gpa_sub)>0 else 0.0
        sem_sub = sub[sub["이수학기"].notna()]
        sem2_pct = p(len(sem_sub[sem_sub["이수학기"]==2]), len(sem_sub)) if len(sem_sub)>0 else 0.0
        sem1_pct = p(len(sem_sub[sem_sub["이수학기"]==1]), len(sem_sub)) if len(sem_sub)>0 else 0.0
        sem23_pct= p(len(sem_sub[sem_sub["이수학기"].between(2,3)]), len(sem_sub)) if len(sem_sub)>0 else 0.0
        top_nat  = sub["국적_그룹"].value_counts()
        top_nat1 = top_nat.index[0] if len(top_nat)>0 else "N/A"
        top_nat_pct = p(int(top_nat.iloc[0]), n) if len(top_nat)>0 else 0.0
        sp_pct = p(len(sub[sub["교육과정구분"]=="전공심화"]), n)
        return dict(n=n,low_pct=low_pct,hi_pct=hi_pct,sem2_pct=sem2_pct,
                    sem1_pct=sem1_pct,sem23_pct=sem23_pct,
                    top_nat=top_nat1,top_nat_pct=top_nat_pct,sp_pct=sp_pct)

    ps = {r: path_stats(r) for r in ["미등록제적","학적변동","졸업","입학실패"]}

    kf_lines = []
    if ps["미등록제적"].get("n",0)>0:
        s = ps["미등록제적"]
        kf_lines.append(f"• <b>미등록제적</b>: 2학기 집중 {f0(s['sem2_pct'])}%, {s['top_nat']} {f0(s['top_nat_pct'])}%")
    if ps["학적변동"].get("n",0)>0:
        s = ps["학적변동"]
        kf_lines.append(f"• <b>학적변동</b>: GPA 2.0 미만 {f0(s['low_pct'])}%, 2~3학기 {f0(s['sem23_pct'])}%, {s['top_nat']} {f0(s['top_nat_pct'])}%")
    if ps["졸업"].get("n",0)>0:
        s = ps["졸업"]
        kf_lines.append(f"• <b>졸업</b>: GPA 3.0 이상 {f0(s['hi_pct'])}%, 전공심화 {f0(s['sp_pct'])}%")
    if ps["입학실패"].get("n",0)>0:
        s = ps["입학실패"]
        kf_lines.append(f"• <b>입학실패</b>: 1학기 집중 {f0(s['sem1_pct'])}%, 유령학생 유형")
    if kf_lines: kf("<br>".join(kf_lines))

    col1,col2 = st.columns([2,3])
    with col1:
        fig = px.pie(pg, names="불체발생경로", values="인원", hole=0.45,
                     title="경로별 비중",
                     color_discrete_sequence=[CR,CW,CY,CI,"#555"])
        fig.update_traces(textinfo="percent+label+value")
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df_pg = df_bp.dropna(subset=["GPA구간"]).copy()
        pg_gpa = (df_pg.groupby(["불체발생경로","GPA구간"],observed=True)
                  .size().reset_index(name="인원"))
        pg_gpa["비율(%)"] = pg_gpa["인원"] / pg_gpa.groupby("불체발생경로")["인원"].transform("sum") * 100
        fig2 = px.bar(pg_gpa, x="불체발생경로", y="비율(%)", color="GPA구간",
                      barmode="stack", title="경로 × GPA구간 구성비",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

    col3,col4 = st.columns(2)
    with col3:
        df_ps = df_bp.dropna(subset=["이수학기"]).copy()
        df_ps["이수학기_s"] = df_ps["이수학기"].astype(int).astype(str)+"학기"
        ps2 = df_ps.groupby(["불체발생경로","이수학기_s"]).size().reset_index(name="인원")
        ps2["비율(%)"] = ps2["인원"] / ps2.groupby("불체발생경로")["인원"].transform("sum") * 100
        fig3 = px.bar(ps2, x="불체발생경로", y="비율(%)", color="이수학기_s",
                      barmode="stack", title="경로 × 이수학기 구성비",
                      color_discrete_sequence=px.colors.qualitative.Pastel1)
        fig3.update_layout(height=380)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        pn = df_bp.groupby(["불체발생경로","국적_그룹"]).size().reset_index(name="인원")
        pn["비율(%)"] = pn["인원"] / pn.groupby("불체발생경로")["인원"].transform("sum") * 100
        fig4 = px.bar(pn, x="비율(%)", y="불체발생경로", color="국적_그룹",
                      barmode="stack", orientation="h",
                      title="경로 × 국적 구성비 (100%)",
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      text=pn["비율(%)"].apply(lambda v: f"{float(v):.0f}%"))
        fig4.update_traces(textposition="inside")
        fig4.update_layout(height=380)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📌 졸업 후 미출국 — 정책 공백 유형 심층 분석")
    st.info("이 유형은 학업 실패가 아닌 **정착 의향**으로 발생합니다. "
            "합법적 체류 전환 경로가 없어 결과적으로 불법체류로 분류되는 **구조적 문제**입니다.")

    df_졸 = df_bp[df_bp["불체발생경로"]=="졸업"].copy()
    z_n   = len(df_졸)
    z_gpa = df_졸[df_졸["평점평균"].notna()]
    z_hi  = p(len(z_gpa[z_gpa["평점평균"]>=3.0]), len(z_gpa)) if len(z_gpa)>0 else 0.0
    z_sp  = p(len(df_졸[df_졸["교육과정구분"]=="전공심화"]), z_n) if z_n>0 else 0.0

    zc1,zc2,zc3 = st.columns(3)
    zc1.metric("졸업경로 불체", f"{z_n}명")
    zc2.metric("GPA 3.0 이상", f"{f0(z_hi)}%")
    zc3.metric("전공심화 비율", f"{f0(z_sp)}%")

    if z_n > 0 and len(z_gpa) > 0:
        comp_d = df_bp.dropna(subset=["평점평균"]).copy()
        fig_z = px.box(comp_d, x="불체발생경로", y="평점평균", color="불체발생경로",
                       title="발생경로별 GPA 분포 비교",
                       color_discrete_sequence=[CR,CW,CY,CI,"#555"],
                       points="outliers")
        fig_z.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_z, use_container_width=True)

    st.markdown("""<div class="policy-box">
    <b>[정책 공백 진단]</b><br>
    • D-2 비자 졸업 후 취업/정주 연계 지원 부재<br>
    • 합법적 체류 전환 경로(D-10 구직비자 등) 안내 미흡<br>
    • 졸업 우수자임에도 불법체류로 분류되는 구조<br><br>
    <b>[개선 방향]</b><br>
    • 졸업 전 취업연계 프로그램 운영<br>
    • D-10(구직비자) 전환 절차 안내 강화<br>
    • 귀국 지원 프로그램과 정주 지원 병행<br>
    • 전공심화 졸업자 별도 사후관리 체계 수립
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# 탭6: 어학급수 위험도
# ════════════════════════════════════════════════════════
with t6:
    df_bu_lg = df_bu_f[df_bu_f["어학급수그룹"].notna()]
    df_dn_lg = df_dn_f[df_dn_f["어학급수그룹"].notna()]
    lang_g = calc_rate(df_bu_lg, df_dn_lg, "어학급수그룹")
    lang_g["_o"] = lang_g["어학급수그룹"].apply(lambda x: _ORD.index(x) if x in _ORD else 99)
    lang_g = lang_g.sort_values("_o").drop(columns="_o")

    r_m = lang_g[lang_g["어학급수그룹"]=="무시험"]["불체율(%)"].values
    r_h = lang_g[lang_g["어학급수그룹"]=="고급(5~7)"]["불체율(%)"].values
    mul = float(r_m[0])/float(r_h[0]) if (len(r_m)>0 and len(r_h)>0 and float(r_h[0])>0) else 0.0
    m_s = f1(r_m[0]) if len(r_m)>0 else "N/A"
    h_s = f1(r_h[0]) if len(r_h)>0 else "N/A"

    won_g = calc_rate(df_bu_f, df_dn_f, "어학원출신")
    r_wo = won_g[won_g["어학원출신"]=="어학원출신"]["불체율(%)"].values
    wo_s = f1(r_wo[0]) if len(r_wo)>0 else "N/A"

    kf(f"무시험 입학생 불체율 <span class='r'>{m_s}%</span> — 고급 대비 <b>{f1(mul)}배</b><br>"
       f"어학원 출신 불체율: <b>{wo_s}%</b> | 어학 기준 강화가 불체 예방의 가장 직접적 수단")

    col1,col2 = st.columns(2)
    with col1:
        bc6 = [CR if str(g) in ["무시험","저급(1~2)"] else
               (CW if str(g) in ["중급(3~4)","영어트랙"] else CS)
               for g in lang_g["어학급수그룹"]]
        fig = go.Figure(go.Bar(
            x=lang_g["어학급수그룹"].astype(str),
            y=lang_g["불체율(%)"].astype(float),
            marker_color=bc6,
            text=[f"{f1(v)}\n({int(t)}명)" for v,t in zip(lang_g["불체율(%)"],lang_g["총원"])],
            textposition="outside", textfont_size=13))
        fig.update_layout(title="어학급수 그룹별 불체율",
                          height=420, yaxis_tickformat=".1f",
                          plot_bgcolor="#fafafa", paper_bgcolor="#fafafa")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        won_c = [CR if str(x)=="비출신" else CS for x in won_g["어학원출신"]]
        fig2 = go.Figure(go.Bar(
            x=won_g["어학원출신"].astype(str),
            y=won_g["불체율(%)"].astype(float),
            marker_color=won_c,
            text=[f"{f1(v)}%" for v in won_g["불체율(%)"]],
            textposition="outside", textfont_size=14))
        fig2.update_layout(title="어학원 출신 여부별 불체율",
                           height=420, yaxis_tickformat=".1f",
                           plot_bgcolor="#fafafa", paper_bgcolor="#fafafa")
        st.plotly_chart(fig2, use_container_width=True)

    b3_g = calc_rate(df_bu_lg, df_dn_lg, ["어학급수그룹","국적_그룹"])
    piv6 = b3_g.pivot(index="어학급수그룹",columns="국적_그룹",values="불체율(%)").fillna(0)
    piv6 = piv6.reindex([r for r in _ORD if r in piv6.index])
    fig3 = go.Figure(go.Heatmap(
        z=piv6.values.astype(float), x=piv6.columns.tolist(), y=piv6.index.tolist(),
        colorscale="RdYlGn_r",
        text=[[f"{float(v):.1f}%" for v in row] for row in piv6.values],
        texttemplate="%{text}", showscale=True))
    fig3.update_layout(title="어학급수 × 국적 불체율 히트맵", height=380)
    st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════
# 탭7: 위험군 조기발견
# ════════════════════════════════════════════════════════
with t7:
    df_재학 = df_dn_f[df_dn_f["학적"] == "재학"].copy()

    def risk_score_row(row):
        sc = 0; fac = []
        gpa = row.get("평점평균"); sem = row.get("이수학기")
        nat = row.get("국적_그룹",""); crs = row.get("교육과정구분","")
        lg  = row.get("어학급수그룹"); trk = row.get("트랙구분")

        if pd.notna(gpa) and float(gpa) < 2.0:
            sc += 30; fac.append("GPA<2.0")
        if pd.notna(sem) and 2 <= int(sem) <= 3:
            sc += 20; fac.append(f"{int(sem)}학기")
        if nat in ["몽골","우즈베키스탄","베트남"]:
            sc += 15; fac.append(f"국적({nat})")
        if crs == "일반과정":
            sc += 10; fac.append("일반과정")
        if pd.notna(lg) and lg in ["무시험","저급(1~2)"]:
            sc += 15; fac.append(f"어학({lg})")
        if pd.notna(trk) and str(trk) == "한국어트랙":
            sc += 10; fac.append("한국어트랙")

        grade = "고위험" if sc >= 60 else ("주의" if sc >= 30 else "관심")
        return pd.Series({"위험점수": sc, "위험등급": grade,
                          "주요위험요인": ", ".join(fac) if fac else "-"})

    if len(df_재학) > 0:
        risk_df = df_재학.apply(risk_score_row, axis=1)
        df_재학["위험점수"]    = risk_df["위험점수"]
        df_재학["위험등급"]    = risk_df["위험등급"]
        df_재학["주요위험요인"] = risk_df["주요위험요인"]

        total_j = len(df_재학)
        hi_n = int((df_재학["위험등급"]=="고위험").sum())
        wa_n = int((df_재학["위험등급"]=="주의").sum())
        ca_n = int((df_재학["위험등급"]=="관심").sum())

        kf(f"재학생 {total_j:,}명 중 고위험 <span class='r'>{hi_n}명 ({f1(p(hi_n,total_j))}%)</span>, "
           f"주의 <b>{wa_n}명 ({f1(p(wa_n,total_j))}%)</b>")

        jc1,jc2,jc3,jc4 = st.columns(4)
        jc1.metric("전체 재학생", f"{total_j:,}명")
        jc2.metric("🔴 고위험", f"{hi_n}명 ({f1(p(hi_n,total_j))}%)")
        jc3.metric("🟠 주의", f"{wa_n}명 ({f1(p(wa_n,total_j))}%)")
        jc4.metric("🟡 관심", f"{ca_n}명 ({f1(p(ca_n,total_j))}%)")

        col1,col2,col3 = st.columns(3)
        with col1:
            pie_d = pd.DataFrame({"등급":["고위험","주의","관심"],"인원":[hi_n,wa_n,ca_n]})
            fig = px.pie(pie_d, names="등급", values="인원", hole=0.45,
                         title="위험등급별 분포",
                         color="등급",
                         color_discrete_map={"고위험":CR,"주의":CW,"관심":CY})
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            rn_g = df_재학.groupby(["위험등급","국적_그룹"]).size().reset_index(name="인원")
            rn_g["비율(%)"] = rn_g["인원"] / rn_g.groupby("위험등급")["인원"].transform("sum") * 100
            fig2 = px.bar(rn_g, x="위험등급", y="비율(%)", color="국적_그룹",
                          barmode="stack", title="위험등급 × 국적 구성",
                          color_discrete_sequence=px.colors.qualitative.Set2,
                          category_orders={"위험등급":["고위험","주의","관심"]})
            fig2.update_layout(height=380)
            st.plotly_chart(fig2, use_container_width=True)
        with col3:
            hi_dept = (df_재학[df_재학["위험등급"]=="고위험"]
                       .groupby("학과").size().reset_index(name="인원")
                       .nlargest(10,"인원").sort_values("인원"))
            fig3 = px.bar(hi_dept, x="인원", y="학과", orientation="h",
                          color="인원", color_continuous_scale="Reds",
                          text="인원", title="고위험 학생 학과별 분포 (상위 10)")
            fig3.update_traces(textposition="outside")
            fig3.update_layout(coloraxis_showscale=False, height=380)
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### 📋 위험군 학생 상세 목록")
        grade_flt = st.radio("위험등급 필터", ["전체","고위험만","주의 이상"], horizontal=True)
        show_d = df_재학.copy()
        if grade_flt == "고위험만":   show_d = show_d[show_d["위험등급"]=="고위험"]
        elif grade_flt == "주의 이상": show_d = show_d[show_d["위험등급"].isin(["고위험","주의"])]

        show_d = show_d.sort_values("위험점수", ascending=False)
        show_d["학번_표시"] = show_d["학번"].astype(str).apply(
            lambda x: x[:6]+"****" if len(x)>=6 else x)

        cols_show = ["학번_표시","국적_그룹","학과","교육과정구분",
                     "평점평균","이수학기","위험점수","위험등급","주요위험요인"]
        cols_ok = [c for c in cols_show if c in show_d.columns]
        tbl7 = show_d[cols_ok].reset_index(drop=True)
        tbl7.columns = [c.replace("_표시","").replace("_그룹","") for c in tbl7.columns]
        st.dataframe(tbl7, use_container_width=True, height=420)

        csv_buf = io.StringIO()
        tbl7.to_csv(csv_buf, index=False, encoding="utf-8-sig")
        st.download_button("📥 CSV 다운로드",
                           data=csv_buf.getvalue().encode("utf-8-sig"),
                           file_name="위험군_학생목록.csv", mime="text/csv")

        if hi_n > 0:
            top_nat7 = df_재학[df_재학["위험등급"]=="고위험"]["국적_그룹"].value_counts()
            top_fac7 = (df_재학[df_재학["위험등급"]=="고위험"]["주요위험요인"]
                        .str.split(", ").explode().value_counts())
            nat7_s = ", ".join([f"{k}({v}명)" for k,v in top_nat7.head(3).items()])
            fac7_s = ", ".join([f"{k}({v}건)" for k,v in top_fac7.head(3).items()])
            kf(f"고위험 {hi_n}명 주요 구성 — 국적: <b>{nat7_s}</b><br>"
               f"가장 많은 위험요인: <b>{fac7_s}</b>")
    else:
        st.info("현재 필터 조건에 해당하는 재학생 없음")


# ════════════════════════════════════════════════════════
# 탭8: 예측모델 (로지스틱 회귀)
# ════════════════════════════════════════════════════════
with t8:
    if not SKLEARN_OK:
        st.error("scikit-learn이 설치되지 않았습니다. `pip install scikit-learn`을 실행하세요.")
        st.stop()

    FEAT_CONT     = ["평점평균", "이수학기"]
    FEAT_CAT_REQ  = ["국적_그룹", "교육과정구분"]
    FEAT_CAT_OPT  = ["트랙구분", "어학급수그룹", "성별"]
    ALL_FEAT_CAT  = FEAT_CAT_REQ + FEAT_CAT_OPT

    @st.cache_data
    def run_model(df_in):
        df_m = df_in[df_in["불체여부"].isin(["Y","N"])].copy()
        df_m = df_m.dropna(subset=FEAT_CONT + FEAT_CAT_REQ)
        for col in FEAT_CAT_OPT:
            df_m[col] = df_m[col].fillna("미상")

        y = (df_m["불체여부"] == "Y").astype(int).values

        scaler = StandardScaler()
        X_cont = scaler.fit_transform(df_m[FEAT_CONT])

        dum_frames = []
        dummy_col_names = []
        for col in ALL_FEAT_CAT:
            dum = pd.get_dummies(df_m[col], prefix=col, dtype=float)
            dum_frames.append(dum)
            dummy_col_names.extend(dum.columns.tolist())

        X_cat = pd.concat(dum_frames, axis=1).values
        X = np.hstack([X_cont, X_cat])
        all_feat_names = FEAT_CONT + dummy_col_names

        model = LogisticRegression(max_iter=2000, class_weight="balanced",
                                   C=0.5, solver="lbfgs")
        model.fit(X, y)

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        metrics = {
            "AUC":   round(float(roc_auc_score(y, y_prob)), 3),
            "정확도": round(float(accuracy_score(y, y_pred)), 3),
            "정밀도": round(float(precision_score(y, y_pred, zero_division=0)), 3),
            "재현율": round(float(recall_score(y, y_pred, zero_division=0)), 3),
        }

        coefs = model.coef_[0]
        or_df = pd.DataFrame({
            "변수_raw": all_feat_names,
            "계수": coefs,
            "OddsRatio": np.exp(coefs)
        })

        def label_feat(raw):
            for col in ALL_FEAT_CAT:
                pf = col + "_"
                if raw.startswith(pf):
                    val = raw[len(pf):]
                    col_kr = {"국적_그룹":"국적","교육과정구분":"교육과정",
                               "트랙구분":"트랙","어학급수그룹":"어학","성별":"성별"}
                    return f"{col_kr.get(col,col)}: {val}"
            if raw == "평점평균": return "GPA (표준화)"
            if raw == "이수학기": return "이수학기 (표준화)"
            return raw

        or_df["변수"] = or_df["변수_raw"].apply(label_feat)
        or_df = or_df.sort_values("OddsRatio")

        gpa_idx   = FEAT_CONT.index("평점평균")
        gpa_coef  = float(coefs[gpa_idx])
        gpa_std   = float(scaler.scale_[gpa_idx])
        or_per_decrease = float(np.exp(-gpa_coef / gpa_std))

        # 재학생 위험 예측
        df_apply = df_in[df_in["학적"] == "재학"].copy()
        df_apply = df_apply.dropna(subset=FEAT_CONT + FEAT_CAT_REQ)
        for col in FEAT_CAT_OPT:
            df_apply[col] = df_apply[col].fillna("미상")

        X_cont_a = scaler.transform(df_apply[FEAT_CONT])
        dum_a_list = []
        for col in ALL_FEAT_CAT:
            dum_a = pd.get_dummies(df_apply[col], prefix=col, dtype=float)
            expected = [c for c in dummy_col_names if c.startswith(col+"_")]
            dum_a = dum_a.reindex(columns=expected, fill_value=0)
            dum_a_list.append(dum_a)
        X_cat_a  = pd.concat(dum_a_list, axis=1).values
        X_apply  = np.hstack([X_cont_a, X_cat_a])
        prob     = model.predict_proba(X_apply)[:, 1]
        score    = np.round(prob * 100, 1)

        df_apply = df_apply.copy()
        df_apply["위험확률"] = prob
        df_apply["위험점수"] = score
        df_apply["위험등급"] = pd.cut(
            score, bins=[-0.1, 39.9, 69.9, 100.1],
            labels=["관심","주의","고위험"])

        feat_coef_map = dict(zip(all_feat_names, coefs))

        def top2_factors(row):
            factors = []
            gpa = row.get("평점평균", np.nan)
            sem = row.get("이수학기", np.nan)
            if pd.notna(gpa) and float(gpa) < 2.0:
                factors.append(f"GPA {float(gpa):.1f}")
            for col in ALL_FEAT_CAT:
                val = str(row.get(col, "미상"))
                key = f"{col}_{val}"
                if feat_coef_map.get(key, 0) > 0.4:
                    col_kr = {"국적_그룹":"국적","교육과정구분":"교육과정",
                               "트랙구분":"트랙","어학급수그룹":"어학","성별":"성별"}
                    factors.append(f"{col_kr.get(col,col)}:{val}")
            if pd.notna(sem) and int(sem) in [2, 3]:
                factors.append(f"{int(sem)}학기")
            return ", ".join(factors[:2]) if factors else "-"

        df_apply["주요위험요인"] = df_apply.apply(top2_factors, axis=1)
        return model, metrics, or_df, or_per_decrease, df_apply

    with st.spinner("로지스틱 회귀 모델 학습 중..."):
        model_result = run_model(df_denom_base)

    _, m8_metrics, m8_or_df, m8_gpa_or, m8_apply = model_result

    # ── 출력1: 모델 해석 ─────────────────────────────────────────────
    st.markdown("### 📊 출력1 — 모델 해석")

    mc1,mc2,mc3,mc4 = st.columns(4)
    mc1.metric("AUC",   f"{m8_metrics['AUC']:.3f}")
    mc2.metric("정확도", f"{m8_metrics['정확도']:.3f}")
    mc3.metric("정밀도", f"{m8_metrics['정밀도']:.3f}")
    mc4.metric("재현율", f"{m8_metrics['재현율']:.3f}")

    gpa_text = (f"GPA 1점 낮아질수록 불체 위험 <b>{m8_gpa_or:.2f}배</b> — "
                f"GPA는 불체 예측의 핵심 변수입니다. "
                f"평점 관리가 불체 예방의 직접적 지표임을 모델이 확인합니다.")
    kf(gpa_text)

    top_n = min(20, len(m8_or_df))
    plot_or = m8_or_df.reindex(
        m8_or_df["OddsRatio"].apply(lambda x: abs(np.log(x))).nlargest(top_n).index
    ).sort_values("OddsRatio")

    col1,col2 = st.columns([3,2])
    with col1:
        bar_c_or = [CR if float(v)>1 else CS for v in plot_or["OddsRatio"]]
        fig_or = go.Figure(go.Bar(
            x=plot_or["OddsRatio"].astype(float),
            y=plot_or["변수"],
            orientation="h",
            marker_color=bar_c_or,
            text=[f"{float(v):.2f}" for v in plot_or["OddsRatio"]],
            textposition="outside"))
        fig_or.add_vline(x=1.0, line_dash="dash", line_color="#333", line_width=1.5,
                         annotation_text="OR=1 (중립)")
        fig_or.update_layout(
            title="변수별 Odds Ratio (상위 20개, 빨강=위험 증가)",
            height=max(400, top_n*28),
            xaxis_title="Odds Ratio",
            plot_bgcolor="#fafafa", paper_bgcolor="#fafafa",
            margin=dict(l=180,r=80,t=40,b=30))
        st.plotly_chart(fig_or, use_container_width=True)

    with col2:
        st.markdown("**모델 정보**")
        st.info(f"""
- 학습 데이터: 2023~2025 입학생 (불체여부 확정)
- 알고리즘: 로지스틱 회귀 (class_weight=balanced)
- 연속형: 평점평균, 이수학기 → StandardScaler
- 범주형: 국적, 교육과정, 트랙, 어학급수, 성별 → 더미인코딩
- 결측: 해당 변수에서만 제외
        """)
        st.markdown("**OR 해석**")
        st.markdown("""
- OR > 1 : 해당 조건에서 불체 위험 **증가**
- OR < 1 : 해당 조건에서 불체 위험 **감소**
- OR = 1 : 위험 중립
        """)
        hi_or = m8_or_df.nlargest(5,"OddsRatio")[["변수","OddsRatio"]].copy()
        hi_or["OddsRatio"] = hi_or["OddsRatio"].apply(lambda v: f"{float(v):.2f}")
        st.markdown("**위험 상위 5개 요인**")
        st.dataframe(hi_or.reset_index(drop=True), use_container_width=True)

    # ── 출력2: 재학생 위험점수 ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 출력2 — 재학생 위험점수")

    df_app = m8_apply.copy()
    t8_total = len(df_app)
    t8_hi   = int((df_app["위험등급"]=="고위험").sum())
    t8_wa   = int((df_app["위험등급"]=="주의").sum())
    t8_ca   = int((df_app["위험등급"]=="관심").sum())

    ac1,ac2,ac3,ac4 = st.columns(4)
    ac1.metric("분석 재학생", f"{t8_total:,}명")
    ac2.metric("🔴 고위험 (70+)", f"{t8_hi}명 ({f1(p(t8_hi,t8_total))}%)")
    ac3.metric("🟠 주의 (40~69)", f"{t8_wa}명 ({f1(p(t8_wa,t8_total))}%)")
    ac4.metric("🟡 관심 (~39)",   f"{t8_ca}명 ({f1(p(t8_ca,t8_total))}%)")

    bc1,bc2,bc3 = st.columns(3)
    with bc1:
        pie8 = pd.DataFrame({"등급":["고위험","주의","관심"],"인원":[t8_hi,t8_wa,t8_ca]})
        fig8a = px.pie(pie8, names="등급", values="인원", hole=0.5,
                       title="위험등급 도넛차트",
                       color="등급",
                       color_discrete_map={"고위험":CR,"주의":CW,"관심":CY})
        fig8a.update_traces(textinfo="percent+label")
        fig8a.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig8a, use_container_width=True)

    with bc2:
        rn8 = df_app.groupby(["위험등급","국적_그룹"]).size().reset_index(name="인원")
        rn8["비율(%)"] = rn8["인원"] / rn8.groupby("위험등급")["인원"].transform("sum") * 100
        fig8b = px.bar(rn8, x="위험등급", y="비율(%)", color="국적_그룹",
                       barmode="stack", title="위험등급 × 국적 누적막대",
                       color_discrete_sequence=px.colors.qualitative.Set2,
                       category_orders={"위험등급":["고위험","주의","관심"]})
        fig8b.update_layout(height=380)
        st.plotly_chart(fig8b, use_container_width=True)

    with bc3:
        hi8_dept = (df_app[df_app["위험등급"]=="고위험"]
                    .groupby("학과").size().reset_index(name="인원")
                    .nlargest(10,"인원").sort_values("인원"))
        if len(hi8_dept) > 0:
            fig8c = px.bar(hi8_dept, x="인원", y="학과", orientation="h",
                           color="인원", color_continuous_scale="Reds",
                           text="인원", title="고위험 학과 상위 10개")
            fig8c.update_traces(textposition="outside")
            fig8c.update_layout(coloraxis_showscale=False, height=380)
            st.plotly_chart(fig8c, use_container_width=True)
        else:
            st.info("고위험 학생 없음")

    st.markdown("#### 📋 재학생 위험점수 목록")
    gf1, gf2 = st.columns([2,3])
    with gf1:
        g8_flt = st.radio("등급 필터",["전체","고위험만","주의 이상"], horizontal=True)
    with gf2:
        g8_sort = st.selectbox("정렬 기준", ["위험점수 높은순","GPA 낮은순","이수학기순"])

    show8 = df_app.copy()
    if g8_flt == "고위험만":   show8 = show8[show8["위험등급"]=="고위험"]
    elif g8_flt == "주의 이상": show8 = show8[show8["위험등급"].isin(["고위험","주의"])]
    if g8_sort == "위험점수 높은순": show8 = show8.sort_values("위험점수", ascending=False)
    elif g8_sort == "GPA 낮은순":   show8 = show8.sort_values("평점평균", ascending=True)
    else:                           show8 = show8.sort_values("이수학기", ascending=True)

    show8["학번_표시"] = show8["학번"].astype(str).apply(
        lambda x: x[:-4]+"****" if len(x)>=4 else "****")

    cols8 = ["학번_표시","국적_그룹","학과","평점평균","이수학기",
             "위험점수","위험등급","주요위험요인"]
    cols8_ok = [c for c in cols8 if c in show8.columns]
    tbl8 = show8[cols8_ok].reset_index(drop=True)
    tbl8.columns = [c.replace("_표시","").replace("_그룹","") for c in tbl8.columns]
    st.dataframe(tbl8, use_container_width=True, height=440)

    csv8 = io.StringIO()
    tbl8.to_csv(csv8, index=False, encoding="utf-8-sig")
    st.download_button("📥 위험점수 CSV 다운로드",
                       data=csv8.getvalue().encode("utf-8-sig"),
                       file_name="재학생_위험점수.csv", mime="text/csv")


# ════════════════════════════════════════════════════════
# 탭9: 입시 시뮬레이터
# ════════════════════════════════════════════════════════
with t9:
    st.markdown("### 🎯 입시 시뮬레이터")
    st.markdown("""<div class="warn-box">
    ⚠️ <b>주의사항</b>: 과거 데이터 패턴 기반 추정값입니다.
    환율·취업시장·이민정책 등 외부변수는 반영되지 않습니다.
    인증기준 충족 여부는 참고용으로만 활용하세요.
    </div>""", unsafe_allow_html=True)

    # ── 과거 불체율 계산 (base rates) ───────────────────────────────
    def base_rate(grp_col, val):
        bu  = len(df_bu_all[df_bu_all[grp_col]==val])
        dn  = len(df_denom_base[df_denom_base[grp_col]==val])
        return bu/dn if dn>0 else 0.0

    nat_rates = {n: base_rate("국적_그룹", n) for n in _VALID}
    crs_rates = {c: base_rate("교육과정구분", c) for c in ["일반과정","전공심화"]}
    _lang_groups = ["무시험","저급(1~2)","중급(3~4)","고급(5~7)","영어트랙"]
    lang_rates = {l: base_rate("어학급수그룹", l) for l in _lang_groups}

    overall_rate = len(df_bu_all) / len(df_denom_base) if len(df_denom_base)>0 else 0.0

    # 현재 어학 분포 (2023~2025 기준)
    lang_dist_cur = (df_denom_base["어학급수그룹"]
                     .value_counts(normalize=True)
                     .reindex(_lang_groups, fill_value=0))
    cur_lang_rate = sum(float(lang_dist_cur.get(l,0)) * lang_rates.get(l,0) for l in _lang_groups)

    def calc_lang_adj(policy):
        if policy == "현행유지":
            return 1.0
        keep_map = {
            "TOPIK2급이상": [l for l in _lang_groups if l not in ["무시험"]],
            "TOPIK3급이상": ["중급(3~4)","고급(5~7)","영어트랙"],
            "무시험폐지":   [l for l in _lang_groups if l != "무시험"],
        }
        keep = keep_map.get(policy, _lang_groups)
        sub = lang_dist_cur.reindex(keep, fill_value=0)
        total_sub = float(sub.sum())
        if total_sub <= 0: return 1.0
        new_dist = sub / total_sub
        new_rate = sum(float(new_dist.get(l,0)) * lang_rates.get(l,0) for l in keep)
        return new_rate / cur_lang_rate if cur_lang_rate > 0 else 1.0

    # 현재 교육과정 분포
    cur_gen_pct = float((df_denom_base["교육과정구분"]=="일반과정").mean())
    cur_crs_rate = (cur_gen_pct * crs_rates.get("일반과정",0) +
                   (1-cur_gen_pct) * crs_rates.get("전공심화",0))

    def calc_crs_adj(gen_pct):
        new_rate = gen_pct * crs_rates.get("일반과정",0) + (1-gen_pct) * crs_rates.get("전공심화",0)
        return new_rate / cur_crs_rate if cur_crs_rate > 0 else 1.0

    # 현재 국적 분포 (과거 3년 평균)
    cur_nat_dist = (df_denom_base["국적_그룹"]
                    .value_counts(normalize=True)
                    .reindex(_VALID, fill_value=0))
    cur_nat_dist_pct = {n: float(cur_nat_dist.get(n,0))*100 for n in _VALID}

    # ── 입력 패널 ────────────────────────────────────────────────────
    st.markdown("#### ⚙️ 입력 설정")
    sc1, sc2 = st.columns([1,2])

    with sc1:
        total_recruit = st.number_input("총 모집 예정 인원", min_value=100, max_value=10000,
                                        value=2800, step=100)
        lang_policy = st.radio("어학기준",
            ["현행유지","TOPIK2급이상","TOPIK3급이상","무시험폐지"])
        gen_pct_slider = st.slider("일반과정 비중 (%)",
                                   min_value=0, max_value=100,
                                   value=int(cur_gen_pct*100), step=5) / 100.0

    with sc2:
        st.markdown("**국적별 비중 설정 (%)**")
        st.caption(f"현재 실적 기준 — 베트남 {cur_nat_dist_pct['베트남']:.1f}% / "
                   f"우즈베키스탄 {cur_nat_dist_pct['우즈베키스탄']:.1f}% / "
                   f"네팔 {cur_nat_dist_pct['네팔']:.1f}% / "
                   f"몽골 {cur_nat_dist_pct['몽골']:.1f}% / "
                   f"미얀마 {cur_nat_dist_pct['미얀마']:.1f}% / "
                   f"기타 {cur_nat_dist_pct['기타']:.1f}%")

        ns1,ns2 = st.columns(2)
        with ns1:
            vn_s  = st.slider("베트남 (%)",      0, 100, int(cur_nat_dist_pct["베트남"]+0.5),  step=1)
            uz_s  = st.slider("우즈베키스탄 (%)", 0, 100, int(cur_nat_dist_pct["우즈베키스탄"]+0.5), step=1)
            mn_s  = st.slider("몽골 (%)",        0, 100, int(cur_nat_dist_pct["몽골"]+0.5),    step=1)
        with ns2:
            np_s  = st.slider("네팔 (%)",        0, 100, int(cur_nat_dist_pct["네팔"]+0.5),    step=1)
            mm_s  = st.slider("미얀마 (%)",      0, 100, int(cur_nat_dist_pct["미얀마"]+0.5),  step=1)
            etc_s = st.slider("기타 (%)",        0, 100, int(cur_nat_dist_pct["기타"]+0.5),    step=1)

        slider_sum = vn_s + uz_s + mn_s + np_s + mm_s + etc_s
        if slider_sum > 0:
            nat_sim = {
                "베트남":      vn_s/slider_sum,
                "우즈베키스탄": uz_s/slider_sum,
                "몽골":        mn_s/slider_sum,
                "네팔":        np_s/slider_sum,
                "미얀마":      mm_s/slider_sum,
                "기타":        etc_s/slider_sum,
            }
        else:
            nat_sim = {n: 1/len(_VALID) for n in _VALID}

        adj_sum = sum(nat_sim.values())
        st.caption(f"합계 자동조정됨 → 총합 100% (입력: {slider_sum}%)")

    # ── 시나리오 계산 ────────────────────────────────────────────────
    def simulate(total, nat_pct_dict, lang_pol, gen_p):
        nat_weighted = sum(nat_pct_dict.get(n,0) * nat_rates.get(n,0) for n in _VALID)
        lang_adj = calc_lang_adj(lang_pol)
        crs_adj  = calc_crs_adj(gen_p)
        est_rate = nat_weighted * lang_adj * crs_adj
        est_bu   = total * est_rate
        return est_rate * 100, est_bu

    # 현재 시나리오
    cur_nat_pct = {n: float(cur_nat_dist.get(n,0)) for n in _VALID}
    cur_rate_pct, cur_bu = simulate(total_recruit, cur_nat_pct, "현행유지", cur_gen_pct)

    # 시뮬레이션 A (사용자 설정)
    sim_rate_pct, sim_bu = simulate(total_recruit, nat_sim, lang_policy, gen_pct_slider)

    # 최적화 시나리오 B (TOPIK3+일반과정 비중 고정)
    opt_rate_pct, opt_bu = simulate(total_recruit, nat_sim, "TOPIK3급이상", gen_pct_slider)

    CERT_THRESHOLD = 5.0  # 인증기준 5%

    # ── 출력 ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 시뮬레이션 결과")

    rc1,rc2,rc3 = st.columns(3)
    with rc1:
        st.markdown(f"""<div style="background:{CI};color:white;border-radius:10px;padding:20px;text-align:center">
        <div style="font-size:.9em;opacity:.85">현재 패턴 유지</div>
        <div style="font-size:2.2em;font-weight:900">{cur_rate_pct:.2f}%</div>
        <div>예상 불체 {cur_bu:.0f}명</div>
        <div style="margin-top:8px;font-size:.85em">
        {'✅ 인증기준 충족' if cur_rate_pct<=CERT_THRESHOLD else '❌ 인증기준 초과'}</div>
        </div>""", unsafe_allow_html=True)
    with rc2:
        col_bg = CS if sim_rate_pct <= CERT_THRESHOLD else CR
        st.markdown(f"""<div style="background:{col_bg};color:white;border-radius:10px;padding:20px;text-align:center">
        <div style="font-size:.9em;opacity:.85">시나리오 A (사용자 설정)</div>
        <div style="font-size:2.2em;font-weight:900">{sim_rate_pct:.2f}%</div>
        <div>예상 불체 {sim_bu:.0f}명</div>
        <div style="margin-top:8px;font-size:.85em">
        {'✅ 인증기준 충족' if sim_rate_pct<=CERT_THRESHOLD else '❌ 인증기준 초과'}</div>
        </div>""", unsafe_allow_html=True)
    with rc3:
        opt_col = CS if opt_rate_pct <= CERT_THRESHOLD else CW
        st.markdown(f"""<div style="background:{opt_col};color:white;border-radius:10px;padding:20px;text-align:center">
        <div style="font-size:.9em;opacity:.85">시나리오 B (TOPIK3급+)</div>
        <div style="font-size:2.2em;font-weight:900">{opt_rate_pct:.2f}%</div>
        <div>예상 불체 {opt_bu:.0f}명</div>
        <div style="margin-top:8px;font-size:.85em">
        {'✅ 인증기준 충족' if opt_rate_pct<=CERT_THRESHOLD else '❌ 인증기준 초과'}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # 시나리오 비교 테이블
    cmp_data = {
        "항목": ["총 모집인원","예상 불체율(%)","예상 불체인원","인증기준(5%) 충족",
                 "어학기준","일반과정 비중(%)"],
        "현재": [f"{total_recruit:,}명", f"{cur_rate_pct:.2f}%", f"{cur_bu:.0f}명",
                  "✅" if cur_rate_pct<=5 else "❌", "현행유지", f"{cur_gen_pct*100:.0f}%"],
        "시나리오 A": [f"{total_recruit:,}명", f"{sim_rate_pct:.2f}%", f"{sim_bu:.0f}명",
                       "✅" if sim_rate_pct<=5 else "❌", lang_policy, f"{gen_pct_slider*100:.0f}%"],
        "시나리오 B": [f"{total_recruit:,}명", f"{opt_rate_pct:.2f}%", f"{opt_bu:.0f}명",
                       "✅" if opt_rate_pct<=5 else "❌", "TOPIK3급이상", f"{gen_pct_slider*100:.0f}%"],
    }
    st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

    # 국적별 예상 불체 분포 차트
    nat_sim_data = []
    for nat in _VALID:
        prop_a = nat_sim.get(nat, 0)
        prop_c = float(cur_nat_dist.get(nat, 0))
        bu_a = total_recruit * prop_a * nat_rates.get(nat, 0)
        bu_c = total_recruit * prop_c * nat_rates.get(nat, 0)
        nat_sim_data.append({
            "국적": nat,
            "현재 예상불체": round(bu_c,1),
            "시뮬레이션A 예상불체": round(bu_a,1),
            "불체율(%)": round(nat_rates.get(nat,0)*100,2),
        })
    df_sim_nat = pd.DataFrame(nat_sim_data)

    col1, col2 = st.columns(2)
    with col1:
        fig9a = px.bar(df_sim_nat.melt(id_vars="국적",
                                        value_vars=["현재 예상불체","시뮬레이션A 예상불체"],
                                        var_name="시나리오", value_name="예상불체인원"),
                       x="국적", y="예상불체인원", color="시나리오", barmode="group",
                       title="국적별 예상 불체인원 비교",
                       color_discrete_map={"현재 예상불체":CI,"시뮬레이션A 예상불체":CS})
        fig9a.update_layout(height=380)
        st.plotly_chart(fig9a, use_container_width=True)

    with col2:
        fig9b = px.bar(df_sim_nat.sort_values("불체율(%)"),
                       x="불체율(%)", y="국적", orientation="h",
                       text=df_sim_nat.sort_values("불체율(%)")["불체율(%)"].apply(lambda v: f"{v:.2f}%"),
                       title="국적별 과거 불체율 (시뮬레이션 기반)",
                       color="불체율(%)", color_continuous_scale="RdYlGn_r")
        fig9b.update_traces(textposition="outside")
        fig9b.update_layout(coloraxis_showscale=False, height=380)
        st.plotly_chart(fig9b, use_container_width=True)

    kf(f"시뮬레이션A 예상 불체율 <b>{sim_rate_pct:.2f}%</b> / "
       f"현재 대비 {'<span class=\"r\">+' + f'{sim_rate_pct-cur_rate_pct:.2f}%' + ' 증가' if sim_rate_pct>cur_rate_pct else '<span class=\"g\">' + f'{cur_rate_pct-sim_rate_pct:.2f}%' + ' 감소'}</span> / "
       f"TOPIK3급 강화 시 {opt_rate_pct:.2f}% 예상")

# ── 하단 ───────────────────────────────────────────────────────────
st.divider()
st.caption(f"데이터: {FILE} | 불체자 분자: 2019~2025 전수, 분모: 2023~2025 입학생 | "
           f"코호트 탭은 2019~2025 전체 사용 | "
           f"2025년 입학생은 재학 중으로 불체율 과소 추정 가능 | "
           f"위험점수 및 시뮬레이션 결과는 내부 관리 기준으로 외부 공개 금지")
