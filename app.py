# ── RegNavigator Lite – full production build ────────────────────────
# Docket → per-comment summaries, sentiment KPI, one-paragraph brief,
# key-phrase chart, CSV + PDF exports.  No docket abstract—title only.

import asyncio, html, io, json, os, re, traceback
from collections import Counter
from datetime import datetime
from enum import Enum
from pathlib import Path

import httpx, pandas as pd, pdfplumber, plotly.express as px, streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from pydantic import BaseModel, Field, ValidationError
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from tenacity import retry, stop_after_attempt, wait_exponential

# ──────────────────────────────────────────────
# 1.  CONFIG & KEYS
# ──────────────────────────────────────────────
load_dotenv()
REGS_API_KEY  = os.getenv("REGS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not REGS_API_KEY or not GEMINI_API_KEY:
    st.error("Missing API keys in .env"); st.stop()

genai.configure(api_key=GEMINI_API_KEY)

REGS_BASE     = "https://api.regulations.gov/v4"
HTTP_TIMEOUT  = 30.0
API_DELAY     = 0.4     # seconds between requests
PDF_SEMAPHORE = asyncio.Semaphore(4)

st.set_page_config(page_title="RegNavigator Lite", page_icon="⚖️",
                   layout="wide")

# ──────────────────────────────────────────────
# 2.  DATA SCHEMA
# ──────────────────────────────────────────────
class Stance(str, Enum):
    SUPPORT = "SUPPORT"; OPPOSE = "OPPOSE"; NEUTRAL = "NEUTRAL"

class CommentAnalysis(BaseModel):
    organization: str
    summary:      str
    stance:       Stance
    keyphrases:   list[str]

# ──────────────────────────────────────────────
# 3.  HTTP HELPERS
# ──────────────────────────────────────────────
@retry(wait=wait_exponential(1, 2, 10), stop=stop_after_attempt(4))
async def _api_get(endpoint: str, params: dict | None = None) -> dict:
    async with httpx.AsyncClient(base_url=REGS_BASE,
                                 headers={"X-Api-Key": REGS_API_KEY},
                                 timeout=HTTP_TIMEOUT) as c:
        await asyncio.sleep(API_DELAY)
        r = await c.get(endpoint, params=params or {}); r.raise_for_status()
        return r.json()

async def _download_pdf(url: str) -> bytes | None:
    async with PDF_SEMAPHORE, httpx.AsyncClient(timeout=60) as c:
        try:
            r = await c.get(url); r.raise_for_status(); return r.content
        except Exception: return None

# ──────────────────────────────────────────────
# 4.  TEXT EXTRACTION
# ──────────────────────────────────────────────
async def extract_pdf_text(url: str) -> str | None:
    data = await _download_pdf(url)
    if not data: return None
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "".join(p.extract_text() or "" for p in pdf.pages)
    except Exception: return None

def clean_inline(txt: str | None) -> str | None:
    if not txt or re.match(r"^\s*(see|please\s+see)\s+attached", txt, re.I):
        return None
    return html.unescape(txt.strip())

# ──────────────────────────────────────────────
# 5.  GEMINI UTILITIES
# ──────────────────────────────────────────────
SAFE_NONE = {c: HarmBlockThreshold.BLOCK_NONE for c in [
    HarmCategory.HARM_CATEGORY_HARASSMENT,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
]}
MODEL = genai.GenerativeModel("gemini-1.5-flash-latest")

async def gemini_structured(prompt: str, schema) -> dict | None:
    try:
        r = await MODEL.generate_content_async(
            prompt,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema),
            safety_settings=SAFE_NONE)
        return json.loads(r.text)
    except Exception: return None

async def gemini_brief(prompt: str) -> str | None:
    try:
        r = await MODEL.generate_content_async(prompt,
              generation_config=GenerationConfig(max_output_tokens=120),
              safety_settings=SAFE_NONE)
        return r.text.strip().replace("\n", " ")
    except Exception: return None

# ──────────────────────────────────────────────
# 6.  COMMENT PIPELINE
# ──────────────────────────────────────────────
async def process_comment(cid: str) -> dict | None:
    try:
        det = await _api_get(f"/comments/{cid}", {"include": "attachments"})
        attr = det["data"]["attributes"]
        attachments = det.get("included", [])

        text = None
        for att in attachments:
            for f in att.get("attributes", {}).get("fileFormats", []):
                if f.get("format","").lower() == "pdf" and f.get("fileUrl"):
                    text = await extract_pdf_text(f["fileUrl"]); break
            if text: break
        if not text: text = clean_inline(attr.get("comment"))
        if not text: return None

        payload = await gemini_structured(
            "Extract JSON: organization, ≤30-word summary, "
            "stance (SUPPORT/OPPOSE/NEUTRAL), 2-5 keyphrases.\n###\n"
            + text[:25000], CommentAnalysis)
        if not payload: return None
        data = CommentAnalysis(**payload).model_dump()
        return {"id": cid, "title": attr.get("title") or "N/A", **data}
    except Exception:
        st.session_state.errors.append(f"⚠️ Comment {cid} failed")
        return None

# ──────────────────────────────────────────────
# 7.  EXECUTIVE BRIEF
# ──────────────────────────────────────────────
async def build_brief(summaries: list[str], kpis: dict) -> str:
    sentiment_word = ("supportive" if kpis["sentiment"] > 0.25 else
                      "opposed"     if kpis["sentiment"] < -0.25 else
                      "mixed")
    base_prompt = (
      f"There are {kpis['total']} public comments, overall sentiment is "
      f"{sentiment_word}. In ≤100 words summarise the key reasons people "
      f"support or oppose the proposal, mention any unique outlier view, "
      f"and avoid bullets."
      "\n===\n" + "\n".join(f"- {s}" for s in summaries[:200])
    )
    brief = await gemini_brief(base_prompt)
    return brief or "(Brief unavailable)"

# ──────────────────────────────────────────────
# 8.  PDF REPORT
# ──────────────────────────────────────────────
def build_pdf(df: pd.DataFrame, k: dict, docket_id: str,
              title: str, brief: str) -> bytes:
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter; m = 50; y = h - m
    c.setFont("Helvetica-Bold", 16); c.drawString(m, y, f"Docket {docket_id}"); y -= 20
    c.setFont("Helvetica", 12); c.drawString(m, y, title[:95]); y -= 18
    c.setFont("Helvetica-Oblique", 10); c.drawString(m, y, brief[:110]); y -= 24
    c.setFont("Helvetica-Bold", 11)
    c.drawString(m, y, f"Comments {k['total']}   |   Stakeholders {k['unique_orgs']}   "
                       f"|   S/O/N {k['support']}/{k['oppose']}/{k['neutral']}   "
                       f"|   Sentiment {k['sentiment']}"); y -= 16
    c.line(m, y, w - m, y); y -= 12
    c.setFont("Helvetica-Bold", 10)
    c.drawString(m, y, "Organization"); c.drawString(m+200, y, "Stance")
    c.drawString(m+260, y, "Summary"); y -= 12
    c.setFont("Helvetica", 9)
    for _, r in df.head(25).iterrows():
        if y < m + 40: c.showPage(); y = h - m; c.setFont("Helvetica", 9)
        c.drawString(m, y, r["organization"][:34])
        c.drawString(m+200, y, r["stance"])
        c.drawString(m+260, y, r["summary"][:80]); y -= 11
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(m, m-10, f"© {datetime.now():%Y-%m-%d %H:%M}")
    c.save(); buf.seek(0); return buf.getvalue()

# ──────────────────────────────────────────────
# 9.  MASTER PIPELINE
# ──────────────────────────────────────────────
async def analyse(docket_id: str, max_comments: int):
    st.session_state.errors = []
    docs = await _api_get("/documents",
        {"filter[docketId]": docket_id, "page[size]": 250})
    if not docs["data"]: st.error("No documents found"); return
    primary = next((d for d in docs["data"]
        if d["attributes"]["documentType"] == "Proposed Rule"), docs["data"][0])
    st.session_state.title = primary["attributes"]["title"]
    obj_id = primary["attributes"]["objectId"]

    comments = await _api_get("/comments",
        {"filter[commentOnId]": obj_id, "page[size]": max_comments})
    ids = [c["id"] for c in comments["data"]]
    if not ids: st.error("No comments found"); return

    results = [r for r in await asyncio.gather(
        *[process_comment(i) for i in ids]) if r]
    if not results: st.error("No comment text available"); return

    df = pd.DataFrame(results)
    sent_map = {'SUPPORT': 1, 'NEUTRAL': 0, 'OPPOSE': -1}
    sentiment = round(df['stance'].map(sent_map).mean(), 2)
    k = {"total": len(df), "unique_orgs": df['organization'].nunique(),
         "support": (df['stance']=="SUPPORT").sum(),
         "oppose":  (df['stance']=="OPPOSE").sum(),
         "neutral": (df['stance']=="NEUTRAL").sum(),
         "sentiment": sentiment}
    brief = await build_brief(df['summary'].tolist(), k)

    st.session_state.update(dict(df=df, kpis=k, brief=brief,
        pdf=build_pdf(df, k, docket_id, st.session_state.title, brief)))

# ──────────────────────────────────────────────
# 10.  UI HELPERS
# ──────────────────────────────────────────────
def css(path="style.css"):
    if Path(path).exists():
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def kpi_cards(k):
    a,b,c,d = st.columns(4)
    a.metric("Comments", k['total'])
    b.metric("Stakeholders", k['unique_orgs'])
    c.metric("S/O/N", f"{k['support']}/{k['oppose']}/{k['neutral']}")
    d.metric("Sentiment (-1→1)", k['sentiment'])

def phrase_chart(df):
    phrases=[p.strip().title() for sub in df['keyphrases']
             if isinstance(sub,list) for p in sub]
    if not phrases: st.info("No keyphrases"); return
    top=pd.DataFrame(Counter(phrases).most_common(15),
                     columns=["Keyphrase","Count"]).sort_values("Count")
    fig=px.bar(top,x="Count",y="Keyphrase",orientation='h',
               template="plotly_dark",color_discrete_sequence=["#4A90E2"])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig,use_container_width=True)

# ──────────────────────────────────────────────
# 11.  FRONT-END
# ──────────────────────────────────────────────
def main():
    css()
    with st.sidebar:
        st.title("⚖️ RegNavigator Lite")
        st.write("AI digest of regulatory dockets.")
        st.text("Ex: IRS-2022-0029")

    st.header("AI briefings for federal dockets")
    with st.form("f"):
        docket=st.text_input("Docket ID","IRS-2022-0029",
                             label_visibility="collapsed")
        n=st.slider("Comments to analyse",5,250,50)
        go=st.form_submit_button("Analyze",type="primary")

    if go and docket:
        with st.spinner("Crunching…"):
            asyncio.run(analyse(docket.strip(), n))

    if 'df' in st.session_state:
        st.subheader(st.session_state.title)
        st.markdown(f"*Executive takeaway:* {st.session_state.brief}")
        kpi_cards(st.session_state.kpis)
        c1,c2 = st.columns([3,1])
        with c1: phrase_chart(st.session_state.df)
        with c2:
            st.download_button("⬇️ CSV",
                st.session_state.df.to_csv(index=False),
                file_name=f"{docket}_analysis.csv")
            st.download_button("⬇️ PDF brief", st.session_state.pdf,
                file_name=f"{docket}_brief.pdf", mime="application/pdf")
        with st.expander("Raw data"):
            st.dataframe(st.session_state.df, use_container_width=True)

    if errs := st.session_state.get("errors"):
        with st.expander(f"Errors ({len(errs)})"):
            st.code("\n".join(errs))

if __name__ == "__main__":
    main()