#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RegNavigator Lite – AI briefing generator for US federal dockets
"""

# ──────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────
import asyncio, html, io, json, os, re, textwrap
from collections import Counter
from datetime import datetime
from enum import Enum
from pathlib import Path

import httpx
import pandas as pd
import pdfplumber
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import (
    GenerationConfig,
    HarmCategory,
    HarmBlockThreshold,
)
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from tenacity import retry, wait_exponential, stop_after_attempt

# ──────────────────────────────────────────────
# 0. Config & keys
# ──────────────────────────────────────────────
load_dotenv()
REGS_API_KEY = os.getenv("REGS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not REGS_API_KEY or not GEMINI_API_KEY:
    st.error("❌  Missing REGS_API_KEY or GEMINI_API_KEY in .env")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

REGS_BASE = "https://api.regulations.gov/v4"
HTTP_TIMEOUT = 30.0
API_DELAY = 0.6  # polite throttle

# Concurrency gates (keep <50 req/min per service)
REGS_SEM = asyncio.Semaphore(10)
GEMINI_SEM = asyncio.Semaphore(10)
PDF_SEM = asyncio.Semaphore(4)

st.set_page_config("RegNavigator Lite", "⚖️", layout="wide")

# ──────────────────────────────────────────────
# 1. Data schema
# ──────────────────────────────────────────────
class Stance(str, Enum):
    SUPPORT = "SUPPORT"
    OPPOSE = "OPPOSE"
    NEUTRAL = "NEUTRAL"


class CommentAnalysis(BaseModel):
    organization: str
    summary: str
    stance: Stance
    keyphrases: list[str]


# ──────────────────────────────────────────────
# 2. HTTP utilities
# ──────────────────────────────────────────────
@retry(wait=wait_exponential(1, 2, 10), stop=stop_after_attempt(4))
async def _api_get(endpoint: str, params: dict | None = None) -> dict:
    async with REGS_SEM:
        async with httpx.AsyncClient(
            base_url=REGS_BASE,
            headers={"X-Api-Key": REGS_API_KEY},
            timeout=HTTP_TIMEOUT,
        ) as c:
            await asyncio.sleep(API_DELAY)
            r = await c.get(endpoint, params=params or {})
            r.raise_for_status()
            return r.json()


async def _download_pdf(url: str) -> bytes | None:
    async with PDF_SEM, httpx.AsyncClient(timeout=60) as c:
        try:
            r = await c.get(url)
            r.raise_for_status()
            return r.content
        except Exception:
            return None


# ──────────────────────────────────────────────
# 3. Parse helpers
# ──────────────────────────────────────────────
async def extract_pdf_text(url: str) -> str | None:
    data = await _download_pdf(url)
    if not data:
        return None
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        return None


def clean_inline(txt: str | None) -> str | None:
    """Skip boilerplate like 'See attached'."""
    if not txt or re.match(r"^\s*(see|please\s+see)\s+attached", txt, re.I):
        return None
    return html.unescape(txt.strip())


# ──────────────────────────────────────────────
# 4. Gemini wrappers
# ──────────────────────────────────────────────
SAFE_NONE = {
    c: HarmBlockThreshold.BLOCK_NONE
    for c in [
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    ]
}
MODEL = genai.GenerativeModel("gemini-1.5-flash-latest")


async def gemini_json(prompt: str, schema) -> dict | None:
    async with GEMINI_SEM:
        try:
            r = await MODEL.generate_content_async(
                prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
                safety_settings=SAFE_NONE,
            )
            return json.loads(r.text)
        except Exception:
            return None


async def gemini_bullets(prompt: str) -> str | None:
    async with GEMINI_SEM:
        try:
            r = await MODEL.generate_content_async(
                prompt,
                generation_config=GenerationConfig(max_output_tokens=120),
                safety_settings=SAFE_NONE,
            )
            return r.text.strip()
        except Exception:
            return None


# ──────────────────────────────────────────────
# 5. Comment pipeline
# ──────────────────────────────────────────────
async def process_comment(cid: str) -> dict | None:
    try:
        det = await _api_get("/comments/" + cid, {"include": "attachments"})
        attr = det["data"]["attributes"]
        attachments = det.get("included", [])

        # Prefer PDF attachment
        text = None
        for att in attachments:
            for f in att.get("attributes", {}).get("fileFormats", []):
                if f.get("format", "").lower() == "pdf" and f.get("fileUrl"):
                    text = await extract_pdf_text(f["fileUrl"])
                    break
            if text:
                break

        if not text:
            text = clean_inline(attr.get("comment"))
        if not text:
            return None

        payload = await gemini_json(
            "Extract JSON: organization, ≤30-word summary, stance "
            "(SUPPORT/OPPOSE/NEUTRAL), 2-5 keyphrases.\n###\n"
            + text[:25000],
            CommentAnalysis,
        )
        if not payload:
            return None

        return {
            "id": cid,
            "title": attr.get("title") or "N/A",
            **CommentAnalysis(**payload).model_dump(),
        }
    except Exception:
        st.session_state.errors.append(f"⚠️ {cid} failed")
        return None


# ──────────────────────────────────────────────
# 6. Executive bullet brief
# ──────────────────────────────────────────────
async def build_brief(summaries: list[str], k: dict) -> str:
    mood = (
        "supportive"
        if k["sentiment"] > 0.25
        else "opposed"
        if k["sentiment"] < -0.25
        else "mixed"
    )
    prompt = textwrap.dedent(
        f"""
        Produce 3-5 Markdown bullet points (≤ 20 words each) summarising
        overall {mood} sentiment, top reasons, and any notable outlier.
        ---
        {" ".join(summaries[:200])}
        """
    ).strip()
    bullets = await gemini_bullets(prompt)
    return bullets or "- Brief unavailable"


# ──────────────────────────────────────────────
# 7. PDF export (vertical record layout)
# ──────────────────────────────────────────────
def build_pdf(
    df: pd.DataFrame,
    k: dict,
    docket: str,
    title: str,
    brief_md: str,
) -> bytes:
    """
    Generate a PDF where each comment is printed vertically, e.g.:

        Organization: Intel Corporation
        Stance: SUPPORT
        Summary: …

    Text is wrapped so nothing is cut off on the right edge.
    """
    bullets = [
        ln.lstrip("-• ").strip()
        for ln in brief_md.splitlines()
        if ln.strip().startswith(("-", "•"))
    ]

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    m = 50
    y = h - m

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(m, y, f"Docket {docket}")
    y -= 20

    c.setFont("Helvetica", 12)
    c.drawString(m, y, title[:95])
    y -= 16

    # Bullet brief
    if bullets:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(m, y, "Key takeaways:")
        y -= 14
        c.setFont("Helvetica-Oblique", 9)
        for ln in bullets:
            c.drawString(m + 10, y, f"• {ln[:110]}")
            y -= 12
        y -= 4

    # KPI line
    c.setFont("Helvetica-Bold", 11)
    c.drawString(
        m,
        y,
        f"Comments {k['total']} | Orgs {k['unique_orgs']} | "
        f"S/O/N {k['support']}/{k['oppose']}/{k['neutral']} | "
        f"Sentiment {k['sentiment']}",
    )
    y -= 18
    c.line(m, y, w - m, y)
    y -= 14

    # Records
    c.setFont("Helvetica", 9)
    for _, row in df.iterrows():
        record_lines = [
            f"Organization: {row['organization']}",
            f"Stance: {row['stance']}",
        ]
        wrapped = textwrap.wrap(str(row["summary"]), width=95)
        if wrapped:
            record_lines.append(f"Summary: {wrapped[0]}")
            record_lines.extend([" " * 9 + ln for ln in wrapped[1:]])
        else:
            record_lines.append("Summary: ")

        for ln in record_lines:
            if y < m + 40:  # new page
                c.showPage()
                y = h - m
                c.setFont("Helvetica", 9)
            c.drawString(m, y, ln)
            y -= 12

        y -= 6  # spacing

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(m, m - 10, f"© {datetime.now():%Y-%m-%d %H:%M}")
    c.save()
    buf.seek(0)
    return buf.getvalue()


# ──────────────────────────────────────────────
# 8. Master pipeline (over-fetch + progress bar)
# ──────────────────────────────────────────────
async def analyse(docket_id: str, target: int, prog, header_ph):
    st.session_state.errors = []
    docs = await _api_get("/documents", {"filter[docketId]": docket_id, "page[size]": 250})
    if not docs["data"]:
        st.error("No documents found")
        return

    primary = next(
        (d for d in docs["data"] if d["attributes"]["documentType"] == "Proposed Rule"),
        docs["data"][0],
    )
    title = primary["attributes"]["title"]
    header_ph.subheader(title)  # show title right away
    st.session_state.title = title
    obj_id = primary["attributes"]["objectId"]

    # Over-fetch IDs
    ids = []
    page_params = {
        "filter[commentOnId]": obj_id,
        "page[size]": min(250, target * 2),
        "page[number]": 1,
    }
    while len(ids) < target:
        batch = await _api_get("/comments", page_params)
        ids.extend(c["id"] for c in batch["data"])
        if not batch["meta"]["hasNextPage"]:
            break
        page_params["page[number]"] += 1
    ids = ids[: min(250, target * 2)]

    results = []
    n = len(ids)
    for i, cid in enumerate(ids, 1):
        if len(results) >= target:
            break
        row = await process_comment(cid)
        if row:
            results.append(row)
        prog.progress(i / n)

    if len(results) < target:
        st.warning(f"Only {len(results)} comments had usable text.")

    df = pd.DataFrame(results[:target])
    sent_map = {"SUPPORT": 1, "NEUTRAL": 0, "OPPOSE": -1}
    k = {
        "total": len(df),
        "unique_orgs": df["organization"].nunique(),
        "support": (df["stance"] == "SUPPORT").sum(),
        "oppose": (df["stance"] == "OPPOSE").sum(),
        "neutral": (df["stance"] == "NEUTRAL").sum(),
        "sentiment": round(df["stance"].map(sent_map).mean() if len(df) else 0, 2),
    }
    brief = await build_brief(df["summary"].tolist(), k) if len(df) else "- No comments summarised"

    st.session_state.update(
        dict(df=df, kpis=k, brief=brief, pdf=build_pdf(df, k, docket_id, title, brief))
    )


# ──────────────────────────────────────────────
# 9. UI helpers
# ──────────────────────────────────────────────
def css(path: str = "style.css"):
    if Path(path).exists():
        st.markdown(f"<style>{Path(path).read_text()}</style>", unsafe_allow_html=True)


def kpi_cards(k: dict):
    a, b, c, d = st.columns(4)
    a.metric("Comments", k["total"])
    b.metric("Stakeholders", k["unique_orgs"])
    c.metric("S/O/N", f"{k['support']}/{k['oppose']}/{k['neutral']}")
    d.metric("Sentiment", k["sentiment"])


def phrase_chart(df: pd.DataFrame):
    phrases = [
        p.strip().title()
        for sub in df["keyphrases"]
        if isinstance(sub, list)
        for p in sub
    ]
    if not phrases:
        st.info("No keyphrases extracted.")
        return
    top = (
        pd.DataFrame(Counter(phrases).most_common(15), columns=["Keyphrase", "Count"])
        .sort_values("Count")
    )
    fig = px.bar(
        top,
        x="Count",
        y="Keyphrase",
        orientation="h",
        template="plotly_dark",
        color_discrete_sequence=["#4A90E2"],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# 10. Front-end
# ──────────────────────────────────────────────
def main():
    css()
    with st.sidebar:
        st.title("⚖️ RegNavigator Lite")
        st.write("AI digest of U.S. federal dockets.")
        st.caption("Example ID: IRS-2022-0029")

    st.header("AI briefings for federal dockets")
    with st.form("docket_form"):
        docket = st.text_input("Docket ID", "IRS-2022-0029", label_visibility="collapsed")
        n = st.slider("Comments to analyze", 5, 250, 50)
        go = st.form_submit_button("Analyze", type="primary")

    if go and docket:
        header_ph = st.empty()          # placeholder for live title
        prog = st.progress(0)
        with st.spinner("Crunching…"):
            asyncio.run(analyse(docket.strip(), n, prog, header_ph))
        prog.empty()

    if "df" in st.session_state:
        st.markdown(st.session_state.brief)
        kpi_cards(st.session_state.kpis)

        # Key‑phrase chart
        phrase_chart(st.session_state.df)

        # Download buttons just below the brief, aligned left
        st.markdown("**Download reports:**")
        btn_cols = st.columns(2)
        with btn_cols[0]:
            st.download_button(
                "CSV file",
                st.session_state.df.to_csv(index=False),
                file_name=f"{docket}_analysis.csv",
                use_container_width=True,
            )
        with btn_cols[1]:
            st.download_button(
                "PDF brief",
                st.session_state.pdf,
                file_name=f"{docket}_brief.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with st.expander("Raw data"):
            st.dataframe(st.session_state.df, use_container_width=True)

    if errs := st.session_state.get("errors"):
        with st.expander(f"Errors ({len(errs)})"):
            st.code("\n".join(errs))


if __name__ == "__main__":
    main()