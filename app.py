import os
import io
import json
import base64
import re
import html
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import fitz
from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except Exception:
    arabic_reshaper = None
    get_display = None

load_dotenv()
st.set_page_config(page_title="دستیار تحلیل آزمایش خون", page_icon="🩸", layout="wide")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "12"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
try:\n    API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)\nexcept Exception:\n    API_KEY = os.getenv("OPENAI_API_KEY")

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "patient_name": {"type": ["string", "null"]},
        "report_date": {"type": ["string", "null"]},
        "laboratory": {"type": ["string", "null"]},
        "tests": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": ["number", "string", "null"]},
                "unit": {"type": ["string", "null"]},
                "reference_text": {"type": ["string", "null"]},
                "reference_low": {"type": ["number", "null"]},
                "reference_high": {"type": ["number", "null"]},
                "printed_flag": {"type": ["string", "null"]},
                "page": {"type": ["integer", "null"]}
            },
            "required": ["name", "value", "unit", "reference_text", "reference_low", "reference_high", "printed_flag", "page"],
            "additionalProperties": False
        }},
        "uncertainties": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["patient_name", "report_date", "laboratory", "tests", "uncertainties"],
    "additionalProperties": False
}

INTERPRET_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "test_name": {"type": "string"},
                "plain_language": {"type": "string"},
                "follow_up": {"type": "string"},
                "urgency": {"type": "string"}
            },
            "required": ["test_name", "plain_language", "follow_up", "urgency"],
            "additionalProperties": False
        }},
        "general_note": {"type": "string"}
    },
    "required": ["summary", "findings", "general_note"],
    "additionalProperties": False
}


def safe_json_from_response(response):
    raw = getattr(response, "output_text", "") or ""
    if not raw:
        raise ValueError("پاسخ متنی معتبری از مدل دریافت نشد.")
    return json.loads(raw)


def image_pages(uploaded_file):
    data = uploaded_file.getvalue()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"حجم فایل بیشتر از {MAX_UPLOAD_MB} مگابایت است.")
    name = uploaded_file.name.lower()
    pages = []
    if name.endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        if doc.page_count > 8:
            raise ValueError("برای این نسخه حداکثر ۸ صفحه PDF مجاز است.")
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
            pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        pages.append(img)
    output = []
    for img in pages:
        img.thumbnail((2200, 2200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88, optimize=True)
        output.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    return output


def agent_1_extract(client, page_data_urls):
    content = [{"type": "input_text", "text": (
        "You are an OCR and transcription agent for laboratory reports. "
        "Transcribe every visible blood-test row exactly. Do not diagnose. "
        "Use the printed reference interval only; do not invent one. "
        "Return numeric value and numeric low/high only when clearly printed and parseable. "
        "If uncertain, use null and add a concise uncertainty. Preserve units and decimal points. "
        "Do not infer patient identity. Output JSON matching the requested schema. "
        "The images are pages in order; page numbers start at 1."
    )}]
    for url in page_data_urls:
        content.append({"type": "input_image", "image_url": url, "detail": "high"})
    response = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "json_schema", "name": "blood_report_extraction", "schema": EXTRACTION_SCHEMA, "strict": True}}
    )
    return safe_json_from_response(response)


def parse_numeric(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("٫", ".")
        match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
        if match:
            return float(match.group())
    return None


def agent_2_compare(extraction):
    results = []
    for test in extraction.get("tests", []):
        value = parse_numeric(test.get("value"))
        low = parse_numeric(test.get("reference_low"))
        high = parse_numeric(test.get("reference_high"))
        status = "نامشخص"
        reason = "محدوده مرجع عددیِ قابل‌خواندن در برگه موجود نیست؛ از مقایسه حدسی خودداری شد."
        if value is not None and low is not None and high is not None:
            if value < low:
                status, reason = "پایین‌تر از محدوده", f"مقدار {value:g} کمتر از حد پایین {low:g} است."
            elif value > high:
                status, reason = "بالاتر از محدوده", f"مقدار {value:g} بیشتر از حد بالای {high:g} است."
            else:
                status, reason = "در محدوده درج‌شده", f"مقدار {value:g} بین {low:g} و {high:g} قرار دارد."
        elif value is not None and test.get("printed_flag"):
            status = "علامت‌گذاری‌شده در برگه"
            reason = "پرچم آزمایشگاه ثبت شد؛ محدوده عددی برای راستی‌آزمایی کافی نیست."
        results.append({**test, "numeric_value": value, "status": status, "comparison_note": reason})
    return results


def agent_3_explain(client, comparisons, context):
    payload = {"results": comparisons, "context": context}
    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": (
                "You are a cautious health-information explainer, not a clinician. "
                "Explain only the supplied laboratory values. Never diagnose, prescribe, recommend stopping medication, "
                "or claim a result proves a disease. Abnormal values can have many causes; use conditional language. "
                "Prioritize clinician review and ask the user to compare the transcription with the original report. "
                "Do not recommend supplements or treatment. Suggest only low-risk general steps such as discussing results "
                "with a clinician, bringing medication/history lists, and seeking urgent care for severe symptoms. "
                "Do not label anything urgent solely from a single lab value; use 'پیگیری معمول' unless supplied context "
                "contains severe symptoms, then say seek urgent medical care. Write clear Persian. Return JSON only."
            )},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ],
        text={"format": {"type": "json_schema", "name": "cautious_interpretation", "schema": INTERPRET_SCHEMA, "strict": True}}
    )
    return safe_json_from_response(response)


def rtl(text):
    text = html.escape(str(text or ""))
    if arabic_reshaper and get_display:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text


def register_persian_font():
    candidates = [
        Path("assets/fonts/Vazirmatn-Regular.ttf"),
        Path("assets/fonts/NotoNaskhArabic-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf")
    ]
    for path in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont("PersianFont", str(path)))
            return "PersianFont"
    return "Helvetica"


def agent_4_pdf(extraction, comparisons, interpretation, context):
    buffer = io.BytesIO()
    font = register_persian_font()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm, topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("PersianTitle", parent=styles["Title"], fontName=font, fontSize=16, leading=23, alignment=TA_CENTER, textColor=colors.HexColor("#17324D"))
    body = ParagraphStyle("PersianBody", parent=styles["BodyText"], fontName=font, fontSize=9, leading=15, alignment=TA_RIGHT, spaceAfter=5)
    small = ParagraphStyle("PersianSmall", parent=body, fontSize=8, leading=12)
    story = [Paragraph(rtl("گزارش کمکی تحلیل آزمایش خون"), title), Spacer(1, 5*mm)]
    meta = f"تاریخ تهیه گزارش: {datetime.now().strftime('%Y-%m-%d %H:%M')} | تاریخ برگه: {extraction.get('report_date') or 'نامشخص'}"
    story.append(Paragraph(rtl(meta), small))
    story.append(Paragraph(rtl("این گزارش جایگزین تشخیص یا نظر پزشک نیست. مقادیر استخراج‌شده را با تصویر اصلی تطبیق دهید."), body))
    if context:
        story.append(Paragraph(rtl("اطلاعات زمینه‌ای ارائه‌شده توسط کاربر: " + json.dumps(context, ensure_ascii=False)), small))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(rtl("خلاصه"), body))
    story.append(Paragraph(rtl(interpretation.get("summary", "")), body))
    table_data = [[rtl("آزمایش"), rtl("مقدار"), rtl("واحد"), rtl("محدوده مرجع"), rtl("وضعیت")]]
    for item in comparisons:
        table_data.append([
            rtl(item.get("name", "")),
            rtl(item.get("value", "")),
            rtl(item.get("unit") or ""),
            rtl(item.get("reference_text") or "درج نشده"),
            rtl(item.get("status", "نامشخص"))
        ])
    table = Table(table_data, repeatRows=1, colWidths=[43*mm, 22*mm, 22*mm, 42*mm, 39*mm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), font), ("FONTSIZE", (0,0), (-1,-1), 7),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E8EFF5")),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#B8C4CE")),
        ("ALIGN", (0,0), (-1,-1), "RIGHT"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5)
    ]))
    story += [Spacer(1, 3*mm), table, Spacer(1, 4*mm), Paragraph(rtl("توضیح و پیگیری پیشنهادی"), body)]
    finding_map = {x.get("test_name"): x for x in interpretation.get("findings", [])}
    for item in comparisons:
        finding = finding_map.get(item.get("name"))
        if finding:
            story.append(KeepTogether([
                Paragraph(rtl(f"• {item.get('name')} — {item.get('status')}"), body),
                Paragraph(rtl(finding.get("plain_language", "")), body),
                Paragraph(rtl("پیگیری: " + finding.get("follow_up", "")), body)
            ]))
    story.append(Paragraph(rtl(interpretation.get("general_note", "")), body))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(rtl("محدودیت‌ها: این ابزار ممکن است در خواندن تصویر، واحدها یا محدوده‌ها خطا کند. تفسیر آزمایش به شرایط فرد و روش آزمایشگاه وابسته است. دارو را بر اساس این گزارش شروع، قطع یا تغییر ندهید."), small))
    doc.build(story)
    return buffer.getvalue()


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Vazirmatn', sans-serif; }
.block-container { max-width: 1120px; padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("🩸 دستیار هوشمند تحلیل آزمایش خون")
st.caption("استخراج اطلاعات، مقایسه با محدوده مرجع برگه، توضیح محتاطانه و گزارش PDF")

with st.expander("مهم: حریم خصوصی و محدودیت پزشکی", expanded=True):
    st.warning(
        "این برنامه ابزار آموزشی و اطلاع‌رسانی است، نه پزشک و نه سامانه تشخیص قطعی. "
        "تصویر برای تحلیل به سرویس هوش مصنوعی ارسال می‌شود. نام، کد ملی و اطلاعات هویتی را پیش از بارگذاری بپوشانید. "
        "در وضعیت اورژانسی یا علائم شدید، از خدمات درمانی فوری کمک بگیرید."
    )

if not API_KEY:
    st.error("کلید OPENAI_API_KEY تنظیم نشده است. آن را در فایل .env یا Secrets میزبان اضافه کنید.")
    st.stop()

client = OpenAI(api_key=API_KEY)
uploaded = st.file_uploader("تصویر یا PDF برگه آزمایش را بارگذاری کنید", type=["png", "jpg", "jpeg", "pdf"])

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("سن (اختیاری)", min_value=0, max_value=120, value=0)
    sex = st.selectbox("جنس ثبت‌شده در پرونده (اختیاری)", ["ذکر نشده", "زن", "مرد", "سایر"])
with col2:
    symptoms = st.text_area("علائم فعلی (اختیاری)", placeholder="مثلاً خستگی؛ در صورت نداشتن، خالی بگذارید.")
    history = st.text_area("سابقه پزشکی و داروها (اختیاری)", placeholder="فقط موارد مرتبط را وارد کنید.")

consent = st.checkbox("می‌دانم فایل برای تحلیل به سرویس API ارسال می‌شود و با این ارسال موافقم.")
run = st.button("تحلیل آزمایش و ساخت گزارش", type="primary", disabled=not (uploaded and consent))

if run:
    try:
        with st.spinner("مرحله ۱ از ۴: خواندن تصویر و استخراج اطلاعات..."):
            page_urls = image_pages(uploaded)
            extraction = agent_1_extract(client, page_urls)
        with st.spinner("مرحله ۲ از ۴: مقایسه با محدوده مرجع درج‌شده در برگه..."):
            comparisons = agent_2_compare(extraction)
        context = {
            "age": age if age else None,
            "sex": None if sex == "ذکر نشده" else sex,
            "symptoms": symptoms.strip() or None,
            "history_and_medicines": history.strip() or None
        }
        context = {k: v for k, v in context.items() if v is not None}
        with st.spinner("مرحله ۳ از ۴: تهیه توضیح محتاطانه..."):
            interpretation = agent_3_explain(client, comparisons, context)
        with st.spinner("مرحله ۴ از ۴: ساخت PDF..."):
            pdf_bytes = agent_4_pdf(extraction, comparisons, interpretation, context)
        st.session_state["analysis"] = {
            "extraction": extraction, "comparisons": comparisons,
            "interpretation": interpretation, "pdf": pdf_bytes
        }
    except Exception as exc:
        st.error(f"تحلیل کامل نشد: {exc}")
        st.info("کیفیت تصویر، اتصال API، دسترسی مدل و تنظیم کلید را بررسی کنید. در صورت نیاز فایل را با کیفیت بهتر بارگذاری کنید.")

result = st.session_state.get("analysis")
if result:
    st.success("گزارش آماده شد.")
    st.subheader("خلاصه")
    st.write(result["interpretation"].get("summary", ""))
    st.subheader("نتایج استخراج‌شده")
    st.dataframe([
        {"آزمایش": x.get("name"), "مقدار": x.get("value"), "واحد": x.get("unit"),
         "محدوده مرجع": x.get("reference_text"), "وضعیت": x.get("status")}
        for x in result["comparisons"]
    ], use_container_width=True, hide_index=True)
    if result["extraction"].get("uncertainties"):
        st.subheader("موارد نیازمند بازبینی")
        for item in result["extraction"]["uncertainties"]:
            st.write("• " + item)
    st.subheader("توضیحات")
    for finding in result["interpretation"].get("findings", []):
        with st.expander(finding.get("test_name", "توضیح")):
            st.write(finding.get("plain_language", ""))
            st.write("**پیگیری پیشنهادی:** " + finding.get("follow_up", ""))
    st.info(result["interpretation"].get("general_note", ""))
    st.download_button(
        "⬇️ دانلود گزارش PDF",
        data=result["pdf"],
        file_name="blood_test_report_fa.pdf",
        mime="application/pdf",
        type="primary"
    )
