import os, io, base64, json, re
from typing import List, Dict, Any
from dataclasses import dataclass
import streamlit as st
from PIL import Image
import pandas as pd
from dotenv import load_dotenv
import openai

try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except ImportError:
    pass  # Not in Streamlit

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OpenAI API key not found. Set it in Streamlit secrets or as an environment variable.")

# ============== General Helpers ==============
def to_data_url(image_bytes: bytes, mime: str) -> str:
    mime = mime if (mime and str(mime).startswith("image/")) else "image/jpeg"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def parse_json_strict(text: str) -> Dict[str, Any]:
    """
    Try to parse strict JSON; if fences or extra text appear, salvage the largest JSON object.
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", str(text).strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise

# ============== Pretty amenity labels (no per-item map) ==============
SMALL_WORDS = {"and","or","of","the","a","an","in","on","for","to"}
ACRONYMS = {"tv","ac","co","bbq","usb","hdmi","led","vr"}  # extend if you need

def pretty_amenity_auto(name: str) -> str:
    if not name:
        return ""
    s = name.strip().lower()
    s = s.replace("_or_", " / ").replace("_and_", " & ").replace("_", " ")
    s = re.sub(r"\bwifi\b", "wi-fi", s)

    parts = re.split(r"([/&])", s)
    out = []
    for part in parts:
        if part in {"/","&"}:
            out.append(f" {part} ")
            continue
        words = part.split()
        fmt = []
        for i, w in enumerate(words):
            if w in ACRONYMS:          # only known acronyms → uppercase
                fmt.append(w.upper())
            elif i > 0 and w in SMALL_WORDS:
                fmt.append(w)
            else:
                fmt.append(w.capitalize())
        out.append(" ".join(fmt))
    label = "".join(out)
    return re.sub(r"\s{2,}", " ", label).strip()

def amenity_badges(labels):
    if not labels:
        return "<div class='badges'>—</div>"
    # add a literal space between spans so they never collapse together
    items = " ".join(f"<span class='badge'>{label}</span>" for label in labels)
    return f"<div class='badges'>{items}</div>"

BADGE_CSS = """
<style>
.badges { display:flex; flex-wrap:wrap; gap:.4rem; margin:.25rem 0 .5rem; }
.badge  { display:inline-block; padding:.25rem .6rem; border-radius:999px;
          border:1px solid #ddd; background:#f7f7f9; font-size:.90rem;
          margin-right:.4rem; margin-bottom:.35rem; }  /* fallback spacing */
</style>
"""


# Inject CSS once
if "badge_css" not in st.session_state:
    st.session_state["badge_css"] = True
    st.markdown(BADGE_CSS, unsafe_allow_html=True)

# ============== ROOMS CLASSIFIER ==============
ALLOWED_ROOMS = [
    "bedroom", "living_room", "kitchen", "bathroom", "dining_room",
    "hallway", "balcony_terrace", "exterior", "facade_building",
    "garden_patio", "pool", "laundry", "toilet_half_bath",
    "office_study", "kids_room", "parking_garage", "view_window",
    "staircase", "other_unclear"
]

SYSTEM_INSTRUCTIONS_ROOMS = (
    "You are an expert real-estate photo classifier. "
    "Classify each image into Airbnb-relevant room types. "
    "Use ONLY the allowed labels from the user prompt, "
    "and return STRICT JSON following the provided schema."
)

USER_PROMPT_TEMPLATE_ROOMS = f"""
Classify the image into 1–2 room types from this STRICT list:
{ALLOWED_ROOMS}

Rules:
- If it clearly shows a single room, return that one as primary.
- If it's an open-plan scene (e.g., kitchen + living_room), include both (multi_room=true).
- If unsure, use "other_unclear".
- Only rely on what is visible in the photo.
- Be concise.

Return ONLY strict JSON (no markdown, no commentary) in exactly this schema:
{{
  "primary_room": "<one of {ALLOWED_ROOMS}>",
  "secondary_rooms": ["<optional second from list>"],
  "multi_room": <true|false>,
  "confidence_primary": <float between 0 and 1>,
  "notes": "<max 20 words>"
}}
"""

@dataclass
class ClassificationRooms:
    primary_room: str
    secondary_rooms: List[str]
    multi_room: bool
    confidence_primary: float
    notes: str

def classify_image_rooms(image_bytes: bytes, mime: str, model: str) -> ClassificationRooms:
    data_url = to_data_url(image_bytes, mime)
    resp = openai.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS_ROOMS},
            {"role": "user",
             "content": [
                 {"type": "text", "text": USER_PROMPT_TEMPLATE_ROOMS},
                 {"type": "image_url", "image_url": {"url": data_url}},
             ],
            },
        ],
    )
    raw = resp.choices[0].message.content
    if isinstance(raw, list):
        raw = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in raw)
    data = parse_json_strict(raw)

    primary = data.get("primary_room", "other_unclear")
    if primary not in ALLOWED_ROOMS:
        primary = "other_unclear"
    secondary = [s for s in data.get("secondary_rooms", []) if s in ALLOWED_ROOMS]
    multi_room = bool(data.get("multi_room", False))
    try:
        conf = float(data.get("confidence_primary", 0.0))
    except Exception:
        conf = 0.0
    notes = str(data.get("notes", ""))[:120]

    return ClassificationRooms(primary, secondary, multi_room, conf, notes)

# ============== AMENITIES CLASSIFIER (no explicit label list) ==============
SYSTEM_INSTRUCTIONS_AMENITIES = (
    "You are an expert at visually detecting amenities in real-estate photos. "
    "ONLY report amenities that are clearly visible in the photo. "
    "Do not guess (e.g., do not include 'wifi' unless a router/modem is obviously visible). "
    "Prefer concise snake_case names (e.g., hair_dryer, coffee_maker, tv, oven, bathtub, microwave, dishwasher, "
    "washer, dryer, bbq_grill, dining_table, desk, smoke_alarm, co_alarm, fire_extinguisher, smart_lock, "
    "security_camera_exterior, video_doorbell, air_conditioning, ceiling_fan, kettle, toaster, blender, "
    "refrigerator, freezer). "
    "Return STRICT JSON."
)

USER_PROMPT_TEMPLATE_AMENITIES = """
Detect only the visually obvious amenities present in this photo.

Return ONLY strict JSON (no commentary) with this schema:
{
  "amenities_detected": [
    {"name": "<snake_case name>", "confidence": <float 0..1>},
    ...
  ]
}

Rules:
- Output at most 12 items.
- Names must be snake_case (lowercase letters, digits, and underscores only).
- Only include amenities that you can SEE in the image.
"""

@dataclass
class ClassificationAmenities:
    amenities_detected: List[Dict[str, Any]]

def classify_image_amenities(image_bytes: bytes, mime: str, model: str) -> ClassificationAmenities:
    data_url = to_data_url(image_bytes, mime)
    resp = openai.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS_AMENITIES},
            {"role": "user",
             "content": [
                 {"type": "text", "text": USER_PROMPT_TEMPLATE_AMENITIES},
                 {"type": "image_url", "image_url": {"url": data_url}},
             ],
            },
        ],
    )
    raw = resp.choices[0].message.content
    if isinstance(raw, list):
        raw = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in raw)
    data = parse_json_strict(raw)

    # Normalize: dedupe, keep sane chars
    detected = []
    seen = set()
    for item in data.get("amenities_detected", []):
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        # keep only snake_case-ish tokens
        if not re.fullmatch(r"[a-z0-9_]{2,64}", name):
            continue
        if name in seen:
            continue
        seen.add(name)
        try:
            conf = float(item.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        detected.append({"name": name, "confidence": conf})

    return ClassificationAmenities(detected)

# ============== UI ==============
st.set_page_config(page_title="Airbnb Photo Analyzer", layout="wide")
st.title("🏠 Airbnb Photo Analyzer")

st.sidebar.header("Settings")
analysis_mode = st.sidebar.radio("Analysis", ["Rooms", "Amenities"], index=0)
model_choice = st.sidebar.selectbox(
    "Model",
    options=["gpt-4o-mini", "gpt-4o"],
    index=0,
    help="Use gpt-4o-mini for speed/cost; gpt-4o for tougher images."
)

uploads = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="Drag & drop multiple images here."
)

if uploads:
    rows = []
    for f in uploads:
        bytes_ = f.read()

        # Optional: downscale very large images to save tokens/latency
        try:
            img = Image.open(io.BytesIO(bytes_)).convert("RGB")
            if max(img.size) > 1800:
                img.thumbnail((1800, 1800))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90)
                bytes_ = buf.getvalue()
        except Exception:
            img = None

        with st.spinner(f"Analyzing {f.name}..."):
            try:
                if analysis_mode == "Rooms":
                    r = classify_image_rooms(bytes_, f.type, model_choice)
                    rows.append({
                        "file": f.name,
                        "primary_room": r.primary_room,
                        "secondary_rooms": ", ".join(r.secondary_rooms) if r.secondary_rooms else "",
                        "multi_room": r.multi_room,
                        "confidence_primary": round(r.confidence_primary, 3),
                        "notes": r.notes
                    })
                else:
                    a = classify_image_amenities(bytes_, f.type, model_choice)
                    amenity_names = [d["name"] for d in a.amenities_detected]
                    amenity_labels = [pretty_amenity_auto(n) for n in amenity_names]
                    rows.append({
                        "file": f.name,
                        "amenities": amenity_labels,                 # list for chips
                        "amenities_str": ", ".join(amenity_labels),  # for table/export
                    })
            except Exception as e:
                if analysis_mode == "Rooms":
                    rows.append({
                        "file": f.name, "primary_room": "error", "secondary_rooms": "",
                        "multi_room": "", "confidence_primary": "", "notes": f"Parse/Model error: {e}"
                    })
                else:
                    rows.append({
                        "file": f.name, "amenities": [], "amenities_str": "",
                    })

        # Display image + per-file result
        cols = st.columns([1, 2])
        with cols[0]:
            if img is not None:
                st.image(img, caption=f.name)
            else:
                st.write("Preview not available.")
        with cols[1]:
            last = rows[-1]
            if analysis_mode == "Rooms":
                st.markdown(
                    f"""
                    **Primary:** `{last.get('primary_room', '')}`  
                    **Secondary:** `{last.get('secondary_rooms', '—') or '—'}`  
                    **Multi-room:** `{last.get('multi_room', '')}`  
                    **Confidence:** `{last.get('confidence_primary', '')}`  
                    **Notes:** {last.get('notes', '')}
                    """
                )
            else:
                st.markdown(
                    amenity_badges(last.get("amenities", [])),
                    unsafe_allow_html=True
                )
        st.markdown("---")

    # Summary table + CSV
    st.subheader("Summary")
    if analysis_mode == "Amenities":
        df = pd.DataFrame([{"file": r["file"], "amenities": ", ".join(r.get("amenities", []))} for r in rows])
    else:
        df = pd.DataFrame(rows)
    st.dataframe(df)


    # Optional: global list of unique amenities found
    if analysis_mode == "Amenities":
        all_found = sorted({label for r in rows for label in r.get("amenities", [])})
        st.markdown("**Amenities found across all images:** " + (", ".join(all_found) if all_found else "—"))
else:
    st.info("Upload a few Airbnb photos to get started. Use the sidebar to switch between **Rooms** and **Amenities**.")
