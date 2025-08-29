import json
import re
from typing import List, Optional
from slide_schema import SlidePlan, SlideItem

# ---- Prompt helpers ----------------------------------------------------------

PROMPT_JSON_SPEC = """
Return ONLY valid minified JSON with this exact schema:
{
  "slides": [
    { "title": "string", "bullets": ["string", ...], "notes": "string" },
    ...
  ]
}
Do not include markdown fences or any prose before/after the JSON.
"""

def build_planning_prompt(user_text: str, tone: Optional[str]) -> str:
    style_hint = f"Desired tone/style: {tone}." if tone else ""
    return f"""
You are a slide content planner. Analyze the user's text and produce a well-structured presentation.

Goals:
- Choose a reasonable number of slides (not too many, not too few).
- Create concise, parallel bullets (5-8 words when possible).
- Each bullet should be complete, crisp statements.
- Use clear titles that summarize the slide.
- 
- Include optional speaker notes to expand each slide concisely.
- Prefer grouping: Overview → Key Points → Details → Examples → Conclusion/Next steps.

{style_hint}

User text:
\"\"\"{user_text}\"\"\"

{PROMPT_JSON_SPEC}
""".strip()

# ---- JSON cleaners -----------------------------------------------------------

def clean_llm_output(text: str) -> str:
    """Remove markdown fences and extract JSON block if present."""
    text = text.strip()
    # Remove triple backtick fences like ```json ... ``` or ``` ... ```
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"```$", "", text)
    # Extract JSON object if wrapped in prose
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def safe_json_parse(s: str) -> Optional[SlidePlan]:
    try:
        cleaned = clean_llm_output(s)
        obj = json.loads(cleaned)
        return SlidePlan(**obj)
    except Exception:
        return None

# ---- Heuristic fallback if LLM fails ----------------------------------------

def simple_chunker(text: str, max_chars: int = 600) -> SlidePlan:
    import textwrap
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 > max_chars:
            if buf:
                chunks.append(buf.strip())
                buf = ""
        buf += (" " if buf else "") + p
    if buf:
        chunks.append(buf.strip())

    slides: List[SlideItem] = []
    for i, c in enumerate(chunks, 1):
        bullets = [b.strip() for b in textwrap.wrap(c, width=80)]
        slides.append(SlideItem(title=f"Section {i}", bullets=bullets[:6], notes=""))
    return SlidePlan(slides=slides)

# ---- Post-processing: dedupe & drop empties ---------------------------------

def slide_richness(slide: SlideItem) -> int:
    """Score slide by content amount to keep the richer duplicate."""
    bullets_count = len(slide.bullets or [])
    notes_len = len((slide.notes or "").strip())
    return bullets_count * 10 + notes_len  # bullets matter more than notes

def clean_slide_plan(plan: SlidePlan) -> SlidePlan:
    """Remove title-only slides and deduplicate by title (keep richer one)."""
    by_title = {}
    for s in plan.slides:
        has_bullets = bool(s.bullets and len(s.bullets) > 0)
        has_notes = bool(s.notes and s.notes.strip())
        if not has_bullets and not has_notes:
            # skip title-only
            continue
        key = (s.title or "").strip()
        if not key:
            # slides must have a title; if missing, skip
            continue
        if key not in by_title or slide_richness(s) > slide_richness(by_title[key]):
            by_title[key] = s
    cleaned = list(by_title.values())
    # Preserve original order as best-effort by sorting with first occurrence
    order_index = { (s.title or "").strip(): i for i, s in enumerate(plan.slides) }
    cleaned.sort(key=lambda s: order_index.get((s.title or "").strip(), 0))
    return SlidePlan(slides=cleaned)

# ---- Provider calls ----------------------------------------------------------

def call_openai(api_key: str, prompt: str, user_text: str) -> SlidePlan:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        # Enforce JSON output when supported
        response_format={"type": "json_object"}
    )
    content = resp.choices[0].message.content.strip()
    plan = safe_json_parse(content) or simple_chunker(user_text)
    return clean_slide_plan(plan)

def call_anthropic(api_key: str, prompt: str, user_text: str) -> SlidePlan:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2000,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    # Claude often returns fenced json; cleaner handles it.
    content = resp.content[0].text.strip()
    plan = safe_json_parse(content) or simple_chunker(user_text)
    return clean_slide_plan(plan)

def call_gemini(api_key: str, prompt: str, user_text: str) -> SlidePlan:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()
    plan = safe_json_parse(text) or simple_chunker(user_text)
    return clean_slide_plan(plan)

# ---- Router entrypoint -------------------------------------------------------

def plan_slides(provider: str, api_key: str, user_text: str, tone: Optional[str]) -> SlidePlan:
    prompt = build_planning_prompt(user_text, tone)
    if provider == "openai":
        return call_openai(api_key, prompt, user_text)
    if provider == "anthropic":
        return call_anthropic(api_key, prompt, user_text)
    if provider == "gemini":
        return call_gemini(api_key, prompt, user_text)
    raise ValueError("Unsupported provider")


# in llm_router.py (or a shared utils file)

def clean_slide_plan(plan: SlidePlan) -> SlidePlan:
    """Remove empty slides and duplicates by title, keeping only the richer one."""
    cleaned_slides = []
    seen_titles = set()
    for slide in plan.slides:
        # Skip slides that only have a title (no bullets/notes)
        if (not slide.bullets or len(slide.bullets) == 0) and not slide.notes:
            continue
        # Skip duplicate titles (keep first full one)
        if slide.title in seen_titles:
            continue
        seen_titles.add(slide.title)
        cleaned_slides.append(slide)
    return SlidePlan(slides=cleaned_slides)

