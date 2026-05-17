from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
from backend import app


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress if available; else invoke.
    Yields ("updates"/"values"/"final", payload).
    """
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        yield ("final", None)
        return
    except Exception:
        pass

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        yield ("final", None)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# -----------------------------
# Markdown renderer that supports local images
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            try:
                st.image(src, caption=caption or (alt or None), use_container_width=False)
            except Exception as e:
                st.error(f"Failed to load image from URL: {src}\nError: {e}")
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists() and img_path.is_file():
                try:
                    st.image(str(img_path), caption=caption or (alt or None), use_container_width=False)
                except Exception as e:
                    st.error(f"Failed to load image: {img_path}\nError: {e}")
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1


# -----------------------------
# ✅ NEW: Past blogs helpers
# -----------------------------
def list_past_blogs() -> List[Path]:
    """
    Returns .md files in current working directory, newest first.
    Filters out obvious non-blog markdown files if needed.
    """
    cwd = Path(".")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    """
    Use first '# ' heading as title if present.
    """
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")

st.title("Blog Writing Agent")

with st.sidebar:
    st.header("Generate New Blog")
    topic = st.text_area(
        "Topic",
        height=120,
    )
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary")

    # ✅ NEW: Past blogs list (keeps everything else intact)
    st.divider()
    st.subheader("Past blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (*.md in current folder).")
        selected_md_file = None
    else:
        # Build labels from file name + (optional) parsed title
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a blog to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load selected blog"):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                json_file = selected_md_file.with_suffix('.json')
                
                state_loaded = False
                if json_file.exists():
                    try:
                        st.session_state["last_out"] = json.loads(json_file.read_text(encoding="utf-8"))
                        st.session_state["last_out"]["final"] = md_text
                        state_loaded = True
                    except Exception as e:
                        pass
                        
                if not state_loaded:
                    st.session_state["last_out"] = {
                        "plan": None,          # old files don't include plan
                        "evidence": [],        # old files don't include evidence
                        "image_specs": [],     # optional (not persisted)
                        "final": md_text,      # markdown body
                    }
                # also update the topic input to the title (best-effort) without changing UI
                st.session_state["topic_prefill"] = extract_title_from_md(md_text, selected_md_file.stem)

    

# Keep your topic input as-is; optionally prefill for next run after loading a blog
if "topic_prefill" in st.session_state and isinstance(st.session_state["topic_prefill"], str):
    # Do not mutate widgets; just keep as a hint.
    pass

# Storage for latest run
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# Layout
tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs"]
)

logs: List[str] = []

# --- Missing Execution Logic ---
if run_btn:
    if not topic.strip():
        st.error("Please enter a topic before generating.")
        st.stop()
        
    inputs = {
        "topic": topic.strip(),
        "as_of": as_of.isoformat()
    }
    
    with st.status("🚀 Running Blog Writer...", expanded=True) as status_box:
        st.session_state["last_out"] = {}
        for kind, payload in try_stream(app, inputs):
            if kind == "updates":
                st.session_state["last_out"] = extract_latest_state(st.session_state["last_out"], payload)
                step_name = list(payload.keys())[0] if isinstance(payload, dict) else str(payload)
                msg = f"Update from step: {step_name}"
                logs.append(msg)
                status_box.update(label=f"Running step: {step_name}")
                st.write(msg)
            elif kind == "values":
                st.session_state["last_out"] = extract_latest_state(st.session_state["last_out"], payload)
            elif kind == "final":
                if payload is not None:
                    st.session_state["last_out"] = payload
                else:
                    payload = st.session_state["last_out"]
                status_box.update(label="Finished!", state="complete")
                st.write("Done!")
                
                # Save full state to JSON companion file
                try:
                    plan_obj = payload.get("plan")
                    title = plan_obj.blog_title if hasattr(plan_obj, "blog_title") else (plan_obj.get("blog_title", "blog") if isinstance(plan_obj, dict) else "blog")
                    json_path = Path(f"{safe_slug(title)}.json")
                    
                    import copy
                    save_payload = copy.deepcopy(payload)
                    if hasattr(save_payload.get("plan"), "model_dump"):
                        save_payload["plan"] = save_payload["plan"].model_dump()
                    if save_payload.get("evidence"):
                        save_payload["evidence"] = [e.model_dump() if hasattr(e, "model_dump") else e for e in save_payload["evidence"]]
                        
                    json_path.write_text(json.dumps(save_payload, indent=2), encoding="utf-8")
                except Exception as e:
                    st.toast(f"Note: Could not save JSON state companion file: {e}")

# --- Missing Rendering Logic ---
last_out = st.session_state.get("last_out")
if last_out:
    # 1. Preview
    with tab_preview:
        md = last_out.get("final") or last_out.get("merged_md")
        if md:
            render_markdown_with_local_images(md)
            
            st.divider()
            # ZIP Download
            plan_obj = last_out.get("plan")
            title = "blog"
            if plan_obj and hasattr(plan_obj, "blog_title"):
                title = plan_obj.blog_title
            elif isinstance(plan_obj, dict) and "blog_title" in plan_obj:
                title = plan_obj["blog_title"]
                
            md_filename = f"{safe_slug(title)}.md"
            zip_bytes = bundle_zip(md, md_filename, Path("images"))
            st.download_button(
                label="📦 Download ZIP (Markdown + Images)",
                data=zip_bytes,
                file_name=f"{safe_slug(title)}.zip",
                mime="application/zip"
            )
        else:
            st.info("No markdown content generated yet.")

    # 2. Plan
    with tab_plan:
        plan = last_out.get("plan")
        if plan:
            # Handle Pydantic model vs dict
            if hasattr(plan, "model_dump"):
                st.json(plan.model_dump())
            else:
                st.json(plan)
        else:
            st.info("No plan generated.")

    # 3. Evidence
    with tab_evidence:
        evidence = last_out.get("evidence")
        if evidence:
            st.json([e.model_dump() if hasattr(e, "model_dump") else e for e in evidence])
        else:
            st.info("No research evidence collected for this run.")

    # 4. Images
    with tab_images:
        image_specs = last_out.get("image_specs")
        if image_specs:
            st.json(image_specs)
        else:
            st.info("No images planned/generated.")

    # 5. Logs
    with tab_logs:
        if logs:
            for l in logs:
                st.text(l)
        else:
            st.info("No logs captured.")