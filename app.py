"""
Second Brain — Local Media RAG Application
===========================================
Upload audio/video → Transcribe via Groq Whisper → Store in SQLite FTS5 → Chat with your media.

Run:  streamlit run app.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ─── Load .env ───
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from database import init_db, get_connection, insert_transcript, get_all_media, delete_media, get_stats
from transcriber import process_media, safe_remove, ALL_MEDIA, format_duration
from retriever import chat_with_brain


# ═══════════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Second Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
/* Source cards */
.source-card {
    background: rgba(30,41,59,0.6);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.85rem;
}
.source-title { font-weight: 600; color: #a5b4fc; }
.source-meta { color: #64748b; font-size: 0.78rem; }
/* Media library item */
.media-item {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
}
.media-title { font-weight: 700; color: #e2e8f0; font-size: 0.95rem; }
.media-meta { color: #64748b; font-size: 0.78rem; margin-top: 4px; }
.media-preview {
    color: #94a3b8; font-size: 0.82rem;
    margin-top: 8px; line-height: 1.5;
    max-height: 80px; overflow: hidden;
}
/* Stats cards */
.stat-card {
    text-align: center; padding: 16px;
    background: rgba(30,41,59,0.5);
    border-radius: 10px;
    border: 1px solid rgba(99,102,241,0.15);
}
.stat-number { font-size: 1.8rem; font-weight: 700; color: #818cf8; }
.stat-label { font-size: 0.78rem; color: #64748b; margin-top: 4px; }
/* Keywords */
.keyword-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Database Init
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def _init_db():
    conn = get_connection()
    init_db(conn)
    return conn

db = _init_db()


# ═══════════════════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════════════════

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ═══════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 Second Brain")
    st.caption("Upload → Transcribe → Chat with your media")

    # API Key Status
    st.markdown("---")
    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    gemini_ok = bool(os.getenv("GEMINI_API_KEY"))
    st.markdown(f"{'🟢' if groq_ok else '🔴'} Groq (Whisper)")
    st.markdown(f"{'🟢' if gemini_ok else '🔴'} Gemini (Chat)")
    if not groq_ok:
        st.warning("GROQ_API_KEY needed for transcription.")

    # Stats
    st.markdown("---")
    stats = get_stats(db)
    m1, m2, m3 = st.columns(3)
    m1.metric("📁 Files", stats["total_media"])
    m2.metric("📝 Chars", f"{stats['total_characters']:,}")
    m3.metric("🗣️ Languages", len(stats["by_type"]))

    # Media Library
    st.markdown("---")
    st.markdown("### 📚 Library")
    media_list = get_all_media(db)

    if not media_list:
        st.info("No media uploaded yet. Drop some files above!")
    else:
        for item in media_list:
            ts = datetime.fromtimestamp(item["timestamp"]).strftime("%b %d, %H:%M")
            with st.expander(f"{'🎬' if item['file_type'] in ('mp4','mkv','avi','mov','webm') else '🎵'} {item['title']}", expanded=False):
                st.markdown(f"**Duration:** {item['duration']} • **Type:** {item['file_type']} • **Added:** {ts}")
                st.markdown(f"**Characters:** {item['char_count']:,}")
                st.markdown("---")
                st.markdown(item["transcript"][:500] + ("..." if len(item["transcript"]) > 500 else ""))
                st.markdown("---")
                cols = st.columns([1, 1])
                with cols[0]:
                    st.download_button(
                        "📥 Export",
                        item["transcript"],
                        file_name=f"{item['title'].replace(' ', '_')}.txt",
                        mime="text/plain",
                        key=f"exp_{item['id']}",
                    )
                with cols[1]:
                    if st.button("🗑️ Delete", key=f"del_{item['id']}"):
                        delete_media(db, item["id"])
                        st.rerun()


# ═══════════════════════════════════════════════════════════════════
# Main Content
# ═══════════════════════════════════════════════════════════════════

st.markdown("# 🧠 Second Brain")
st.markdown("Upload audio/video files → Transcribed via Groq Whisper → Search & chat with your media library.")

# ── File Upload ──
st.markdown("---")
col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_files = st.file_uploader(
        "Drop audio/video files here",
        type=[ext.lstrip(".") for ext in ALL_MEDIA],
        accept_multiple_files=True,
        help=f"Accepted: {', '.join(sorted(ALL_MEDIA))}",
    )

with col_info:
    st.markdown("""
    **How it works:**
    1. 📤 Upload `.mp4`, `.mp3`, `.m4a`, etc.
    2. 🎙️ Audio extracted & transcribed (Groq Whisper)
    3. 🏷️ Auto-titled via Gemini
    4. 💾 Stored in SQLite FTS5
    5. 💬 Chat with your transcripts!
    """)

# ── Process Uploaded Files ──
if uploaded_files:
    st.markdown("---")
    st.markdown("### 📥 Processing Files")

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_ext = Path(file_name).suffix.lower()

        # Check if already processed (by name)
        existing = db.execute(
            "SELECT id FROM media WHERE title LIKE ?",
            (f"%{file_name[:30]}%",),
        ).fetchone()
        if existing:
            st.info(f"⏭️ Already in library: {file_name}")
            continue

        with st.spinner(f"Processing {file_name}..."):
            # Write to temp file
            tmp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
            tmp_file.write(uploaded_file.read())
            tmp_file.close()

            try:
                # Run the full pipeline
                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    process_media(tmp_file.name, file_name)
                )
                loop.close()

                # Store in database
                media_id = insert_transcript(
                    db,
                    title=result["title"],
                    transcript=result["transcript"],
                    duration=result["duration"],
                    file_type=result["file_type"],
                )

                st.success(
                    f"✅ **{result['title']}** — {result['duration']} "
                    f"({result['file_type']}, {len(result['transcript']):,} chars)"
                )

            except ValueError as e:
                st.error(f"🔑 {e}")
            except Exception as e:
                st.error(f"❌ Failed to process {file_name}: {e}")
            finally:
                # Aggressive cleanup
                safe_remove(tmp_file.name)

# ── Chat with Second Brain ──
st.markdown("---")
st.markdown("### 💬 Chat with Your Brain")

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🧠"):
            # Show keywords as badges
            if msg.get("keywords"):
                badges = " ".join(
                    [f"<span class='keyword-badge'>{kw}</span>" for kw in msg["keywords"]]
                )
                st.markdown(f"🔎 Search terms: {badges}", unsafe_allow_html=True)

            st.markdown(msg["content"])

            # Show sources
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source(s) used", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f"""<div class="source-card">
                                <span class="source-title">{src['title']}</span>
                                <span class="source-meta"> • {src['duration']}</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )

# Chat input
if user_input := st.chat_input("Ask about your media library..."):
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("🧠 Searching your brain..."):
            # Run the RAG pipeline
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                chat_with_brain(user_input, st.session_state.chat_history)
            )
            loop.close()

            # Display keywords
            if result["keywords"]:
                badges = " ".join(
                    [f"<span class='keyword-badge'>{kw}</span>" for kw in result["keywords"]]
                )
                st.markdown(f"🔎 Search terms: {badges}", unsafe_allow_html=True)

            # Display answer
            st.markdown(result["answer"])

            # Display sources
            if result["sources"]:
                with st.expander(f"📎 {len(result['sources'])} source(s) used", expanded=False):
                    for src in result["sources"]:
                        st.markdown(
                            f"""<div class="source-card">
                                <span class="source-title">{src['title']}</span>
                                <span class="source-meta"> • {src['duration']}</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )

    # Append assistant response
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "keywords": result["keywords"],
        "sources": result["sources"],
    })

# ── Bottom Toolbar ──
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
with col2:
    if st.button("📥 Export Chat"):
        import json
        export = {
            "history": st.session_state.chat_history,
            "exported_at": datetime.now().isoformat(),
        }
        st.download_button(
            "⬇️ Download",
            json.dumps(export, indent=2, default=str),
            file_name=f"brain_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
        )


# ═══════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#64748b;font-size:0.75rem'>"
    "Second Brain • SQLite FTS5 + Groq Whisper + LiteLLM • "
    "Optimised for Windows 11 / 8 GB RAM"
    "</div>",
    unsafe_allow_html=True,
)
