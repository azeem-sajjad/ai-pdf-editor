import streamlit as st
from utils import (
    extract_text_and_blocks,
    create_index,
    search,
    ask_llm_stream,
    get_structured_edits,
    rewrite_full_document,
    apply_changes_to_pdf,
    create_styled_pdf,
    compute_diff_html,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI PDF Editor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    st.markdown("**AI Model** (Ollama — runs locally, free)")
    model_name = st.selectbox(
        "model",
        options=["llama3.2", "llama3", "mistral", "gemma2:2b", "phi3"],
        index=0,
        label_visibility="collapsed",
        help="Must be pulled via: ollama pull <model-name>",
    )

    st.divider()

    with st.expander("ℹ️ How to add a model"):
        st.code(f"ollama pull {model_name}", language="bash")
        st.caption("Run this in your terminal to download a model.")

    st.divider()
    st.caption("**Embeddings:** all-MiniLM-L6-v2 (local)")
    st.caption("**Edit modes:** Precise inline · Full rewrite")
    st.caption("**Cost:** $0.00 — runs 100% on your machine")

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("📄 AI PDF Editor")
st.caption(
    "Upload a PDF · Ask questions · Edit with natural language · "
    "Download with formatting preserved · 100% free & local"
)
st.divider()

# ── Ollama check ───────────────────────────────────────────────────────────────

try:
    import ollama as _ollama_check
    _ollama_check.list()  # will fail if Ollama isn't running
    ollama_ok = True
except Exception:
    ollama_ok = False
    st.error(
        "**Ollama is not running.** "
        "Open a terminal and run: `ollama serve` — then refresh this page.",
        icon="🔴",
    )

# ── Upload ─────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
    help="Recommended max: 20 MB",
)

if uploaded_file is None:
    st.info("👆 Upload a PDF above to get started.", icon="📂")
    st.stop()

# ── Process PDF — cached in session state ──────────────────────────────────────

file_id = f"{uploaded_file.name}__{uploaded_file.size}"

if st.session_state.get("file_id") != file_id:
    for key in ["edited_pdf_bytes", "edited_text", "changes", "chat_history"]:
        st.session_state.pop(key, None)

    with st.spinner("Reading and indexing PDF…"):
        try:
            pdf_bytes = uploaded_file.read()
            full_text, page_blocks = extract_text_and_blocks(pdf_bytes)
            chunks, index = create_index(full_text)

            st.session_state.file_id     = file_id
            st.session_state.pdf_bytes   = pdf_bytes
            st.session_state.full_text   = full_text
            st.session_state.page_blocks = page_blocks
            st.session_state.chunks      = chunks
            st.session_state.index       = index
        except Exception as exc:
            st.error(f"❌ Failed to process PDF: {exc}")
            st.stop()

pdf_bytes  = st.session_state.pdf_bytes
full_text  = st.session_state.full_text
chunks     = st.session_state.chunks
index      = st.session_state.index

st.success(
    f"✅ **{uploaded_file.name}** — "
    f"{len(full_text):,} characters · {len(chunks)} indexed chunks · "
    f"Model: `{model_name}`",
    icon="📄",
)

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_qa, tab_edit = st.tabs(["💬  Ask Questions", "✏️  Edit Document"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Q&A
# ═══════════════════════════════════════════════════════════════════════════════

with tab_qa:
    st.subheader("Ask anything about your document")
    st.caption(f"Using model: `{model_name}` — responses stream in real time")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Type your question…")

    if query:
        if not ollama_ok:
            st.error("Start Ollama first: run `ollama serve` in your terminal.")
            st.stop()

        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            context = search(query, chunks, index)

            if not context:
                st.warning("No relevant content found. Try rephrasing your question.")
            else:
                with st.expander("📎 Context used", expanded=False):
                    for i, c in enumerate(context, 1):
                        st.caption(f"**Chunk {i}:** {c[:200]}…")

                try:
                    placeholder = st.empty()
                    full_answer = ""
                    for delta in ask_llm_stream(query, context, model_name):
                        full_answer += delta
                        placeholder.markdown(full_answer + "▌")
                    placeholder.markdown(full_answer)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": full_answer}
                    )
                except Exception as exc:
                    st.error(f"LLM error: {exc}")
                    st.info("Make sure Ollama is running and the model is downloaded: "
                            f"`ollama pull {model_name}`")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDIT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_edit:
    st.subheader("Edit your document with natural language")

    col_left, col_right = st.columns([3, 1], gap="large")

    with col_left:
        instruction = st.text_area(
            "What would you like to change?",
            placeholder=(
                "Examples:\n"
                '• "Replace all instances of Contractor with Service Provider"\n'
                '• "Change the payment terms in section 3 to net-30"\n'
                '• "Make the introduction shorter and more direct"\n'
                '• "Translate the entire document to French"'
            ),
            height=140,
        )

    with col_right:
        st.markdown("**Edit mode**")
        edit_mode = st.radio(
            "mode",
            options=["🎯 Precise (inline)", "🔄 Full rewrite"],
            label_visibility="collapsed",
        )
        st.caption(
            "**Precise** — finds exact phrases and replaces them. "
            "Original formatting, images, and layout are fully preserved.\n\n"
            "**Full rewrite** — Claude rewrites the whole document. "
            "Outputs a clean styled PDF."
        )

        run_btn = st.button(
            "✏️ Apply Edit",
            type="primary",
            use_container_width=True,
            disabled=not ollama_ok,
        )

    # ── Run edit ──────────────────────────────────────────────────────────────

    if run_btn:
        if not instruction.strip():
            st.warning("Please enter an instruction above.")
        else:
            with st.spinner(f"Running `{model_name}` locally — this may take 30-60 seconds…"):
                try:
                    if "Precise" in edit_mode:
                        changes = get_structured_edits(full_text, instruction, model_name)

                        if not changes:
                            st.warning(
                                "The model didn't produce specific changes. "
                                "Try rephrasing your instruction or switch to **Full rewrite** mode."
                            )
                        else:
                            edited_pdf = apply_changes_to_pdf(pdf_bytes, changes)
                            st.session_state.edited_pdf_bytes = edited_pdf
                            st.session_state.changes          = changes
                            st.session_state.edited_text      = None

                    else:
                        edited_text = rewrite_full_document(full_text, instruction, model_name)
                        edited_pdf  = create_styled_pdf(edited_text)
                        st.session_state.edited_text      = edited_text
                        st.session_state.edited_pdf_bytes = edited_pdf
                        st.session_state.changes          = None

                except Exception as exc:
                    st.error(f"Edit failed: {exc}")
                    st.info(f"Make sure the model is downloaded: `ollama pull {model_name}`")

    # ── Show results ──────────────────────────────────────────────────────────

    if st.session_state.get("edited_pdf_bytes"):
        st.divider()

        changes     = st.session_state.get("changes")
        edited_text = st.session_state.get("edited_text")

        if changes is not None:
            st.success(
                f"✅ **{len(changes)} change(s)** applied inline — "
                "original formatting fully preserved."
            )
            with st.expander("📋 Changes applied", expanded=True):
                for i, c in enumerate(changes, 1):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**{i}. Removed**")
                        st.code(c.get("find", ""), language=None)
                    with c2:
                        st.markdown(f"**{i}. Replaced with**")
                        st.code(c.get("replace", ""), language=None)

        elif edited_text:
            st.success("✅ Document fully rewritten.")
            with st.expander("📊 Diff — red = removed · green = added", expanded=True):
                diff_html = compute_diff_html(full_text, edited_text)
                st.components.v1.html(
                    f"""<div style="max-height:400px;overflow-y:auto;padding:16px;
                    border:1px solid #e0e0e0;border-radius:8px;background:#fafafa;">
                    {diff_html}</div>""",
                    height=440,
                    scrolling=True,
                )

        st.download_button(
            label="📥 Download Edited PDF",
            data=st.session_state.edited_pdf_bytes,
            file_name=f"edited_{uploaded_file.name}",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )