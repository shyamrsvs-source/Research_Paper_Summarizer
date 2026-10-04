"""
Research Paper Summarizer - Streamlit Web Application
AI-powered academic paper summarization using Transformer-based NLP architectures.
"""

import os
import time
import json
from typing import Optional
import streamlit as st
import pymupdf as fitz

from utils.pdf_processor import extract_text_and_metadata, PDFProcessingError
from utils.text_processor import clean_text, chunk_text
from utils.summarizer import summarize_document, extract_key_points, SUMMARY_LENGTH_PRESETS
from utils.keyword_extractor import extract_keywords
from utils.evaluator import evaluate_summary
from models.model_loader import load_summarization_model, get_available_models, get_device

# Configure Streamlit Page
st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Academic UI Styling
st.markdown("""
<style>
    /* Academic Theme Styling */
    .main-title {
        font-family: 'Georgia', serif;
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .keyword-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.88rem;
        margin: 3px;
        font-weight: 500;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None
if "key_points" not in st.session_state:
    st.session_state.key_points = None
if "keywords" not in st.session_state:
    st.session_state.keywords = None
if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0.0
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model Selection
    models_dict = get_available_models()
    model_options = list(models_dict.keys())
    
    selected_model_id = st.selectbox(
        "Transformer Model:",
        options=model_options,
        index=0,
        format_func=lambda x: f"{models_dict[x]['name']} ({x})"
    )
    
    selected_model_info = models_dict[selected_model_id]
    st.caption(f"ℹ️ {selected_model_info['description']}")
    
    # Device status badge
    device = get_device()
    if device.type == "cuda":
        st.success(f"⚡ Hardware Accelerator: GPU (CUDA)")
    else:
        st.info(f"🖥️ Hardware Accelerator: CPU")
        
    st.markdown("---")
    
    # Summary Length Preset
    st.subheader("📏 Summary Settings")
    length_preset = st.radio(
        "Target Length Profile:",
        options=["Short", "Medium", "Detailed"],
        index=1,
        help="Controls generation token limits (max_length, min_length) and length penalty."
    )
    preset_info = SUMMARY_LENGTH_PRESETS[length_preset]
    st.caption(f"📝 {preset_info['description']}")
    
    # Advanced Hyperparameters
    with st.expander("🛠️ Advanced Hyperparameters"):
        num_beams = st.slider("Beam Search Width (num_beams):", min_value=1, max_value=6, value=preset_info["num_beams"])
        length_penalty = st.slider("Length Penalty:", min_value=1.0, max_value=3.0, value=float(preset_info["length_penalty"]), step=0.1)
        no_repeat_ngram = st.slider("No-Repeat N-Gram Size:", min_value=2, max_value=4, value=preset_info["no_repeat_ngram_size"])
        chunk_token_limit = st.slider("Max Chunk Tokens:", min_value=500, max_value=900, value=750, step=50,
                                     help="Chunk size strictly below BART's 1024 token limit to preserve full attention.")

    st.markdown("---")
    with st.expander("📖 AI Lab / Viva Notes"):
        st.markdown("""
        **Core Technical Concepts:**
        - **Model Architecture:** Sequence-to-Sequence (Seq2Seq) Transformer with bidirectional encoder and autoregressive decoder.
        - **Attention Mechanism:** Multi-head scaled dot-product self-attention draws long-range document dependencies.
        - **Chunking Strategy:** Preserves sentence boundaries; solves $O(N^2)$ attention limit of 1024 tokens.
        - **Evaluation:** ROUGE-1 (unigram), ROUGE-2 (bigram), ROUGE-L (LCS) computed against gold reference summaries.
        """)


# Main Interface Header
st.markdown("<div class='main-title'>📚 Research Paper Summarizer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered academic research paper summarization using Transformer-based NLP</div>", unsafe_allow_html=True)


# File Uploader & Sample Paper Loader
upload_col, sample_col = st.columns([3, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload an Academic Paper in PDF Format (.pdf):",
        type=["pdf"],
        help="Upload standard digital academic research papers (e.g. arXiv, IEEE, ACM, Springer)."
    )

with sample_col:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    load_sample = st.button("📄 Load Sample Paper", help="Loads a built-in academic paper sample for quick testing.")


# Handle Sample Paper Loading
if load_sample:
    sample_paper_text = (
        "Attention Is All You Need\n\n"
        "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin\n"
        "Google Brain, Google Research, University of Toronto\n\n"
        "Abstract\n"
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks "
        "that include an encoder and a decoder. The best performing models also connect the encoder and decoder through "
        "an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention "
        "mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks "
        "show these models to be superior in quality while being more parallelizable and requiring significantly less "
        "time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over "
        "the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, "
        "our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, "
        "a small fraction of the training costs of the best models from the literature.\n\n"
        "1 Introduction\n"
        "Recurrent neural networks, long short-term memory (LSTM) and gated recurrent neural networks (GRU) in particular, "
        "have been firmly established as state of the art approaches in sequence modeling and transduction problems such as "
        "language modeling and machine translation. Numerous efforts have since continued to push the boundaries of recurrent "
        "language models and encoder-decoder architectures. Recurrent models typically factor computation along the symbol "
        "positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a "
        "sequence of hidden states ht, as a function of the previous hidden state ht-1 and the input for position t. "
        "This inherently sequential nature precludes parallelization within training examples, which becomes critical at "
        "longer sequence lengths, as memory constraints limit batching across examples.\n\n"
        "Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various "
        "tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences. In all but "
        "a few cases, however, such attention mechanisms are used in conjunction with a recurrent network.\n\n"
        "In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an "
        "attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more "
        "parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours "
        "on eight P100 GPUs.\n\n"
        "2 Model Architecture\n"
        "Most competitive neural sequence transduction models have an encoder-decoder structure. Here, the encoder maps an input "
        "sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, "
        "the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is "
        "auto-regressive, consuming the previously generated symbols as additional input when generating the next.\n\n"
        "The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers "
        "for both the encoder and decoder. Multi-Head Attention allows the model to jointly attend to information from different "
        "representation subspaces at different positions. With one attention head, averaging inhibits this.\n\n"
        "3 Results and Conclusion\n"
        "On the WMT 2014 English-to-German translation task, the big transformer model outperforms the best previously reported "
        "models by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. On the WMT 2014 English-to-French "
        "translation task, our big model achieves a BLEU score of 41.8, outperforming all previously published single models. "
        "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing "
        "the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention."
    )
    st.session_state.extracted_data = {
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, et al. (Google Brain / Research)",
        "page_count": 5,
        "word_count": len(sample_paper_text.split()),
        "full_text": sample_paper_text,
        "is_scanned": False,
        "has_images": True,
        "warning": None,
    }
    st.session_state.uploaded_filename = "Attention_Is_All_You_Need_Sample.pdf"
    st.session_state.summary_result = None
    st.session_state.key_points = None
    st.session_state.keywords = None
    st.success("Loaded sample research paper: 'Attention Is All You Need'!")


# Process Uploaded PDF
if uploaded_file is not None and (st.session_state.uploaded_filename != uploaded_file.name):
    try:
        with st.spinner("Extracting text and analyzing PDF structure..."):
            extracted = extract_text_and_metadata(uploaded_file)
            st.session_state.extracted_data = extracted
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.summary_result = None
            st.session_state.key_points = None
            st.session_state.keywords = None
    except PDFProcessingError as pe:
        st.error(f"❌ PDF Processing Error: {str(pe)}")
        st.session_state.extracted_data = None
    except Exception as e:
        st.error(f"❌ Unexpected Error while reading PDF: {str(e)}")
        st.session_state.extracted_data = None


# Display Paper Info and Controls if Paper Loaded
data = st.session_state.extracted_data

if data:
    # Display Warnings if scanned or low text
    if data.get("warning"):
        st.markdown(f"<div class='warning-box'>{data['warning']}</div>", unsafe_allow_html=True)

    # Paper Metadata Card
    st.markdown("### 📋 Paper Information")
    meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
    with meta_c1:
        st.metric("Document Pages", data["page_count"])
    with meta_c2:
        st.metric("Total Words", f"{data['word_count']:,}")
    with meta_c3:
        st.metric("Estimated Chunks", max(1, data["word_count"] // 500))
    with meta_c4:
        st.metric("Format Status", "Digital Text" if not data["is_scanned"] else "Scanned Image")

    st.markdown(f"**Title:** {data['title']}")
    st.markdown(f"**Authors:** {data['authors']}")

    # Cleaned Text Preview Expander
    with st.expander("📖 Extracted Text Preview", expanded=False):
        cleaned_preview = clean_text(data["full_text"])
        st.text_area(
            "Extracted and Cleaned Content (Editable for targeted testing):",
            value=cleaned_preview,
            height=240,
            key="current_cleaned_text"
        )

    # Summarize Button
    st.markdown("---")
    can_summarize = not data["is_scanned"] and data["word_count"] >= 30

    if not can_summarize:
        st.warning("Cannot summarize document: document has insufficient extractable text or is a scanned image.")
    else:
        btn_col, _ = st.columns([2, 5])
        with btn_col:
            generate_clicked = st.button("🚀 Generate Summary", type="primary", use_container_width=True)

        if generate_clicked:
            active_text = st.session_state.get("current_cleaned_text", clean_text(data["full_text"]))
            
            # Progress components
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(pct: float, msg: str):
                progress_bar.progress(min(1.0, max(0.0, pct)))
                status_text.markdown(f"**Status:** {msg}")

            try:
                # 1. Load Model
                update_progress(0.02, f"Loading Transformer model `{selected_model_id}`...")
                start_time = time.time()
                tokenizer, model, dev = load_summarization_model(selected_model_id)

                # 2. Summarize Document
                custom_kwargs = {
                    "num_beams": num_beams,
                    "length_penalty": length_penalty,
                    "no_repeat_ngram_size": no_repeat_ngram
                }
                
                summary_output = summarize_document(
                    text=active_text,
                    model=model,
                    tokenizer=tokenizer,
                    device=dev,
                    length_preset=length_preset,
                    custom_params=custom_kwargs,
                    progress_callback=update_progress
                )

                # 3. Extract Key Points
                update_progress(0.92, "Extracting factual key points...")
                key_pts = extract_key_points(summary_output["final_summary"], active_text, num_points=7)

                # 4. Extract Keywords
                update_progress(0.97, "Extracting technical keywords via TF-IDF vectorization...")
                keywords_list = extract_keywords(active_text, top_n=12)

                elapsed = round(time.time() - start_time, 2)
                update_progress(1.0, f"Completed in {elapsed} seconds!")

                # Save to session state
                st.session_state.summary_result = summary_output
                st.session_state.key_points = key_pts
                st.session_state.keywords = keywords_list
                st.session_state.processing_time = elapsed

                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                st.rerun()

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error during summarization: {str(e)}")


# Display Results if Available
if st.session_state.summary_result:
    summary_res = st.session_state.summary_result
    final_summary_text = summary_res["final_summary"]
    key_points_list = st.session_state.key_points or []
    keywords_list = st.session_state.keywords or []
    elapsed_time = st.session_state.processing_time

    orig_words = data["word_count"] if data else len(final_summary_text.split())
    sum_words = len(final_summary_text.split())
    compression_ratio = round((1.0 - (sum_words / max(orig_words, 1))) * 100, 1)

    st.markdown("---")
    st.markdown("## 📊 Analysis & Summarization Results")

    # Tabs for Modular Results
    tab_summary, tab_points, tab_keywords, tab_stats, tab_eval, tab_export = st.tabs([
        "📄 Executive Summary",
        "📌 Key Findings",
        "🏷️ Technical Keywords",
        "📈 Statistics Dashboard",
        "⚖️ ROUGE Evaluation",
        "💾 Export & Download"
    ])

    # Tab 1: Executive Summary
    with tab_summary:
        st.markdown("### Abstractive Summary")
        st.markdown(f"> *Generated using `{selected_model_id}` ({summary_res['preset_used']} preset)*")
        st.info(final_summary_text)

        if summary_res.get("chunks_processed", 1) > 1:
            with st.expander(f"🔍 View Intermediate Chunk Summaries ({summary_res['chunks_processed']} Chunks)"):
                for i, c_sum in enumerate(summary_res.get("chunk_summaries", [])):
                    st.markdown(f"**Chunk {i+1} Summary:**")
                    st.write(c_sum)

    # Tab 2: Key Findings
    with tab_points:
        st.markdown("### Core Contributions & Key Points")
        st.caption("Extracted using academic discourse markers and informational sentence ranking:")
        for pt in key_points_list:
            st.markdown(f"- {pt}")

    # Tab 3: Technical Keywords
    with tab_keywords:
        st.markdown("### Domain Keywords & Technical Concepts")
        st.caption("Extracted via TF-IDF n-gram vectorization with academic meta-word filtering:")
        
        badge_html = "".join([
            f"<span class='keyword-badge'>{kw['keyword']} ({kw['score']})</span>"
            for kw in keywords_list
        ])
        st.markdown(badge_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Keyword table
        with st.expander("Detailed Keyword Relevance Scores"):
            kw_col1, kw_col2 = st.columns(2)
            half = len(keywords_list) // 2
            with kw_col1:
                for item in keywords_list[:half]:
                    st.write(f"• **{item['keyword']}**: `{item['score']}`")
            with kw_col2:
                for item in keywords_list[half:]:
                    st.write(f"• **{item['keyword']}**: `{item['score']}`")

    # Tab 4: Statistics Dashboard
    with tab_stats:
        st.markdown("### Summarization Metrics & Efficiency")
        stat_c1, stat_c2, stat_c3, stat_c4, stat_c5 = st.columns(5)
        with stat_c1:
            st.metric("Original Words", f"{orig_words:,}")
        with stat_c2:
            st.metric("Summary Words", f"{sum_words:,}")
        with stat_c3:
            st.metric("Compression", f"{compression_ratio}%")
        with stat_c4:
            st.metric("Chunks Processed", summary_res["chunks_processed"])
        with stat_c5:
            st.metric("Processing Time", f"{elapsed_time}s")

        st.markdown("#### Efficiency Analysis")
        st.markdown(f"""
        - **Information Density:** The Transformer reduced document volume by **{compression_ratio}%** while preserving core technical findings.
        - **Chunking Pipeline:** Divided into **{summary_res['chunks_processed']}** overlapping sub-windows to remain within the model's 1024-token self-attention window.
        - **Compute Device:** Executed on **{get_device().type.upper()}** using PyTorch Seq2Seq pipeline.
        """)

    # Tab 5: Evaluation & ROUGE
    with tab_eval:
        st.markdown("### Summary Evaluation with ROUGE")
        st.markdown("""
        **Academic Notice on ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**  
        ROUGE measures unigram, bigram, and longest common subsequence overlap between machine summaries and **human ground-truth reference summaries**.  
        *Calculating ROUGE against the full raw paper or the summary itself is methodologically invalid.*
        """)

        ref_input = st.text_area(
            "Paste Ground-Truth / Author's Reference Abstract for Evaluation:",
            placeholder="Paste author's abstract or human benchmark summary here...",
            height=130
        )

        col_eval_btn, _ = st.columns([2, 5])
        with col_eval_btn:
            eval_clicked = st.button("📊 Calculate ROUGE Scores", use_container_width=True)

        if eval_clicked or ref_input:
            eval_res = evaluate_summary(final_summary_text, ref_input)
            if not eval_res["has_reference"]:
                st.info(eval_res["message"])
            elif eval_res.get("error"):
                st.error(eval_res["error"])
            else:
                st.success("✅ ROUGE Evaluation Computed Successfully!")
                st.markdown(f"**Interpretation:** {eval_res['quality_assessment']}")

                r_col1, r_col2, r_col3 = st.columns(3)
                scores = eval_res["scores"]
                with r_col1:
                    st.markdown("#### ROUGE-1 (Unigrams)")
                    st.metric("F1-Measure", scores["rouge1"]["f1_pct"])
                    st.caption(f"Precision: {scores['rouge1']['precision_pct']} | Recall: {scores['rouge1']['recall_pct']}")
                    st.caption(eval_res["metrics_explanation"]["ROUGE-1"])

                with r_col2:
                    st.markdown("#### ROUGE-2 (Bigrams)")
                    st.metric("F1-Measure", scores["rouge2"]["f1_pct"])
                    st.caption(f"Precision: {scores['rouge2']['precision_pct']} | Recall: {scores['rouge2']['recall_pct']}")
                    st.caption(eval_res["metrics_explanation"]["ROUGE-2"])

                with r_col3:
                    st.markdown("#### ROUGE-L (LCS)")
                    st.metric("F1-Measure", scores["rougeL"]["f1_pct"])
                    st.caption(f"Precision: {scores['rougeL']['precision_pct']} | Recall: {scores['rougeL']['recall_pct']}")
                    st.caption(eval_res["metrics_explanation"]["ROUGE-L"])

    # Tab 6: Export & Download
    with tab_export:
        st.markdown("### Export Analysis & Summaries")
        
        # Prepare Markdown Report
        markdown_report = f"""# Research Paper Summary Report

## Paper Details
- **Title:** {data['title'] if data else 'N/A'}
- **Authors:** {data['authors'] if data else 'N/A'}
- **Total Pages:** {data['page_count'] if data else 'N/A'}
- **Original Word Count:** {orig_words}
- **Summary Word Count:** {sum_words}
- **Compression Ratio:** {compression_ratio}%
- **Model Used:** {selected_model_id}

---

## Executive Summary
{final_summary_text}

---

## Key Points & Major Contributions
""" + "\n".join([f"- {p}" for p in key_points_list]) + f"""

---

## Technical Keywords
""" + ", ".join([k['keyword'] for k in keywords_list]) + "\n"

        # Prepare JSON Data
        json_report = json.dumps({
            "paper_metadata": {
                "title": data['title'] if data else 'N/A',
                "authors": data['authors'] if data else 'N/A',
                "page_count": data['page_count'] if data else 'N/A',
                "original_word_count": orig_words,
                "summary_word_count": sum_words,
                "compression_ratio": compression_ratio,
                "model": selected_model_id,
                "processing_time_seconds": elapsed_time
            },
            "summary": final_summary_text,
            "chunk_summaries": summary_res.get("chunk_summaries", []),
            "key_points": key_points_list,
            "keywords": keywords_list
        }, indent=2)

        dl_c1, dl_c2, dl_c3 = st.columns(3)
        with dl_c1:
            st.download_button(
                label="📥 Download Markdown Report (.md)",
                data=markdown_report,
                file_name=f"summary_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with dl_c2:
            st.download_button(
                label="📥 Download Plain Text (.txt)",
                data=final_summary_text,
                file_name=f"summary_{int(time.time())}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with dl_c3:
            st.download_button(
                label="📥 Download JSON Metadata (.json)",
                data=json_report,
                file_name=f"report_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )

else:
    # Initial Empty State Guide
    st.info("👆 Upload an academic research paper in PDF format or click **'Load Sample Paper'** to begin summarization.")
    
    st.markdown("""
    ### 🔬 How the Pipeline Works:
    1. **Document Ingestion:** PyMuPDF parses the PDF, extracts structured page text, and inspects typography for metadata.
    2. **Noise Reduction:** Filters out arXiv banners, hyphenation across line wraps, and running page stamps while retaining mathematical and scientific formulas.
    3. **Token-Aware Chunking:** Segments long papers into overlapping text windows bounded by complete sentences to respect the model's 1024-token self-attention limit.
    4. **Seq2Seq Summarization:** Utilizes pretrained BART / DistilBART encoder-decoder Transformers to synthesize multi-chunk summaries hierarchically.
    5. **NLP Key Points & Keywords:** Ranks high-information sentences with academic discourse markers and extracts keyphrases via TF-IDF vectorization.
    6. **Academic Evaluation:** Provides ROUGE-1, ROUGE-2, and ROUGE-L metrics against author ground-truth abstracts.
    """)
