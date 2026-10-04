"""
Unit test and verification script for Research Paper Summarizer modules.
Tests PDF extraction, text cleaning, chunking, keyword extraction, and ROUGE evaluation.
"""

import os
import pymupdf as fitz
from utils.pdf_processor import extract_text_and_metadata, PDFProcessingError
from utils.text_processor import clean_text, chunk_text
from utils.keyword_extractor import extract_keywords
from utils.evaluator import evaluate_summary


def test_pdf_processing():
    print("\n--- Testing PDF Processing ---")
    
    # 1. Create a dummy multi-page academic PDF using PyMuPDF
    test_pdf_path = "test_sample_paper.pdf"
    doc = fitz.open()
    
    # Page 1: Title, author, abstract, introduction
    page1 = doc.new_page()
    rect1 = fitz.Rect(50, 50, 550, 750)
    text_p1 = (
        "Attention Is All You Need: Modern Transformers\n\n"
        "Ashish Vaswani, Noam Shazeer, Niki Parmar\n"
        "Google Brain and Google Research\n\n"
        "Abstract\n"
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. "
        "We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an "
        "attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly "
        "more parallelization and can reach a new state of the art in translation quality after being trained for as little "
        "as twelve hours on eight P100 GPUs.\n\n"
        "1 Introduction\n"
        "Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly "
        "established as state of the art approaches in sequence modeling and transduction problems such as language modeling "
        "and machine translation. Numerous efforts have since continued to push the boundaries of recurrent language models "
        "and encoder-decoder architectures. Recurrent models typically factor computation along the symbol positions of the input "
        "and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states."
    )
    page1.insert_textbox(rect1, text_p1, fontsize=11)

    # Page 2: Model Architecture
    page2 = doc.new_page()
    rect2 = fitz.Rect(50, 50, 550, 750)
    text_p2 = (
        "2 Model Architecture\n\n"
        "Most competitive neural sequence transduction models have an encoder-decoder structure. Here, the encoder maps an "
        "input sequence of symbol representations to a sequence of continuous representations. Given z, the decoder then "
        "generates an output sequence of symbols one element at a time. At each step the model is auto-regressive, consuming "
        "the previously generated symbols as additional input when generating the next.\n\n"
        "The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers "
        "for both the encoder and decoder. Multi-Head Attention allows the model to jointly attend to information from different "
        "representation subspaces at different positions."
    )
    page2.insert_textbox(rect2, text_p2, fontsize=11)
    
    doc.save(test_pdf_path)
    doc.close()
    
    result = extract_text_and_metadata(test_pdf_path)
    print(f"Extracted Title: {result['title']}")
    print(f"Extracted Authors: {result['authors']}")
    print(f"Page Count: {result['page_count']}")
    print(f"Word Count: {result['word_count']}")
    print(f"Is Scanned: {result['is_scanned']}")
    assert result['page_count'] == 2, "Expected 2 pages"
    assert result['word_count'] > 100, "Expected word count > 100"
    assert not result['is_scanned'], "Digital PDF should not be marked as scanned"
    print("[PASS] PDF Processing Test Passed!")

    # 2. Test empty / scanned detection
    scanned_pdf_path = "test_empty_paper.pdf"
    doc_empty = fitz.open()
    doc_empty.new_page()  # Blank page
    doc_empty.save(scanned_pdf_path)
    doc_empty.close()
    
    res_empty = extract_text_and_metadata(scanned_pdf_path)
    print(f"Blank PDF Scanned Flag: {res_empty['is_scanned']}")
    print(f"Blank PDF Warning: {res_empty['warning']}")
    assert res_empty['is_scanned'] is True or res_empty['word_count'] == 0
    print("[PASS] Empty/Scanned PDF Detection Test Passed!")

    # Clean up test files
    if os.path.exists(test_pdf_path):
        os.remove(test_pdf_path)
    if os.path.exists(scanned_pdf_path):
        os.remove(scanned_pdf_path)


def test_text_processing():
    print("\n--- Testing Text Cleaning and Chunking ---")
    noisy_text = (
        "The dominant sequence trans-\nformation models rely on recur-\nrent neural networks. "
        "Page 1 of 12\narXiv:1706.03762v5 [cs.CL] 12 Jun 2017\n"
        "We propose the Transformer architecture.    It achieves superior translation accuracy. \n\n\n\n"
        "The multi-head attention mechanism draws global dependencies efficiently."
    )
    cleaned = clean_text(noisy_text)
    print("Cleaned Text:\n" + cleaned)
    assert "transformation" in cleaned, "Dehyphenation failed (transformation)"
    assert "recurrent" in cleaned, "Dehyphenation failed (recurrent)"
    assert "arXiv" not in cleaned, "arXiv pattern should be removed"
    assert "Page 1 of 12" not in cleaned, "Page stamp should be removed"
    print("[PASS] Text Cleaning Test Passed!")

    # Test chunking
    long_text = " ".join([f"Sentence {i} explains a vital aspect of deep learning and self-attention." for i in range(120)])
    chunks = chunk_text(long_text, max_chunk_tokens=100, overlap_sentences=1)
    print(f"Generated {len(chunks)} chunks from {len(long_text.split())} words.")
    assert len(chunks) > 1, "Expected multiple chunks for long text"
    print("[PASS] Text Chunking Test Passed!")


def test_keyword_extraction():
    print("\n--- Testing Keyword Extraction ---")
    academic_text = (
        "In this work, we present deep learning methods for natural language processing using "
        "transformer models and attention mechanisms. Recurrent neural networks have limitations "
        "in capturing long-range dependencies. The self-attention mechanism enables parallel computation "
        "and improves text summarization benchmarks. Transformer architectures achieve state of the art "
        "results across language translation and summarization tasks."
    )
    keywords = extract_keywords(academic_text, top_n=6)
    print("Extracted Keywords:")
    for kw in keywords:
        print(f"  - {kw['keyword']}: score {kw['score']}")
    assert len(keywords) > 0, "Keywords should not be empty"
    kw_names = [k["keyword"].lower() for k in keywords]
    assert any("attention" in k or "transformer" in k or "neural" in k or "language" in k for k in kw_names)
    print("[PASS] Keyword Extraction Test Passed!")


def test_evaluator():
    print("\n--- Testing ROUGE Evaluator ---")
    # Case 1: No reference
    res_no_ref = evaluate_summary("The model achieves high accuracy.")
    print("No Reference status:", res_no_ref["has_reference"])
    assert res_no_ref["has_reference"] is False
    assert "ROUGE" in res_no_ref["message"]

    # Case 2: With reference
    gen_sum = "We propose the Transformer model based entirely on self-attention mechanisms for language translation."
    ref_sum = "We propose the Transformer, an architecture relying solely on attention mechanisms to achieve superior translation."
    res_ref = evaluate_summary(gen_sum, ref_sum)
    print("ROUGE-1:", res_ref["scores"]["rouge1"])
    print("ROUGE-2:", res_ref["scores"]["rouge2"])
    print("ROUGE-L:", res_ref["scores"]["rougeL"])
    print("Quality Assessment:", res_ref["quality_assessment"])
    assert res_ref["has_reference"] is True
    assert res_ref["scores"]["rouge1"]["f1"] > 0.4
    print("[PASS] ROUGE Evaluator Test Passed!")


if __name__ == "__main__":
    test_pdf_processing()
    test_text_processing()
    test_keyword_extraction()
    test_evaluator()
    print("\n=== ALL UNIT TESTS PASSED SUCCESSFULLY! ===")
