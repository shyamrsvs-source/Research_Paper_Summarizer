"""
Comprehensive End-to-End Test Suite for Research Paper Summarizer.
Tests all scenarios specified in the project requirements:
1. Short research paper
2. Medium-length research paper
3. Long research paper (hierarchical chunking)
4. Multi-page PDF
5. Invalid/non-PDF file
6. PDF with little or no extractable text (scanned PDF detection)
7. Full pipeline with model inference and ROUGE evaluation
"""

import os
import pymupdf as fitz
from utils.pdf_processor import extract_text_and_metadata, PDFProcessingError
from utils.text_processor import clean_text, chunk_text
from utils.summarizer import summarize_document, extract_key_points
from utils.keyword_extractor import extract_keywords
from utils.evaluator import evaluate_summary
from models.model_loader import load_summarization_model


def create_test_pdf(filename: str, pages_data: list) -> str:
    """Helper to create test PDF files with specified page contents."""
    os.makedirs("test_fixtures", exist_ok=True)
    filepath = os.path.join("test_fixtures", filename)
    doc = fitz.open()
    for page_text in pages_data:
        page = doc.new_page()
        if page_text:
            rect = fitz.Rect(50, 50, 550, 750)
            page.insert_textbox(rect, page_text, fontsize=10)
    doc.save(filepath)
    doc.close()
    return filepath


def run_all_e2e_tests():
    print("=" * 60)
    print("STARTING COMPREHENSIVE END-TO-END PIPELINE TESTS")
    print("=" * 60)

    # 1. Test Short Research Paper
    print("\n[TEST 1] Short Research Paper (1-2 pages, ~350 words)...")
    short_content = [
        (
            "Deep Residual Learning for Image Recognition\n\n"
            "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun\n"
            "Microsoft Research\n\n"
            "Abstract\n"
            "Deeper neural networks are more difficult to train. We present a residual learning framework "
            "to ease the training of networks that are substantially deeper than those used previously. "
            "We explicitly reformulate the layers as learning residual functions with reference to the layer "
            "inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence "
            "showing that these residual networks are easier to optimize, and can gain accuracy from considerably "
            "increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers, "
            "which is 8x deeper than VGG nets while still having lower complexity.\n\n"
            "1 Introduction\n"
            "Deep convolutional neural networks have led to a series of breakthroughs for image classification. "
            "Deep networks naturally integrate low/mid/high-level features and classifiers in an end-to-end "
            "multi-layer manner, and the 'levels' of features can be enriched by the number of stacked layers. "
            "Driven by the significance of depth, recent leading architectures for ImageNet all exploit 'very deep' "
            "models. When deeper networks are able to start converging, a degradation problem has been exposed: "
            "with the network depth increasing, accuracy gets saturated and then degrades rapidly."
        )
    ]
    short_pdf = create_test_pdf("short_paper.pdf", short_content)
    res_short = extract_text_and_metadata(short_pdf)
    print(f"  - Title: {res_short['title']}")
    print(f"  - Pages: {res_short['page_count']}, Words: {res_short['word_count']}")
    assert res_short["word_count"] > 100
    assert not res_short["is_scanned"]
    print("  [OK] Short paper processed successfully.")

    # 2. Test Medium-Length Research Paper (~1,200 words across 4 pages)
    print("\n[TEST 2] Medium-Length Research Paper (4 pages, ~1,200 words)...")
    p1 = (
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding\n\n"
        "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova\n"
        "Google AI Language\n\n"
        "Abstract\n"
        "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder "
        "Representations from Transformers. Unlike recent language representation models, BERT is designed to "
        "pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left "
        "and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one "
        "additional output layer to create state-of-the-art models for a wide range of tasks, such as question "
        "answering and language inference, without substantial task-specific architecture modifications."
    )
    p2 = " ".join([f"Section 2 discusses pre-training objective {i} using masked language models and next sentence prediction. "
                   f"The Transformer bidirectional self-attention encoder learns representations effectively across layers." for i in range(12)])
    p3 = " ".join([f"Section 3 outlines empirical evaluation {i} on the GLUE benchmark suite and Stanford Question Answering Dataset. "
                   f"Experimental results show BERT advances state of the art on eleven natural language processing tasks." for i in range(12)])
    p4 = " ".join([f"Section 4 details ablation studies {i} confirming the necessity of bidirectional pre-training over unidirectional models. "
                   f"The authors conclude with future work on multi-task language representation architectures." for i in range(10)])

    med_pdf = create_test_pdf("medium_paper.pdf", [p1, p2, p3, p4])
    res_med = extract_text_and_metadata(med_pdf)
    print(f"  - Title: {res_med['title']}")
    print(f"  - Pages: {res_med['page_count']}, Words: {res_med['word_count']}")
    chunks_med = chunk_text(clean_text(res_med["full_text"]), max_chunk_tokens=600)
    print(f"  - Chunks generated: {len(chunks_med)}")
    assert res_med["page_count"] == 4
    assert len(chunks_med) >= 2
    print("  [OK] Medium paper chunking verified.")

    # 3. Test Long Research Paper (~3,000 words across 8 pages)
    print("\n[TEST 3] Long Research Paper (8 pages, ~3,000 words)...")
    long_pages = [p1] + [
        " ".join([f"Comprehensive analysis and mathematical formulation paragraph {page_idx}_{para_idx} covering cross-entropy loss, "
                  f"attention weight matrices, positional embeddings, layer normalization, and gradient descent optimization routines."
                  for para_idx in range(10)])
        for page_idx in range(2, 9)
    ]
    long_pdf = create_test_pdf("long_paper.pdf", long_pages)
    res_long = extract_text_and_metadata(long_pdf)
    print(f"  - Pages: {res_long['page_count']}, Words: {res_long['word_count']}")
    chunks_long = chunk_text(clean_text(res_long["full_text"]), max_chunk_tokens=600)
    print(f"  - Hierarchical chunks generated: {len(chunks_long)}")
    assert res_long["page_count"] == 8
    assert len(chunks_long) >= 4
    print("  [OK] Long paper hierarchical chunking verified.")

    # 4. Test Multi-Page PDF Handling
    print("\n[TEST 4] Multi-Page Structural Integrity Check...")
    assert len(res_long["pages_text"]) == 8
    assert all(isinstance(p, str) for p in res_long["pages_text"])
    print("  [OK] Multi-page extraction and page array structure verified.")

    # 5. Test Invalid / Non-PDF File Handling
    print("\n[TEST 5] Invalid / Corrupted File Handling...")
    invalid_file_path = os.path.join("test_fixtures", "corrupt_file.pdf")
    with open(invalid_file_path, "wb") as f:
        f.write(b"NOT_A_VALID_PDF_HEADER_JUST_RANDOM_GARBAGE_BYTES_1234567890")

    try:
        extract_text_and_metadata(invalid_file_path)
        assert False, "Should have raised PDFProcessingError"
    except PDFProcessingError as pe:
        print(f"  - Gracefully caught custom exception: {str(pe)[:80]}...")
        print("  [OK] Invalid PDF handled gracefully without uncaught crash.")

    # 6. Test Scanned / Empty PDF Detection
    print("\n[TEST 6] Empty / Scanned PDF Detection...")
    empty_pdf = create_test_pdf("scanned_empty_paper.pdf", [""])
    res_empty = extract_text_and_metadata(empty_pdf)
    print(f"  - Is Scanned Flag: {res_empty['is_scanned']}")
    print(f"  - Warning Message: {res_empty['warning'][:75]}...")
    assert res_empty["is_scanned"] is True or res_empty["word_count"] == 0
    assert res_empty["warning"] is not None
    print("  [OK] Scanned/empty PDF successfully detected with user-friendly warning.")

    # 7. Test Full End-to-End Pipeline on Short Paper (Model + Keywords + Key Points + ROUGE)
    print("\n[TEST 7] Full End-to-End Summarization Pipeline...")
    tokenizer, model, device = load_summarization_model()
    print(f"  - Model loaded on {device}")
    
    cleaned_input = clean_text(res_short["full_text"])
    sum_result = summarize_document(
        text=cleaned_input,
        model=model,
        tokenizer=tokenizer,
        device=device,
        length_preset="Short"
    )
    print(f"  - Generated Summary ({len(sum_result['final_summary'].split())} words):")
    print(f"    \"{sum_result['final_summary']}\"")
    assert len(sum_result["final_summary"]) > 20

    key_points = extract_key_points(sum_result["final_summary"], cleaned_input, num_points=5)
    print(f"  - Extracted {len(key_points)} Key Points:")
    for pt in key_points:
        print(f"    - {pt}")
    assert len(key_points) >= 4

    keywords = extract_keywords(cleaned_input, top_n=6)
    print(f"  - Extracted {len(keywords)} Keywords:")
    for kw in keywords:
        print(f"    - {kw['keyword']} ({kw['score']})")
    assert len(keywords) > 0

    # ROUGE Evaluation with Reference
    ref_abstract = (
        "We present a residual learning framework to ease the training of networks that are substantially "
        "deeper than those used previously. We explicitly reformulate layers as learning residual functions."
    )
    eval_result = evaluate_summary(sum_result["final_summary"], ref_abstract)
    print(f"  - ROUGE-1 F1: {eval_result['scores']['rouge1']['f1_pct']}")
    print(f"  - ROUGE-2 F1: {eval_result['scores']['rouge2']['f1_pct']}")
    print(f"  - ROUGE-L F1: {eval_result['scores']['rougeL']['f1_pct']}")
    assert eval_result["has_reference"] is True
    print("  [OK] Full pipeline executed successfully.")

    print("\n" + "=" * 60)
    print("ALL 7 END-TO-END SCENARIOS VERIFIED AND PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_e2e_tests()
