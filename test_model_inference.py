"""
Model inference verification script.
Tests model loading and single-chunk / document summarization on sample academic text.
"""

import time
from models.model_loader import load_summarization_model, get_device
from utils.summarizer import summarize_document, extract_key_points

sample_text = (
    "The Transformer is a sequence-to-sequence model architecture introduced by Vaswani et al. "
    "Unlike recurrent neural networks, the Transformer relies entirely on multi-head self-attention mechanisms, "
    "allowing all words in a sequence to be processed simultaneously rather than sequentially. "
    "This parallelization significantly reduces training time on GPU clusters while improving language translation "
    "accuracy on the WMT 2014 English-to-German and English-to-French benchmarks. "
    "The architecture consists of an encoder stack and a decoder stack, each equipped with scaled dot-product attention, "
    "position-wise feed-forward networks, and residual connections with layer normalization. "
    "Experimental results demonstrated state of the art BLEU scores while training in a fraction of the time required "
    "by traditional LSTM and convolutional architectures."
)

print(f"Detected hardware device: {get_device()}")
print("Loading model 'sshleifer/distilbart-cnn-12-6'...")
start_load = time.time()
tokenizer, model, device = load_summarization_model("sshleifer/distilbart-cnn-12-6")
print(f"Model loaded in {time.time() - start_load:.2f} seconds.")

print("Running summarization on sample text...")
start_gen = time.time()
res = summarize_document(
    text=sample_text,
    model=model,
    tokenizer=tokenizer,
    device=device,
    length_preset="Short"
)
gen_time = time.time() - start_gen

print("\n--- Summary Result ---")
print(res["final_summary"])
print(f"Generated in {gen_time:.2f}s across {res['chunks_processed']} chunk(s).")

print("\n--- Key Points ---")
key_pts = extract_key_points(res["final_summary"], sample_text, num_points=4)
for pt in key_pts:
    print(f"  * {pt}")

print("\n=== MODEL INFERENCE VERIFIED! ===")
