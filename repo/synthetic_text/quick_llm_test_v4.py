"""
quick_llm_test_v4.py — Quick test of LLM classification on v4 dataset.
Tests a small sample to estimate LLM accuracy before running full cache.
"""

import gzip
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    print("Loading v4 dataset...")
    with gzip.open('synthetic_text_dataset_v4.pkl.gz', 'rb') as f:
        dataset = pickle.load(f)
    
    documents = dataset['documents']
    category_names = dataset['category_names']
    N = len(documents)
    K = len(category_names)
    
    print(f"  {N} documents, {K} categories")
    print(f"  Categories: {category_names}")
    
    # Show some sample texts
    print(f"\n{'='*70}")
    print("SAMPLE TEXTS")
    print(f"{'='*70}")
    for i in range(5):
        doc = documents[i]
        print(f"\n  [{doc['category_name']}] (llm_friendly={doc['is_llm_friendly']}, confuser={doc['is_confuser']})")
        print(f"    {doc['text'][:150]}")
    
    # Test with flan-t5-small on first 200 samples
    print(f"\n{'='*70}")
    print("TESTING FLAN-T5-SMALL on 200 samples")
    print(f"{'='*70}")
    
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    
    category_list = ", ".join(category_names)
    
    n_test = 200
    correct = 0
    correct_llm_friendly = 0
    correct_linucb_friendly = 0
    n_llm_friendly = 0
    n_linucb_friendly = 0
    
    for i in range(n_test):
        doc = documents[i]
        prompt = f"Classify this text into one of these categories: {category_list}\n\nText: {doc['text']}\n\nCategory:"
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        
        # Match prediction to category
        matched = -1
        for k, cat in enumerate(category_names):
            if cat.lower() in pred_text or pred_text in cat.lower():
                matched = k
                break
        
        is_correct = (matched == doc['category_idx'])
        if is_correct:
            correct += 1
        
        if doc['is_llm_friendly']:
            n_llm_friendly += 1
            if is_correct:
                correct_llm_friendly += 1
        else:
            n_linucb_friendly += 1
            if is_correct:
                correct_linucb_friendly += 1
    
    print(f"\n  Overall accuracy: {correct}/{n_test} = {correct/n_test*100:.1f}%")
    if n_llm_friendly > 0:
        print(f"  LLM-friendly categories: {correct_llm_friendly}/{n_llm_friendly} = {correct_llm_friendly/n_llm_friendly*100:.1f}%")
    if n_linucb_friendly > 0:
        print(f"  LinUCB-friendly categories: {correct_linucb_friendly}/{n_linucb_friendly} = {correct_linucb_friendly/n_linucb_friendly*100:.1f}%")
    
    # Show some predictions
    print(f"\n  Sample predictions (first 10):")
    for i in range(10):
        doc = documents[i]
        prompt = f"Classify this text into one of these categories: {category_list}\n\nText: {doc['text']}\n\nCategory:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print(f"    True: {doc['category_name']:<25} Pred: {pred_text}")


if __name__ == '__main__':
    main()
