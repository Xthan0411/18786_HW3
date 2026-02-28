# question 4b, 4c
import argparse
import torch
from transformers import pipeline
from tqdm import tqdm
import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', nargs='?', default='Qwen/Qwen3-1.7B', 
                        help="HuggingFace model ID")
    parser.add_argument('--eval_corpus_path', type=str, default='birth_dev.tsv')
    args = parser.parse_args()

    print(f"Loading model {args.model_id}...")
    try:
        pipe = pipeline("text-generation", model=args.model_id, torch_dtype="auto", device_map="auto")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check if the model name is correct.")
        return

    with open(args.eval_corpus_path, 'r', encoding='utf-8') as f:
        lines = [line.strip().split('\t') for line in f]

    # Load ground truth for debugging
    true_places = [line[1] for line in lines]

    def predict(prompts):
        preds = []
        print(f"Generating responses for {len(prompts)} prompts...")
        for i, p in tqdm(enumerate(prompts), total=len(prompts)):
            content = p + " Answer with the city or country name only. Do not write a sentence."
            messages = [{"role": "user", "content": content}]
            outputs = pipe(messages, max_new_tokens=50, do_sample=False)
            try:
                generated_text = outputs[0]["generated_text"][-1]["content"]
            except (KeyError, IndexError):
                generated_text = ""
            
            # Enhanced cleaning logic
            pred = generated_text.strip()
            # Remove common prefixes if the model ignores instructions
            for prefix in ["The answer is ", "The birthplace is ", "He was born in ", "She was born in ", "It is "]:
                if pred.lower().startswith(prefix.lower()):
                    pred = pred[len(prefix):]
            
            pred = pred.split('\n')[0] # Take first line
            pred = pred.split(',')[0]  # Remove country if format is "City, Country" (e.g. "London, UK" -> "London")
            pred = pred.strip().rstrip('.') # Remove trailing dots and spaces
            
            # Debug print for the first 5 examples to see what's going wrong
            if i < 5:
                print(f"\n[Debug] Q: {p}")
                print(f"[Debug] Raw: {generated_text}")
                print(f"[Debug] Cleaned: {pred} | Expected: {true_places[i]}")

            preds.append(pred)
        return preds

    print("Evaluating Variant 1 (Original: 'Where was X born?')...")
    prompts_v1 = [line[0] for line in lines]
    preds_v1 = predict(prompts_v1)
    
    total_v1, correct_v1 = utils.evaluate_places(args.eval_corpus_path, preds_v1)
    if total_v1 > 0:
        print(f"Variant 1 Accuracy: {correct_v1}/{total_v1} = {correct_v1/total_v1*100:.2f}%")

    print("Evaluating Variant 2 (Sensitivity: 'What is the birthplace of X?')...")
    prompts_v2 = []
    for line in lines:
        q = line[0]
        if q.startswith("Where was ") and q.endswith(" born?"):
            name = q[10:-6]
            q = f"What is the birthplace of {name}?"
        prompts_v2.append(q)
    
    preds_v2 = predict(prompts_v2)
    
    total_v2, correct_v2 = utils.evaluate_places(args.eval_corpus_path, preds_v2)
    if total_v2 > 0:
        print(f"Variant 2 Accuracy: {correct_v2}/{total_v2} = {correct_v2/total_v2*100:.2f}%")

if __name__ == "__main__":
    main()