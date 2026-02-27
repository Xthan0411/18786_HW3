# question 4b, 4c
import argparse
import torch
from transformers import pipeline
from tqdm import tqdm
import utils

def main():
    parser = argparse.ArgumentParser()
    # 允许用户指定模型 ID
    parser.add_argument('model_id', nargs='?', default='Qwen/Qwen3-1.7B', 
                        help="HuggingFace model ID")
    parser.add_argument('--eval_corpus_path', type=str, default='birth_dev.tsv')
    args = parser.parse_args()

    print(f"Loading model {args.model_id}...")
    try:
        # 使用 pipeline 加载模型，device_map="auto" 会自动使用 GPU
        pipe = pipeline("text-generation", model=args.model_id, torch_dtype="auto", device_map="auto")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check if the model name is correct.")
        return

    # 加载数据集
    with open(args.eval_corpus_path, 'r', encoding='utf-8') as f:
        lines = [line.strip().split('\t') for line in f]

    # 定义推理函数
    def predict(prompts):
        preds = []
        print(f"Generating responses for {len(prompts)} prompts...")
        for p in tqdm(prompts):
            messages = [{"role": "user", "content": p}]
            # max_new_tokens 控制生成长度，do_sample=False 保证结果确定性
            outputs = pipe(messages, max_new_tokens=50, do_sample=False)
            # 提取生成的文本内容
            try:
                generated_text = outputs[0]["generated_text"][-1]["content"]
            except (KeyError, IndexError):
                generated_text = ""
            preds.append(generated_text.strip())
        return preds

    # --- Variant 1: 原始问题 (Original Questions) ---
    print("Evaluating Variant 1 (Original: 'Where was X born?')...")
    prompts_v1 = [line[0] for line in lines]
    preds_v1 = predict(prompts_v1)
    
    # 计算准确率
    total_v1, correct_v1 = utils.evaluate_places(args.eval_corpus_path, preds_v1)
    if total_v1 > 0:
        print(f"Variant 1 Accuracy: {correct_v1}/{total_v1} = {correct_v1/total_v1*100:.2f}%")

    # --- Variant 2: 修改后的问题 (Sensitivity Check) ---
    print("Evaluating Variant 2 (Sensitivity: 'What is the birthplace of X?')...")
    prompts_v2 = []
    for line in lines:
        q = line[0]
        # 动态修改问题格式
        if q.startswith("Where was ") and q.endswith(" born?"):
            name = q[10:-6]
            q = f"What is the birthplace of {name}?"
        prompts_v2.append(q)
    
    preds_v2 = predict(prompts_v2)
    
    # 计算准确率
    total_v2, correct_v2 = utils.evaluate_places(args.eval_corpus_path, preds_v2)
    if total_v2 > 0:
        print(f"Variant 2 Accuracy: {correct_v2}/{total_v2} = {correct_v2/total_v2*100:.2f}%")

if __name__ == "__main__":
    main()