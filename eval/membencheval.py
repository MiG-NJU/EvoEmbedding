import json
import os, random
import numpy as np
from collections import defaultdict

from model.client import qwen3_client, OpenAIClient, EvoEmbeddingClient
from eval.longmemeval import get_response, print_metrics, get_response_rag, get_response_rag_ours

def get_dataset(DATA_PATH, seed=42, tmp_path="final-eval.json"):
    random.seed(seed)
    np.random.seed(seed)

    if os.path.exists(os.path.join(DATA_PATH, tmp_path)):
        return json.load(open(os.path.join(DATA_PATH, tmp_path), "r"))
    print("Loading dataset...")
    file_list = os.listdir(DATA_PATH)
    dataset = {}
    print(file_list, DATA_PATH)
    for file in file_list:
        print(file)
        if file.endswith(".json") is False or file == tmp_path:
            continue
        data = json.load(open(os.path.join(DATA_PATH, file), 'r'))
        if not isinstance(data, dict):
            samples = data
        else:
            samples = []
            for key in data.keys():
                samples.extend(data[key])
        dataset[file.split('.')[0]] = samples[:100] # random.sample(samples, 200)

    with open(os.path.join(DATA_PATH, tmp_path), "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    return dataset


def build_question_id(question_type, sample):
    # return f"{question_type}::{sample['QA']['time']}::{sample['QA']['question']}"
    import hashlib
    history_str = str(sample.get("message_list", ""))
    history_hash = hashlib.md5(history_str.encode()).hexdigest()[:8]
    
    # 加上哈希后缀，保证即时问题一样，只要对话历史不同，ID 就不一样
    return f"{question_type}::{sample['QA']['time']}::{sample['QA']['question']}::{history_hash}"



def build_messages(sample):
    messages = []
    for mess in sample["message_list"]:
        for mess_con in mess:
            messages.extend([
                {
                    "role": "user",
                    "content": f"{mess_con['time']} {mess_con['place']}: {mess_con.get('user_message', mess_con.get('user'))}",
                },
                {
                    "role": "assistant",
                    "content": f"{mess_con.get('assistant_message', mess_con.get('assistant'))}"
                },
            ])
    return messages


def build_question_prompt(sample, model_name):
    question = sample["QA"]["question"]
    all_options = sample["QA"]["choices"]
    return (
        "Please answer the following question based on the history of conversations:\n"
        f"Question time: {sample['QA']['time']} and question: {question}\n"
        f"Options:\n{str(all_options)}\n"
        "Find the most appropriate model response and give your final answer A, B, C, or D only."
    )


def load_existing_results(save_path):
    if not os.path.exists(save_path):
        return []
    result = json.load(open(save_path, "r"))
    if result and isinstance(result[-1], dict) and "statistics" in result[-1]:
        result = result[:-1]
    return result


def eval(DATA_PATH, eval_method, rag_sentence_num, save_path, model_name, embedding_model):
    dataset = get_dataset(DATA_PATH)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result = load_existing_results(save_path)
    completed_question_ids = {
        item.get("question_id")
        for item in result
        if isinstance(item, dict) and item.get("question_id") is not None
    }

    if "qwen" in model_name.lower():
        eval_model = qwen3_client(model_name)
    elif "evoembedding" in model_name.lower():
        eval_model = EvoEmbeddingClient(model_name=model_name)
    else:
        eval_model = OpenAIClient(model_name=model_name)

    total_samples = sum(len(value) for value in dataset.values())
    processed = len(completed_question_ids)

    for idx, (key, value) in enumerate(dataset.items()):
        print(f"Evaluating category {idx} / {len(dataset)}: {key}")
        print(f"Category size: {len(value)}")
        for sample in value:
            question_id = build_question_id(key, sample)
            if question_id in completed_question_ids:
                continue

            processed += 1
            cur_data = {
                "idx": idx,
                "messages": build_messages(sample),
            }
            question_prompt = build_question_prompt(sample, model_name)
            correct_answer = sample["QA"]["ground_truth"]
            cur_data["messages"].append({"role": "user", "content": question_prompt})

            print(f"Processing sample {processed} / {total_samples}: {question_id}")
            print(f"Message count: {len(cur_data['messages'])}")
            
            if eval_method == "full":
                generated_answer = get_response(eval_model, cur_data["messages"])
            else:
                if isinstance(eval_model, EvoEmbeddingClient):
                    generated_answer = get_response_rag_ours(eval_model, cur_data["messages"], rag_sentence_num)
                else:
                    generated_answer = get_response_rag(
                        eval_model,
                        cur_data["messages"],
                        rag_sentence_num,
                        emb_model_name=embedding_model,
                    )
                    
            ans_text = str(generated_answer).strip()
            correct = 1 if (len(ans_text) > 0 and correct_answer == ans_text[0]) else 0            
            
            result.append({
                "question_id": question_id,
                "question": question_prompt,
                "answer": correct_answer,
                "generated_answer": generated_answer,
                "response": generated_answer,
                "correct": correct,
                "question_type": key,
            })
            completed_question_ids.add(question_id)
            print_metrics(result)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)