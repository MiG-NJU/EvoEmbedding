import json
import os, csv
import copy
import re

from model.client import qwen3_client, OpenAIClient, EvoEmbeddingClient
from eval.longmemeval import get_response, get_evaluate_result, print_metrics, retrieve_by_faiss, retrieve_by_grep


def load_rows(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        res = []
        for row_number, row in enumerate(reader, start=1):
            row_data = {}
            for column_name, value in row.items():
                row_data[column_name] = value
            res.append(row_data)
    return res

# 【修改2】补充 emb_model_name 参数
def get_response_rag(model, messages, rag_sentence_num, emb_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    query_message = messages[-1]
    history_messages = messages[:-1]
    if len(history_messages) <= rag_sentence_num:
        return model.send_message(messages)

    aligned_history_messages = []
    i = 0
    while i < len(history_messages):
        cur = history_messages[i]
        if cur["role"] == "user":
            if i + 1 < len(history_messages) and history_messages[i + 1]["role"] == "assistant":
                aligned_history_messages.extend([cur, history_messages[i + 1]])
                i += 2
            else:
                aligned_history_messages.extend([cur, {"role": "assistant", "content": ""}])
                i += 1
        else:
            aligned_history_messages.extend([{"role": "user", "content": ""}, cur])
            i += 1

    turn_chunks = []
    for i in range(0, len(aligned_history_messages), 2):
        turn_chunks.append([aligned_history_messages[i], aligned_history_messages[i + 1]])

    keep_chunks = {0, len(turn_chunks) - 1}
    search_candidates = []
    search_candidate_indices = []
    for idx, chunk in enumerate(turn_chunks):
        if idx not in keep_chunks:
            user_msg, asst_msg = chunk
            if not user_msg.get("content"):
                text = f"Assistant: {asst_msg['content']}"
            elif not asst_msg.get("content"):
                text = f"User: {user_msg['content']}"
            else:
                text = f"User: {user_msg['content']}\nAssistant: {asst_msg['content']}"
            search_candidates.append(text)
            search_candidate_indices.append(idx)

    if not search_candidates:
         final_context_messages = []
         for idx in sorted(list(keep_chunks)):
             final_context_messages.extend(turn_chunks[idx])
         final_context_messages = [
             msg for msg in final_context_messages
             if not (msg.get("content", "") == "" and msg.get("role") in ("user", "assistant"))
         ]
         final_messages = final_context_messages + [query_message]
         return model.send_message(final_messages)

    k = max(1, min(rag_sentence_num, len(search_candidates)))
    if emb_model_name == "grep":
        retrieved_indices = retrieve_by_grep(query_message['content'], search_candidates, k, model)
    else:
        retrieved_indices = retrieve_by_faiss(query_message['content'], search_candidates, k, emb_model_name)
    indices = sorted(retrieved_indices)

    for relative_idx in indices:
        if relative_idx != -1:
            chunk_idx = search_candidate_indices[relative_idx]
            keep_chunks.add(chunk_idx)

    final_context_messages = []
    for idx in sorted(list(keep_chunks)):
        final_context_messages.extend(turn_chunks[idx])
    final_context_messages = [
        msg for msg in final_context_messages
        if not (msg.get("content", "") == "" and msg.get("role") in ("user", "assistant"))
    ]
    final_messages = final_context_messages + [query_message]
    return model.send_message(final_messages)


def get_response_rag_ours(model, messages, rag_sentence_num):
    query_message = messages[-1]
    history_messages = messages[:-1]

    if len(history_messages) <= rag_sentence_num:
         return model.send_message_raw(messages)

    aligned_history_messages = []
    i = 0
    while i < len(history_messages):
        cur = history_messages[i]
        if cur["role"] == "user":
            if i + 1 < len(history_messages) and history_messages[i + 1]["role"] == "assistant":
                aligned_history_messages.extend([cur, history_messages[i + 1]])
                i += 2
            else:
                aligned_history_messages.extend([cur, {"role": "assistant", "content": ""}])
                i += 1
        else:
            aligned_history_messages.extend([{"role": "user", "content": ""}, cur])
            i += 1

    turn_chunks = []
    for i in range(0, len(aligned_history_messages), 2):
        user_msg = aligned_history_messages[i]
        asst_msg = aligned_history_messages[i + 1]
        turn_chunks.append([user_msg, asst_msg])

    keep_chunks = {0, len(turn_chunks) - 1}
    messages_for_retrieve = []
    search_candidate_indices = []
    for idx, chunk in enumerate(turn_chunks):
        if idx in keep_chunks:
            continue
        messages_for_retrieve.extend(chunk)
        search_candidate_indices.append(idx)

    if messages_for_retrieve:
        messages_for_retrieve.append(query_message)
        retrieved_indices = model.send_message_retrieve(messages_for_retrieve, rag_sentence_num)

        for relative_idx in retrieved_indices:
            if isinstance(relative_idx, int) and 0 <= relative_idx < len(search_candidate_indices):
                keep_chunks.add(search_candidate_indices[relative_idx])

    final_context_messages = []
    for idx in sorted(list(keep_chunks)):
        final_context_messages.extend(turn_chunks[idx])

    final_context_messages = [
        msg for msg in final_context_messages
        if not (msg.get("content", "") == "" and msg.get("role") in ("user", "assistant"))
    ]

    final_messages = final_context_messages + [query_message]
    return model.send_message_raw(final_messages)


def get_dataset(DATA_PATH):
    if "personamem32" in DATA_PATH:
        path = "./data/benchmark/personaMem/shared_contexts_32k.jsonl"
        questions_path = "./data/benchmark/personaMem/questions_32k.csv"
    else:
        path = "./data/benchmark/personaMem/shared_contexts_128k.jsonl"
        questions_path = "./data/benchmark/personaMem/questions_128k.csv"
    dataset = {}
    with open(path, "r") as f:
        samples = [json.loads(line) for line in f]
    for sample in samples:
        key = list(sample.keys())[0]
        dataset[key] = {
            "conv": sample[key]
        }
    questions_data = load_rows(questions_path)
    for q in questions_data:
        if q["shared_context_id"] not in dataset.keys():
            continue
        current_questions = dataset[q["shared_context_id"]].get("question", [])
        current_questions.append(q)
        dataset[q["shared_context_id"]]["question"] = current_questions
    question_number = 0
    for key in dataset.keys():
        if "question" not in dataset[key].keys():
            continue
        dataset[key]["question"].sort(key=lambda x: int(x.get("end_index_in_shared_context", 0)))
        question_number += len(dataset[key]["question"])
        print(key, len(dataset[key]["question"]), [q["end_index_in_shared_context"] for q in dataset[key]["question"]])
    print(f"Total questions: {question_number}")
    return dataset

def extract_answer(text):
    if not text or not isinstance(text, str):
        return ""
    return text.split(')')[0].lower()[-1]

def check_result(gt, pred):
    if not pred or not isinstance(pred, str):
        return False
    if gt.split(")")[0].strip().lower()[-1] == pred:
        return True
    return False


def eval(DATA_PATH, eval_method, rag_sentence_num, save_path, model_name, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    dataset = get_dataset(DATA_PATH)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result = []
    if os.path.exists(save_path):
        result = json.load(open(save_path, "r"))
        if result and isinstance(result[-1], dict) and "statistics" in result[-1]:
            result = result[:-1]

    if "qwen" in model_name.lower():
        eval_model = qwen3_client(model_name)
    elif "evoembedding" in model_name.lower():
        eval_model = EvoEmbeddingClient(model_name=model_name)
    else:
        eval_model = OpenAIClient(model_name=model_name)
        
    question_id_num = 0
    
    for idx, (key, value) in enumerate(dataset.items()):
        if "question" not in dataset[key].keys():
            continue
        print(f"Evaluating {idx} / {len(dataset)}")
        cur_data = {}
        cur_data["idx"] = idx
        cur_data["messages"] = []
        cur_data["messages"].append({"role": "user", "content": str(value["conv"][0]["content"])})
        
        for idx_ in range(1, len(value["conv"])):
            if value["conv"][idx_]["role"] == "system":
                continue
            if value["conv"][idx_]["role"] == "user":
                cur_data["messages"].append({"role": "user", "content": value["conv"][idx_]["content"].replace("User:", "").strip()})
            if value["conv"][idx_]["role"] == "assistant":
                cur_data["messages"].append({"role": "assistant", "content": value["conv"][idx_]["content"].replace("Assistant:", "").strip()})
                
        for i, qa in enumerate(value["question"]):
            print(f"Evaluating {idx} / {len(dataset)}, {i} / {len(value['question'])}")
            question_id_num += 1
            if len(result) >= question_id_num:
                continue
            question_type = qa["question_type"]
            topic = qa["topic"]
            user_question_or_message = qa["user_question_or_message"]
            correct_answer = qa["correct_answer"]
            all_options = qa["all_options"]
            question_prompt = user_question_or_message + "\nOptions:\n" + all_options + "\nFind the most appropriate model response and give your final answer (a), (b), (c), or (d)."                
            
            copy_cur_data = copy.deepcopy(cur_data)   
            copy_cur_data["messages"].extend([
                {"role": "user", "content": question_prompt}
            ])
            if eval_method == "full":
                generated_answer = get_response(eval_model, copy_cur_data["messages"])
            else:
                if isinstance(eval_model, EvoEmbeddingClient):
                    generated_answer = get_response_rag_ours(eval_model, copy_cur_data["messages"], rag_sentence_num)
                else:
                    generated_answer = get_response_rag(eval_model, copy_cur_data["messages"], rag_sentence_num, emb_model_name=embedding_model)

            response = extract_answer(generated_answer)
            correct = 1 if check_result(correct_answer, response) else 0

            result.append({
                "question_id": question_id_num,
                "question": question_prompt,
                "answer": correct_answer,
                "generated_answer": generated_answer,
                "response": response,
                "correct": correct,
                "question_type": question_type,
                "topic": topic,
            })
            print_metrics(result)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            # If evaluating the 128k dataset, stop when we've collected 1000 results
            if "128" in DATA_PATH and len(result) >= 1000:
                print("Reached 1000 results for 128k dataset, stopping early.")
                return


