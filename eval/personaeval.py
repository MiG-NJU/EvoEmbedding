import json
import os
import faiss, copy, math
import numpy as np
import random
from collections import defaultdict
random.seed(42)

from model.client import qwen3_client, OpenAIClient, EvoEmbeddingClient
from eval.longmemeval import get_response, get_evaluate_result, true_or_false, retrieve_by_faiss, retrieve_by_grep


def get_response_rag(model, messages, rag_sentence_num, emb_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    query_message = messages[-1]      
    history_messages = messages[:-1]  
    
    keep_indices = set()
    
    turn_chunks = []
    turn_texts = []
    search_candidate_indices = []

    i = 0
    while i < len(history_messages):
        if i in keep_indices:
            i += 1
            continue
            
        if (history_messages[i]['role'] == 'user' and 
            i + 1 < len(history_messages) - 1 and 
            history_messages[i+1]['role'] == 'assistant'):
            
            chunk = [history_messages[i], history_messages[i+1]]
            text = f"User: {history_messages[i]['content']}\nAssistant: {history_messages[i+1]['content']}"
            idx_list = [i, i+1]
            i += 2
        else:
            chunk = [history_messages[i]]
            role_label = "User" if history_messages[i]['role'] == 'user' else "Assistant"
            text = f"{role_label}: {history_messages[i]['content']}"
            idx_list = [i]
            i += 1
            
        turn_chunks.append(chunk)
        turn_texts.append(text)
        search_candidate_indices.append(idx_list)

    if not turn_texts:
        return model.send_message([history_messages[i] for i in sorted(list(keep_indices))] + [query_message])

    k = max(1, min(rag_sentence_num, len(turn_texts)))
    if emb_model_name == "grep":
        retrieved_indices = retrieve_by_grep(query_message['content'], turn_texts, k, model)
    else:
        retrieved_indices = retrieve_by_faiss(query_message['content'], turn_texts, k, emb_model_name)
    indices = sorted(retrieved_indices)    

    for relative_idx in indices:
        if relative_idx != -1:
            for raw_idx in search_candidate_indices[relative_idx]:
                keep_indices.add(raw_idx)

    final_messages = [history_messages[i] for i in sorted(list(keep_indices))] + [query_message]
    return model.send_message(final_messages), [int(i) for i in indices if i != -1]

def get_response_rag_ours(model, messages, rag_sentence_num):
    query_message = messages[-1]
    history_messages = messages[:-1]
    keep_indices = set()
    
    turn_chunks = []
    search_candidate_indices = []
    i = 0
    while i < len(history_messages):
        if i in keep_indices:
            i += 1
            continue
        if (history_messages[i]['role'] == 'user' and 
            i + 1 < len(history_messages) and 
            history_messages[i+1]['role'] == 'assistant'):
            
            chunk = [history_messages[i], history_messages[i+1]]
            idx_list = [i, i+1]
            i += 2
        else:
            # 否则单条消息作为一个 chunk
            chunk = [history_messages[i]]
            idx_list = [i]
            i += 1
            
        turn_chunks.append(chunk)
        search_candidate_indices.append(idx_list)
    messages_for_retrieve = []
    for chunk in turn_chunks:
        messages_for_retrieve.extend(chunk)
    messages_for_retrieve.append(query_message)
    retrieved_indices = model.send_message_retrieve(messages_for_retrieve, rag_sentence_num, _sorted=False)
    retrieved_indices = [idx for idx in retrieved_indices if isinstance(idx, int) and 0 <= idx < len(turn_chunks)]
    indices = copy.deepcopy(retrieved_indices)
    for relative_idx in retrieved_indices:
        if relative_idx != -1:
            for raw_idx in search_candidate_indices[relative_idx]:
                keep_indices.add(raw_idx)

    final_messages = [history_messages[i] for i in sorted(list(keep_indices))] + [query_message]
    return model.send_message_raw(final_messages), indices


def get_response_rag_memory(model, query, memories, rag_sentence_num, emb_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    if not memories:
        prompt = f"{query}"
        return model.send_message([{"role": "user", "content": prompt}]), []

    k = max(1, min(rag_sentence_num, len(memories)))
    if emb_model_name == "grep":
        retrieved_indices = retrieve_by_grep(query, memories, k, model)
    else:
        retrieved_indices = retrieve_by_faiss(query, memories, k, emb_model_name)
    indices = sorted(retrieved_indices)    
    retrieved_indices = [int(i) for i in indices if i != -1]
    retrieved_memories = [memories[i] for i in retrieved_indices]
    
    prompt = f"Background Information:\n" + "\n".join(retrieved_memories) + f"\n\nQuestion: {query}"
    return model.send_message([{"role": "user", "content": prompt}]), retrieved_indices


def get_response_rag_ours_memory(model, query, memories, rag_sentence_num, only_topk=False, native=False):
    print(f"Using evoembedding memory retrieval, memories count: {len(memories)}, rag_sentence_num: {rag_sentence_num}")
    if not memories:
        prompt = f"{query}"
        return model.send_message_raw([{"role": "user", "content": prompt}]), []

    # For evoembedding, we format memories as assistant messages or just chunks
    messages_for_retrieve = []
    for mem in memories:
        messages_for_retrieve.extend([{"role": "user", "content": mem}, {"role": "assistant", "content": ""}])
    messages_for_retrieve.append({"role": "user", "content": query})

    retrieved_indices = model.send_message_retrieve(messages_for_retrieve, rag_sentence_num, _sorted=False, native=native)
    retrieved_indices = [idx for idx in retrieved_indices if isinstance(idx, int) and 0 <= idx < len(memories)]
    if only_topk:
        return None, retrieved_indices
    retrieved_memories = [memories[i] for i in retrieved_indices]

    prompt = f"Background Information:\n" + "\n".join(retrieved_memories) + f"\n\nQuestion: {query}"
    return model.send_message_raw([{"role": "user", "content": prompt}]), retrieved_indices


def print_metrics(result_list):
    if not result_list:
        print("No results to evaluate.")
        return
    statistics = {}
    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    total_correct = 0
    total_count = len(result_list)

    total_correct_ex = 0
    total_count_ex = 0

    for item in result_list:
        is_correct = item.get("correct", 0)
        q_type = item.get("question_type", "unknown")

        # 类别合并 (忽略大小写进行匹配)
        q_type_lower = q_type.lower()
        if "semantic analogy" in q_type_lower:
            q_type = "Analogy"
        elif "state dynamic" in q_type_lower:
            q_type = "Dynamics"

        total_correct += is_correct
        stats[q_type]["total"] += 1
        stats[q_type]["correct"] += is_correct

        if q_type not in ["Analogy", "Causality"]:
            total_correct_ex += is_correct
            total_count_ex += 1

    print("\n" + "="*65)
    print(f"{'Question Type':<35} | {'Count':<8} | {'Accuracy':<10}")
    print("-" * 65)
    for q_type, metrics in sorted(stats.items()):
        acc = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
        print(f"{q_type:<35} | {metrics['total']:<8} | {acc:.2%}")
        statistics[q_type] = acc
    print("-" * 65)
    overall_acc = total_correct / total_count if total_count > 0 else 0
    print(f"{'OVERALL':<35} | {total_count:<8} | {overall_acc:.2%}")
    
    overall_exclude = total_correct_ex / total_count_ex if total_count_ex > 0 else 0
    print(f"{'OVERALL (Excl. Analogy/Causality)':<35} | {total_count_ex:<8} | {overall_exclude:.2%}")
    print("="*65 + "\n")
    
    statistics["overall"] = overall_acc
    statistics["Total_Count"] = total_count

    statistics["overall_exclude"] = overall_exclude
    statistics["total_exclude"] = total_count_ex

    result_list[-1]["statistics"] = statistics
    return statistics


def eval(DATA_PATH, eval_method, rag_sentence_num, save_path, model_name, embedding_model, eval_memory):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".json")]
    all_files = sorted(all_files)
    result = []
    if os.path.exists(save_path):
        result = json.load(open(save_path, "r"))
    if "qwen" in model_name.lower():
        eval_model = qwen3_client(model_name)
    elif "evoembedding" in model_name.lower():
        eval_model = EvoEmbeddingClient(model_name=model_name)
    else:
        eval_model = OpenAIClient(model_name=model_name)
    llm_judge = OpenAIClient(model_name="gpt-4o-mini")
    if len(result) > 0:
        result = result[:-1]
    for idx, item in enumerate(all_files):
        print(f"Evaluating {idx} / {len(all_files)}", item)
        if len(result) > idx:
            continue
        
        if eval_memory:
            samlpe_data = json.load(open(item, "r"))
            query = samlpe_data["meta_v3"]["query"]
            
            pos_list = samlpe_data["meta_v3"]["pos"]
            negs_list = samlpe_data["meta_v3"]["negs"]
            random.shuffle(negs_list)
            
            total_len = len(pos_list) + len(negs_list)
            pos_indices = sorted(random.sample(range(total_len), len(pos_list)))
            
            memories = []
            evidences = []
            negs_iter = iter(negs_list)
            pos_iter = iter(pos_list)
            
            for i in range(total_len):
                if i in pos_indices:
                    memories.append(next(pos_iter))
                    evidences.append(i)
                else:
                    memories.append(next(negs_iter))
            
            if eval_method == "full":
                prompt = "Retrieved Memories:\n" + "\n".join(memories) + f"\n\nQuestion: {query}"
                generated_answer = eval_model.send_message([{"role": "user", "content": prompt}])
            else:
                if isinstance(eval_model, EvoEmbeddingClient):
                    generated_answer, indices = get_response_rag_ours_memory(eval_model, query, memories, rag_sentence_num)
                else:
                    generated_answer, indices = get_response_rag_memory(eval_model, query, memories, rag_sentence_num, embedding_model)
        else:
            samlpe_data = json.load(open(item, "r"))
            cur_data = {}
            cur_data["idx"] = samlpe_data["id"]
            cur_data["messages"] = samlpe_data["messages"][:-1]

            if eval_method == "full":
                generated_answer = get_response(eval_model, cur_data["messages"])
            else:
                if isinstance(eval_model, EvoEmbeddingClient):
                    generated_answer, indices = get_response_rag_ours(eval_model, cur_data["messages"], rag_sentence_num)
                else:
                    generated_answer, indices = get_response_rag(eval_model, cur_data["messages"], rag_sentence_num, emb_model_name=embedding_model)
                evidences = samlpe_data["meta"]["evidence_turns"]

        eval_prompt= """
Evaluate whether the generated answer is correct based on the gold answer for the given question.
Answer `Yes` or `No` only.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}
"""
        question = samlpe_data["messages"][-2]["content"]
        answer = samlpe_data["messages"][-1]["content"]
        category = samlpe_data["meta_q"]["target_type"]
        eval_prompt = eval_prompt.format(question=question, gold_answer=answer, generated_answer=generated_answer)

        response = get_evaluate_result(llm_judge, eval_prompt)
        correct = 1 if true_or_false(response) else 0
        result.append({
            "idx": idx,
            "path": item,
            "question": question,
            "answer": answer,
            "generated_answer": generated_answer,
            "response": response,
            "correct": correct,
            "question_type": category,
            "meta": samlpe_data["meta_q"],
        })
        statistics = print_metrics(result)
        result[-1]["statistics"] = statistics
        if eval_method != "full":
            result[-1]["indices"] = indices
            result[-1]["evidences"] = evidences
            result[-1]["retrieve_result"] = retrieve_result(evidences, indices)
            result[-1]["statistics"]["recall"] = sum([entry["retrieve_result"][0] for entry in result if "retrieve_result" in entry]) / len(result)
            result[-1]["statistics"]["ndcg"] = sum([entry["retrieve_result"][1] for entry in result if "retrieve_result" in entry]) / len(result)
            result[-1]["statistics"]["recall_exclude"] = sum([entry["retrieve_result"][0] for entry in result if "retrieve_result" in entry and entry["question_type"] not in ["Analogy", "Causality"]]) / sum([1 for entry in result if "retrieve_result" in entry and entry["question_type"] not in ["Analogy", "Causality"]])
            result[-1]["statistics"]["ndcg_exclude"] = sum([entry["retrieve_result"][1] for entry in result if "retrieve_result" in entry and entry["question_type"] not in ["Analogy", "Causality"]]) / sum([1 for entry in result if "retrieve_result" in entry and entry["question_type"] not in ["Analogy", "Causality"]])
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

def retrieve_result(evidences, indices):
    assert evidences is not None and isinstance(evidences, list) and len(evidences) > 0
    assert indices is not None and isinstance(indices, list) and len(indices) > 0

    matched = set(evidences) & set(indices)
    recall = len(matched) / len(evidences) if len(evidences) > 0 else 0.0

    dcg = 0.0
    for pos, idx in enumerate(indices):
        if idx in evidences:
            dcg += 1.0 / math.log2(pos + 2)

    idcg = 0.0
    ideal_hits = min(len(indices), len(evidences))
    for pos in range(ideal_hits):
        idcg += 1.0 / math.log2(pos + 2)
    ndcg = dcg / idcg if idcg > 0.0 else 0.0
    return recall, ndcg
