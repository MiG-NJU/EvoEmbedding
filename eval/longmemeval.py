import json
import os
import faiss
from rank_bm25 import BM25Okapi
import numpy as np
import jieba
from collections import defaultdict

from model.client import qwen3_client, OpenAIClient, get_text_embedding, EvoEmbeddingClient


def retrieve_by_faiss(query_text, turn_texts, k, emb_model_name):
    query_vec = get_text_embedding(query_text, emb_model_name, is_query=True)
    corpus_embeddings = get_text_embedding(turn_texts, emb_model_name)

    if len(query_vec.shape) == 1: 
        query_vec = query_vec.reshape(1, -1)
    
    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings.astype('float32'))
    distances, indices = index.search(query_vec.astype('float32'), k)
    return indices[0]

def retrieve_by_grep(query_text, turn_texts, k, model):
    keyword_prompt = (
        "Please extract the core search keywords from the following query. "
        "Return ONLY the keywords separated by spaces, without any extra text or explanation.\n"
        f"Query: {query_text}"
    )
    kw_response = model.send_message([{"role": "user", "content": keyword_prompt}])

    def tokenize(text):
        text = text.lower()
        return jieba.lcut(text)

    tokenized_corpus = [tokenize(doc) for doc in turn_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenize(kw_response)
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = np.argsort(scores)[::-1][:k]
    return top_k_indices


def get_response(model, messages):
    return model.send_message(messages)

def get_evaluate_result(model, prompt):
    return model.send_message([{"role": "user", "content": prompt}])

def get_response_rag(model, messages, rag_sentence_num, emb_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    query_message = messages[-1]
    history_messages = messages[:-1]
    
    turn_chunks = []
    turn_texts = []
    
    i = 0
    while i < len(history_messages):
        current_msg = history_messages[i]
        if current_msg["role"] == "user" and i + 1 < len(history_messages) and history_messages[i+1]["role"] == "assistant":
            user_msg = current_msg
            asst_msg = history_messages[i+1]
            combined_text = f"User: {user_msg['content']}\nAssistant: {asst_msg['content']}"
            turn_chunks.append([user_msg, asst_msg])
            turn_texts.append(combined_text)
            i += 2
        else:
            role_label = "User" if current_msg["role"] == "user" else "Assistant"
            turn_chunks.append([current_msg])
            turn_texts.append(f"{role_label}: {current_msg['content']}")
            i += 1
    
    k = max(1, min(rag_sentence_num, len(turn_texts)))
    if emb_model_name == "grep":
        retrieved_indices = retrieve_by_grep(query_message['content'], turn_texts, k, model)
    else:
        retrieved_indices = retrieve_by_faiss(query_message['content'], turn_texts, k, emb_model_name)
    retrieved_indices = sorted(retrieved_indices)    

    final_context_messages = []
    # print(f"[{emb_model_name}] Retrieved indices: ", retrieved_indices)
    
    for idx in retrieved_indices:
        if idx != -1 and idx < len(turn_chunks):
            final_context_messages.extend(turn_chunks[idx])
    final_messages = final_context_messages + [query_message]
    return model.send_message(final_messages)


def get_response_rag_ours(model, messages, rag_sentence_num):
    query_message = messages[-1]
    history_messages = messages[:-1]

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

    retrieved_indices = None
    final_context_messages = []

    paired_messages_for_retrieve = aligned_history_messages + [query_message]
    retrieved_indices = model.send_message_retrieve(paired_messages_for_retrieve, rag_sentence_num)
    retrieved_indices = [idx for idx in retrieved_indices if isinstance(idx, int) and 0 <= idx < len(turn_chunks)]

    for idx in retrieved_indices:
        if idx != -1 and idx < len(turn_chunks):
            final_context_messages.extend(turn_chunks[idx])

    final_context_messages = [
        msg for msg in final_context_messages
        if not (msg.get("content", "") == "" and msg.get("role") in ("user", "assistant"))
    ]

    final_messages = final_context_messages + [query_message]
    return model.send_message_raw(final_messages)


def print_metrics(result_list):
    if not result_list:
        print("No results to evaluate.")
        return
    statistics = {}
    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    total_correct = 0
    total_count = len(result_list)

    for item in result_list:
        is_correct = item.get("correct", 0)
        q_type = item.get("question_type", "unknown")
        total_correct += is_correct
        stats[q_type]["total"] += 1
        stats[q_type]["correct"] += is_correct

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
    print("="*65 + "\n")
    statistics["overall"] = overall_acc
    statistics["Total_Count"] = total_count
    result_list[-1]["statistics"] = statistics
    return statistics
     

def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response) 
    return prompt


def true_or_false(response):
    if response is None:
        return False
    normalized = str(response).strip().lower()
    if not normalized:
        return False
    first_line = normalized.splitlines()[0].strip()
    tokens = first_line.replace('.', '').replace('!', '').replace(':', '').replace(';', '').split()
    if not tokens:
        return False
    head = tokens[0]
    if head in ("yes", "y"):
        return True
    if head in ("no", "n"):
        return False
    if "yes" in first_line:
        return True
    if "no" in first_line:
        return False
    return False


def eval(DATA_PATH, eval_method, rag_sentence_num, save_path, model_name, embedding_model):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = json.load(open(DATA_PATH, "r"))
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
    result = result[:-1]
    for idx, item in enumerate(data):
        print(f"Evaluating {idx} / {len(data)}")
        if idx < len(result):
            continue
        cur_data = {}
        cur_data["idx"] = idx
        cur_data["messages"] = []
        sessions = item["haystack_sessions"]
        timestamps = item["haystack_dates"]

        for session, timestamp in zip(sessions, timestamps):
            for msg in session:
                content = msg["content"]
                if msg["role"] == "user":
                    content = f"{timestamp}\n{content}"
                cur_data["messages"].append({
                    "role": msg["role"], 
                    "content": content
                })

        question_prompt = (
                "Please answer the following question based on the history of conversations:\n"
                f"Question time: {item['question_date']} and question: {item['question']}"
            )
        cur_data["messages"].extend([
            {"role": "user", "content": question_prompt}
        ])
        if eval_method == "full":
            generated_answer = get_response(eval_model, cur_data["messages"])
        else:
            if isinstance(eval_model, EvoEmbeddingClient):
                generated_answer = get_response_rag_ours(eval_model, cur_data["messages"], rag_sentence_num)
            else:
                generated_answer = get_response_rag(eval_model, cur_data["messages"], rag_sentence_num, emb_model_name=embedding_model)

        if 'abs' in item["question_id"]:
            prompt = get_anscheck_prompt(
                item["question_type"], item["question"], item["answer"], generated_answer, abstention=True
            )
        else:
            prompt = get_anscheck_prompt(
                item["question_type"], item["question"], item["answer"], generated_answer
            )
        response = get_evaluate_result(llm_judge, prompt)
        correct = 1 if true_or_false(response) else 0
        result.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "answer": item["answer"],
            "generated_answer": generated_answer,
            "response": response,
            "correct": correct,
            "question_type": item["question_type"],
        })
        print_metrics(result)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

