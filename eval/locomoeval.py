import json
import os
import faiss, copy
import numpy as np

from model.client import qwen3_client, OpenAIClient, EvoEmbeddingClient
from eval.longmemeval import get_response, get_evaluate_result, true_or_false, print_metrics, retrieve_by_faiss, retrieve_by_grep


def get_response_rag(model, messages, rag_sentence_num, emb_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    query_message = messages[-1]
    history_messages = messages[:-1]  
    
    keep_indices = {0}
    
    turn_chunks = []
    turn_texts = []
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
    return model.send_message(final_messages)


def get_response_rag_ours(model, messages, rag_sentence_num):
    query_message = messages[-1]
    history_messages = messages[:-1]

    keep_indices = set()
    if len(history_messages) > 0:
        keep_indices.add(0)

    fixed_messages = [history_messages[i] for i in sorted(keep_indices)]
    retrieval_history = [msg for i, msg in enumerate(history_messages) if i not in keep_indices]

    aligned_history_messages = []
    i = 0
    while i < len(retrieval_history):
        cur = retrieval_history[i]
        if cur["role"] == "user":
            if i + 1 < len(retrieval_history) and retrieval_history[i + 1]["role"] == "assistant":
                aligned_history_messages.extend([cur, retrieval_history[i + 1]])
                i += 2
            else:
                aligned_history_messages.extend([cur, {"role": "assistant", "content": ""}])
                i += 1
        else:
            aligned_history_messages.extend([{"role": "user", "content": ""}, cur])
            i += 1

    if not aligned_history_messages:
        return model.send_message_raw(fixed_messages + [query_message])

    turn_chunks = []
    for i in range(0, len(aligned_history_messages), 2):
        turn_chunks.append([aligned_history_messages[i], aligned_history_messages[i + 1]])

    paired_messages_for_retrieve = aligned_history_messages + [query_message]
    retrieved_indices = model.send_message_retrieve(paired_messages_for_retrieve, rag_sentence_num)
    retrieved_indices = [idx for idx in retrieved_indices if isinstance(idx, int) and 0 <= idx < len(turn_chunks)]

    final_context_messages = []
    for idx in retrieved_indices:
        if idx != -1 and idx < len(turn_chunks):
            final_context_messages.extend(turn_chunks[idx])

    if not final_context_messages:
        keep_pair_num = min(rag_sentence_num, max(1, len(turn_chunks)))
        final_context_messages = aligned_history_messages[-2 * keep_pair_num:]

    final_context_messages = [
        msg for msg in final_context_messages
        if not (msg.get("content", "") == "" and msg.get("role") in ("user", "assistant"))
    ]

    final_messages = fixed_messages + final_context_messages + [query_message]
    return model.send_message_raw(final_messages)



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
    answer_model_name = getattr(eval_model, "model_name", model_name)
    print(f"Answer / memory inference model: {answer_model_name}")
    print("Evaluation judge model: gpt-4o-mini")
    result = result[:-1]
    question_id_num = 0
    for idx, item in enumerate(data):
        print(f"Evaluating {idx} / {len(data)}")
        cur_data = {}
        cur_data["idx"] = idx
        cur_data["messages"] = []
        speaker_a, speaker_b = item["conversation"]["speaker_a"], item["conversation"]["speaker_b"]
        cur_data["messages"].append({"role": "user", "content": f"You are role-playing as {speaker_b} in a conversation with the user is playing is  {speaker_a}."})

        num_sessions = len(item["session_summary"])
        for session_idx in range(num_sessions):
            session = item["conversation"][f"session_{session_idx+1}"]
            session_time = item["conversation"][f"session_{session_idx+1}_date_time"]
            for dialogue in session:
                text = dialogue['text']
                if "blip_caption" in dialogue and dialogue["blip_caption"]:
                    text = f"{text} (image description: {dialogue['blip_caption']})"

                if dialogue["speaker"] == speaker_a:
                    cur_data["messages"].append({"role": "user", "content": f"{session_time} {speaker_a}\n{text}"})
                elif dialogue["speaker"] == speaker_b:
                    cur_data["messages"].append({"role": "assistant", "content": f"{speaker_b}\n{text}"})
        
        for i, qa in enumerate(item["qa"]):
            print(f"Evaluating {idx} / {len(data)}, {i} / {len(item['qa'])}")
            if str(qa["category"]) == "5":
                continue
            question_id_num += 1
            if len(result) >= question_id_num:
                continue
            question = qa["question"]
            category = qa["category"]
            if "answer" not in qa:
                answer = qa["adversarial_answer"]
            else:
                answer = qa["answer"]
            question_prompt = (
                f"Any content referring to 'User' in the prompt refers to {speaker_a}'s content, and any content referring to 'AI'or 'assiant' refers to {speaker_b}'s content."
                f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
                f"Question: {question}"
            )
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
            
            eval_prompt= """
Your task is to label an answer to a question as `Yes` or `No`. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a `gold` (ground truth) answer, 
    (3) a generated answer
which you will score as Yes/No.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as Yes. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as Yes. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it Yes if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Is the generated answer correct? Answer yes or no only.
"""
            eval_prompt = eval_prompt.format(question=question, gold_answer=answer, generated_answer=generated_answer)

            response = get_evaluate_result(llm_judge, eval_prompt)
            correct = 1 if true_or_false(response) else 0
            result.append({
                "question_id": question_id_num,
                "question": question,
                "answer": answer,
                "generated_answer": generated_answer,
                "response": response,
                "correct": correct,
                "question_type": category,
            })
            statistics = print_metrics([x for x in result if str(x.get("question_type")) != "5"])
            result[-1]["statistics"] = statistics 
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)


