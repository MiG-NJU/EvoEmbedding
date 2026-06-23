import json
import os
import copy
from collections import defaultdict

from model.client import qwen3_client, OpenAIClient, EvoEmbeddingClient
from eval.longmemeval import get_response, get_response_rag, get_response_rag_ours

def build_messages_from_user_data(base_path, json_data_path, history_cache):
    relative_path = json_data_path.replace("./data/Persona-MME/", "")
    actual_path = os.path.join(base_path, relative_path)

    if actual_path in history_cache:
        return history_cache[actual_path]

    if not os.path.exists(actual_path):
        print(f"Warning: User data not found at {actual_path}")
        return []

    user_data = json.load(open(actual_path, "r"))
    messages = []
    
    for session in user_data.get("sessions", []):
        if not session:
            continue
        for interaction in session:
            time = interaction.get('time', '')
            user_msg = interaction.get('user', '')
            bot_msg = interaction.get('assistant', '')

            user_msg = user_msg.replace("<img>", "").strip()

            messages.extend([
                {
                    "role": "user",
                    "content": f"[{time}] {user_msg}"
                },
                {
                    "role": "assistant",
                    "content": bot_msg
                }
            ])
            
    history_cache[actual_path] = copy.deepcopy(messages)
    return history_cache[actual_path]

def check_result(correct_answer, response):
    """验证模型输出是否正确提取了选项字母"""
    if not response:
        return False
    target_char = correct_answer[1].lower() 
    try:
        pred_char = response.split(')')[0].lower()[-1]
        return target_char == pred_char
    except:
        return target_char in response.lower()

def load_existing_results(save_path):
    if not os.path.exists(save_path):
        return []
    try:
        result = json.load(open(save_path, "r"))
        return result
    except:
        return []

def print_final_metrics(results):
    """统计并打印结果，动态保存到最后一个结果中"""
    if not results:
        print("No results to evaluate.")
        return
        
    total_correct = 0
    total_count = 0
    total_correct_align = 0
    total_count_align = 0
    
    category_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
    statistics = {}

    for res in results:
        cat = res["question_type"]
        is_correct = res.get("correct", 0)
        
        category_metrics[cat]["total"] += 1
        category_metrics[cat]["correct"] += is_correct
        
        if "alignment" in cat:
            total_count_align += 1
            total_correct_align += is_correct
        else:
            total_count += 1
            total_correct += is_correct

    print("\n" + "="*65)
    print(f"{'Category':<35} | {'Count':<8} | {'Accuracy':<10}")
    print("-" * 65)
    for cat, metrics in sorted(category_metrics.items()):
        acc = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
        print(f"{cat:<35} | {metrics['total']:<8} | {acc:.2%}")
        statistics[cat] = acc

    print("-" * 65)
    overall_qa_acc = total_correct / total_count if total_count > 0 else 0
    align_acc = total_correct_align / total_count_align if total_count_align > 0 else 0
    
    total_all = total_count + total_count_align
    correct_all = total_correct + total_correct_align
    overall_all_acc = correct_all / total_all if total_all > 0 else 0

    print(f"{'Overall QA Results':<35} | {total_count:<8} | {overall_qa_acc:.2%}")
    print(f"{'Alignment Results':<35} | {total_count_align:<8} | {align_acc:.2%}")
    print(f"{'Total Combined Results':<35} | {total_all:<8} | {overall_all_acc:.2%}")
    print("="*65 + "\n")

    # 把统计结果挂在最后一条记录上，以便保存进 JSON
    statistics["overall_qa"] = overall_qa_acc
    statistics["alignment"] = align_acc
    statistics["overall_all"] = overall_all_acc
    statistics["Total QA Count"] = total_count
    statistics["Total Alignment Count"] = total_count_align
    results[-1]["statistics"] = statistics


def eval(eval_bench, eval_method, rag_sentence_num, save_path, model_name, embedding_model):
    base_data_path = os.getenv(
        "EVOEMBEDDING_PERSONAMME_DIR",
        "./data/benchmark/Persona-MME",
    )
    main_json_path = os.path.join(base_data_path, "Persona-MME.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if eval_bench == "PersonaMME32":
        sub_dirs = ['samples/50', 'samples/100']
    elif eval_bench == "PersonaMME128":
        sub_dirs = ['samples/200', 'samples/500']
    else:
        raise ValueError(f"Unknown eval_bench: {eval_bench}. Expected 'PersonaMME32' or 'PersonaMME128'.")

    allowed_data_paths = set()
    for sub_dir in sub_dirs:
        full_sub_dir = os.path.join(base_data_path, sub_dir)
        if os.path.exists(full_sub_dir):
            try:
                user_ids = sorted(os.listdir(full_sub_dir), key=lambda x: int(x))[:50]
            except ValueError:
                user_ids = sorted(os.listdir(full_sub_dir))[:50]
                
            for uid in user_ids:
                allowed_data_paths.add(f"./data/Persona-MME/{sub_dir}/{uid}/user_data.json")

    # 2. 读取并过滤 Benchmark
    all_benchmark_data = json.load(open(main_json_path, 'r'))
    filtered_benchmark = [q for q in all_benchmark_data if q.get('data_path') in allowed_data_paths]
    print(f"Loaded benchmark {eval_bench}. Filtered {len(filtered_benchmark)} questions from total {len(all_benchmark_data)}.")

    # 3. 准备模型客户端
    if "qwen" in model_name.lower():
        eval_model = qwen3_client(model_name)
    elif "evoembedding" in model_name.lower():
        eval_model = EvoEmbeddingClient(model_name=model_name)
    else:
        eval_model = OpenAIClient(model_name=model_name)

    results = load_existing_results(save_path)
    completed_ids = {item["question_id"] for item in results if "question_id" in item}
    history_cache = {}

    tasks = []
    for q_idx, question in enumerate(filtered_benchmark):
        q_type = f"{question['type']}_{question['question_type']}"
        q_time = question["time"].split('(')[0].strip()
        base_query = question["query"]
        choices = question.get("choices", {})
        answer = question.get("answer", "")
        data_path = question["data_path"]

        # 构建常规 QA
        supplex = "\nOptions:\n" + '\n'.join([f"{k}. {v}" for k, v in choices.items()]) + "\nPlease give your final answer (a), (b), (c) or (d)."
        qa_prompt = f"Question time: {q_time}\nQuestion: {base_query}\n{supplex}"
        # 加入 history hash 防重复或简单的区分id
        qa_id = f"QA_{q_idx}_{q_type}_{data_path.split('/')[-2]}"
        
        tasks.append({
            "id": qa_id,
            "q_type": q_type,
            "data_path": data_path,
            "prompt": qa_prompt,
            "answer": answer
        })

        # 构建 Alignment 的正例和反例 QA
        if "alignment" in question:
            for i in range(2):
                align_type = "chosen" if i == 0 else "rejected"
                align_answer = '(a)' if i == 0 else '(b)'
                align_text = question["alignment"][align_type]
                
                align_supplex = (
                    "Evaluate the following response based on the user's personality, which was revealed from the conversation history.\n"
                    f"\nResponse to Evaluate:\n\"{align_text}\"\n"
                    "\nQuestion:\nDoes the provided response align with and adapt to the user's personality?\n"
                    "\nOptions:\n(a) Yes, it aligns well.\n(b) No, it misaligns.\n"
                    "\nPlease provide only the final letter of your choice: (a) or (b)."
                )
                
                align_prompt = f"Question time: {q_time}\nQuestion: {base_query}\n{align_supplex}"
                align_id = f"Align_{q_idx}_{align_type}_{data_path.split('/')[-2]}"

                tasks.append({
                    "id": align_id,
                    "q_type": f"{q_type}_alignment", 
                    "data_path": data_path,
                    "prompt": align_prompt,
                    "answer": align_answer
                })

    print(f"Total flattened tasks to evaluate: {len(tasks)}")

    # 6. 开始遍历请求
    processed = len(completed_ids)
    for task in tasks:
        task_id = task["id"]
        if task_id in completed_ids:
            continue
            
        processed += 1
        print(f"\nProcessing {processed}/{len(tasks)}: {task_id}")

        # 获取当前对话的缓存上下文
        history_messages = build_messages_from_user_data(base_data_path, task["data_path"], history_cache)
        
        messages = copy.deepcopy(history_messages)
        messages.append({
            "role": "user",
            "content": task["prompt"]
        })

        # 调用具体的评测方式
        if eval_method == "full":
            generated_answer = get_response(eval_model, messages)
        else:
            if isinstance(eval_model, EvoEmbeddingClient):
                generated_answer = get_response_rag_ours(eval_model, messages, rag_sentence_num)
            else:
                generated_answer = get_response_rag(eval_model, messages, rag_sentence_num, emb_model_name=embedding_model)

        ans_text = str(generated_answer).strip()
        is_correct = 1 if check_result(task["answer"], ans_text) else 0

        # 保存并写入进度
        results.append({
            "question_id": task_id,
            "question_type": task["q_type"],
            "data_path": task["data_path"],
            "prompt": task["prompt"],
            "ground_truth": task["answer"],
            "generated_answer": generated_answer,
            "response": generated_answer,
            "correct": is_correct
        })
        completed_ids.add(task_id)
        print_final_metrics(results) 
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


