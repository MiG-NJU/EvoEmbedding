import argparse

from eval import longmemeval, locomoeval, personamemeval, personamme

def eval(eval_bench, eval_method, rag_sentence_num, save_path, model_name, embedding_model, eval_memory):
    if eval_bench == "longmemeval_s":
        DATA_PATH = './data/benchmark/longmemeval_s.json'
        longmemeval.eval(DATA_PATH, eval_method, rag_sentence_num, save_path, model_name, embedding_model)
    elif eval_bench == "locomo":
        DATA_PATH = './data/benchmark/locomo10.json'
        locomoeval.eval(DATA_PATH, eval_method, rag_sentence_num, save_path, model_name, embedding_model)
    elif eval_bench == "personamem32":
        personamemeval.eval(eval_bench, eval_method, rag_sentence_num, save_path, model_name, embedding_model)
    elif eval_bench == "PersonaMME32" or eval_bench == "PersonaMME128":
        personamme.eval(eval_bench, eval_method, rag_sentence_num, save_path, model_name, embedding_model)
    else:
        raise ValueError("Invalid eval_bench")

def parse_args():
    parser = argparse.ArgumentParser(description="Long Memory Evaluation Script")
    parser.add_argument("--eval_method", type=str, default="rag", choices=["full", "rag"],
                        help="Evaluation method: 'full' for full context, 'rag' for RAG-based retrieval.")
    
    parser.add_argument("--rag_sentence_num", type=int, default=16,
                        help="Number of history entries to retrieve when using RAG mode.")
    
    parser.add_argument("--model_name", type=str, default="qwen4B",
                        help="Name of the model to evaluate (e.g., qwen4B, qwen30BA3B, evoembedding EvoEmbedding).")
    
    parser.add_argument("--eval_bench", type=str, default="locomo",
                        choices=["longmemeval_s", "locomo", "personamem32", "PersonaMME32", "PersonaMME128"],
                        help="The benchmark dataset to use.")

    parser.add_argument("--eval_memory", type=str, default="false",
                        choices=["true", "false"],
                        help="Whether to evaluate memory usage.")
    
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="The embedding model used for retrieval.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_method = args.eval_method
    rag_sentence_num = args.rag_sentence_num
    model_name = args.model_name
    eval_bench = args.eval_bench
    embedding_model = args.embedding_model
    eval_memory = args.eval_memory == "true"

    if eval_method == "full":
        save_path = f"./output/eval_results/{eval_bench}/{model_name}_{eval_bench}_{eval_method}_{embedding_model.replace('/', '_')}.json"
    else:
        save_path = f"./output/eval_results/{eval_bench}/{model_name}_{eval_bench}_{eval_method}_{rag_sentence_num}_{embedding_model.replace('/', '_')}.json"
        if "evoembedding" in model_name.lower():
            save_path = f"./output/eval_results/{eval_bench}/0520lora64buffer512_{model_name}_{eval_bench}_{eval_method}_{rag_sentence_num}_{embedding_model.replace('/', '_')}.json"

    print(f"Evaluation Benchmark: {eval_bench}")
    print(f"Evaluation Method: {eval_method}")
    print(f"Model Name: {model_name}")
    print(f"Embedding Model: {embedding_model}")
    print(f"RAG Sentence Num: {rag_sentence_num}")
    print(f"Save Path: {save_path}")
    print(f"Eval Memory: {eval_memory}")
    
    eval(eval_bench, eval_method, rag_sentence_num, save_path, model_name, embedding_model, eval_memory=eval_memory)
