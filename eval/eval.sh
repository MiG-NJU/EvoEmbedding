#!/bin/bash

# MASTER_PORT=29501 CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server     --model Qwen/Qwen3-30B-A3B-Instruct-2507     --tensor-parallel-size 2     --host 0.0.0.0     --port 8080     --trust-remote-code     --gpu-memory-utilization 0.95

#################### Phase 1

DATASETS=("locomo" "longmemeval_s" "personamem32" "PersonaMME32" "PersonaMME128")
EMBEDDING_MODELS=("Qwen/Qwen3-Embedding-0.6B" "Qwen/Qwen3-Embedding-4B" "Qwen/Qwen3-Embedding-8B" "grep" "sentence-transformers/all-MiniLM-L6-v2" "jinaai/jina-embeddings-v5-text-small" "intfloat/multilingual-e5-large-instruct" "BAAI/bge-m3" "tencent/KaLM-Embedding-Gemma3-12B-2511")
RAG_NUMS=(1 2 4 8 16 32)
####################
TASKS=()
for emb in "${EMBEDDING_MODELS[@]}"; do
    for rag in "${RAG_NUMS[@]}"; do
        for bench in "${DATASETS[@]}"; do
            TASKS+=("$bench|$rag|$emb")
        done
    done
done

total_tasks=${#TASKS[@]}
echo "总计生成任务数：$total_tasks"
GPU_PIDS=(0 0 0 0 0 0 0 0 0 0 0 0) 

task_idx=0
while [ $task_idx -lt $total_tasks ]; do
    assigned_gpu=-1
    
    # 找一张空闲的卡
    while [ $assigned_gpu -eq -1 ]; do
        for gpu_id in {0..11}; do
            pid=${GPU_PIDS[$gpu_id]}
            # 如果 pid 是 0（还没分配过），或者用 kill -0 检测发现该进程已经运行结束
            if [ "$pid" == "0" ] || ! kill -0 "$pid" 2>/dev/null; then
                assigned_gpu=$gpu_id
                break
            fi
        done

        if [ $assigned_gpu -eq -1 ]; then
            sleep 3 
        fi
    done

    task_info="${TASKS[$task_idx]}"
    IFS='|' read -r bench rag emb <<< "$task_info"
    
    echo "[进度 $(($task_idx + 1))/$total_tasks] 分配给 GPU $assigned_gpu -> 评测: $bench | RAG: $rag | 模型: $emb"
    CUDA_VISIBLE_DEVICES=$(($assigned_gpu % 6 + 2)) python eval/eval.py --eval_method rag --model_name qwen30ba3b --rag_sentence_num "$rag" --eval_bench "$bench" --embedding_model "$emb" &
    GPU_PIDS[$assigned_gpu]=$!
    ((task_idx++))
done

# ================= 4. 等待最后几个任务收尾 =================
echo "所有任务已派发完毕，等待最后运行的卡收工..."
wait
echo "🎉 全部评估结束！"

#################### Phase 2


REGULAR_BENCHES=("locomo" "longmemeval_s" "personamem32" "personamem128" "PersonaMME32" "PersonaMME128")
QWEN_EMBEDDINGS=(
    "Qwen/Qwen3-Embedding-0.6B" 
    "Qwen/Qwen3-Embedding-4B" 
    "Qwen/Qwen3-Embedding-8B" 
    "grep" 
    "sentence-transformers/all-MiniLM-L6-v2" 
    "jinaai/jina-embeddings-v5-text-small" 
    "intfloat/multilingual-e5-large-instruct" 
    "BAAI/bge-m3" 
    "tencent/KaLM-Embedding-Gemma3-12B-2511"
)
RAG_NUMS=(1 2 4 8 16 32)

TASKS=()
for rag in "${RAG_NUMS[@]}"; do
    for bench in "${REGULAR_BENCHES[@]}"; do
        # 格式: model_name | rag_num | bench_name | emb_model | eval_memory(空代表false)
        TASKS+=("EvoEmbedding|$rag|$bench|Qwen/Qwen3-Embedding-0.6B|")
    done
done

# CUDA_VISIBLE_DEVICES=5 python eval/eval.py --eval_method rag --model_name qwen30ba3b --rag_sentence_num 32 --eval_bench longmemeval_s --embedding_model Qwen/Qwen3-Embedding-8B --eval_memory false &

# for rag in "${RAG_NUMS[@]}"; do
#     for mem_flag in "" "--eval_memory true"; do
#         for emb in "${QWEN_EMBEDDINGS[@]}"; do
#             TASKS+=("qwen30ba3b|$rag|PersonaRAGBench|$emb|$mem_flag")
#         done
#         TASKS+=("EvoEmbedding|$rag|PersonaRAGBench|Qwen/Qwen3-Embedding-0.6B|$mem_flag")
#     done
# done

total_tasks=${#TASKS[@]}
echo "总计生成任务数：$total_tasks"
GPU_PIDS=(0 0 0 0 0 0 0 0 0 0 0 0) 
NUM_GPUS=${#GPU_PIDS[@]}

task_idx=0
while [ $task_idx -lt $total_tasks ]; do
    assigned_gpu=-1
    
    while [ $assigned_gpu -eq -1 ]; do
        for gpu_id in {0..11}; do
            pid=${GPU_PIDS[$gpu_id]}
            if [ "$pid" == "0" ] || ! kill -0 "$pid" 2>/dev/null; then
                assigned_gpu=$gpu_id
                break
            fi
        done
        if [ $assigned_gpu -eq -1 ]; then
            sleep 5 
        fi
    done

    task_info="${TASKS[$task_idx]}"
    IFS='|' read -r model rag bench emb memory <<< "$task_info"
    
    # 打印日志
    mem_status="false"
    [ -n "$memory" ] && mem_status="true"
    echo "[进度 $(($task_idx + 1))/$total_tasks] GPU $assigned_gpu -> 模型: $model | RAG: $rag | 评测: $bench | 向量: ${emb##*/} | 开启内存: $mem_status"
    
    # 启动任务，区分是否带有 --eval_memory true
    if [ -n "$memory" ]; then
        CUDA_VISIBLE_DEVICES=$(($assigned_gpu % 6 + 2)) python eval/eval.py --eval_method rag \
            --model_name "$model" --rag_sentence_num "$rag" \
            --eval_bench "$bench" --embedding_model "$emb" --eval_memory true &
    else
        CUDA_VISIBLE_DEVICES=$(($assigned_gpu % 6 + 2)) python eval/eval.py --eval_method rag \
            --model_name "$model" --rag_sentence_num "$rag" \
            --eval_bench "$bench" --embedding_model "$emb" &
    fi
    
    GPU_PIDS[$assigned_gpu]=$!
    ((task_idx++))
    sleep 1 
done

echo "所有任务 ($total_tasks 个) 已派发完毕，等待最后运行的卡收工..."
wait
echo "🎉 全部评估结束！"

# 最后测试full情况
# CUDA_VISIBLE_DEVICES=2 python eval/eval.py --eval_method full --model_name qwen30ba3b --rag_sentence_num 2 --eval_bench locomo --embedding_model Qwen/Qwen3-Embedding-0.6B --eval_memory false &
# CUDA_VISIBLE_DEVICES=3 python eval/eval.py --eval_method full --model_name qwen30ba3b --rag_sentence_num 2 --eval_bench longmemeval_s --embedding_model Qwen/Qwen3-Embedding-0.6B --eval_memory false &
# CUDA_VISIBLE_DEVICES=4 python eval/eval.py --eval_method full --model_name qwen30ba3b --rag_sentence_num 2 --eval_bench membencheval --embedding_model Qwen/Qwen3-Embedding-0.6B --eval_memory false &
# CUDA_VISIBLE_DEVICES=5 python eval/eval.py --eval_method full --model_name qwen30ba3b --rag_sentence_num 2 --eval_bench personamem32 --embedding_model Qwen/Qwen3-Embedding-0.6B --eval_memory false &
# CUDA_VISIBLE_DEVICES=6 python eval/eval.py --eval_method full --model_name qwen30ba3b --rag_sentence_num 2 --eval_bench personamem128 --embedding_model Qwen/Qwen3-Embedding-0.6B --eval_memory false &
# CUDA_VISIBLE_DEVICES=7 python eval/eval.py --eval_method full --model_name qwen30ba3b --rag_sentence_num 2 --eval_bench PersonaMME32 --embedding_model Qwen/Qwen3-Embedding-0.6B --eval_memory false &
# CUDA_VISIBLE_DEVICES=5 python eval/eval.py --eval_method full --model_name qwen30ba3b --rag_sentence_num 2 --eval_bench PersonaMME128 --embedding_model Qwen/Qwen3-Embedding-0.6B --eval_memory false &
# CUDA_VISIBLE_DEVICES=6 python eval/eval.py --eval_method full --model_name qwen30ba3b --rag_sentence_num 2 --eval_bench PersonaRAGBench --embedding_model Qwen/Qwen3-Embedding-0.6B --eval_memory false &
# CUDA_VISIBLE_DEVICES=7 python eval/eval.py --eval_method full --model_name qwen30ba3b --rag_sentence_num 2 --eval_bench PersonaRAGBench --embedding_model Qwen/Qwen3-Embedding-0.6B --eval_memory true &




