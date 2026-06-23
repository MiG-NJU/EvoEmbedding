#!/bin/bash

DATASETS=("locomo" "longmemeval_s" "personamem32" "PersonaMME32" "PersonaMME128")
EMBEDDING_MODELS=("Qwen/Qwen3-Embedding-0.6B" "Qwen/Qwen3-Embedding-4B" "Qwen/Qwen3-Embedding-8B" "grep" "sentence-transformers/all-MiniLM-L6-v2" "jinaai/jina-embeddings-v5-text-small" "intfloat/multilingual-e5-large-instruct" "BAAI/bge-m3" "tencent/KaLM-Embedding-Gemma3-12B-2511")
RAG_NUMS=(1 2 4 8 16 32)

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
    
    while [ $assigned_gpu -eq -1 ]; do
        for gpu_id in {0..11}; do
            pid=${GPU_PIDS[$gpu_id]}
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

echo "所有任务已派发完毕，等待最后运行的卡收工..."
wait
echo "🎉 全部评估结束！"

REGULAR_BENCHES=("locomo" "longmemeval_s" "personamem32" "PersonaMME32" "PersonaMME128")
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
        TASKS+=("EvoEmbedding30B|$rag|$bench|Qwen/Qwen3-Embedding-0.6B|")
    done
done

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
    
    mem_status="false"
    [ -n "$memory" ] && mem_status="true"
    echo "[进度 $(($task_idx + 1))/$total_tasks] GPU $assigned_gpu -> 模型: $model | RAG: $rag | 评测: $bench | 向量: ${emb##*/} | 开启内存: $mem_status"
    
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

