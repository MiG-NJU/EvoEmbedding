from transformers import PretrainedConfig, PreTrainedModel, AutoModelForCausalLM, AutoModel
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.checkpoint import checkpoint 
import torch.distributed as dist

import random, copy
import os
import json
import hashlib
import numpy as np


def print_rank0(*args, **kwargs):
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


class EvoRAGConfig(PretrainedConfig):
    model_type = "EvoRAG"

    def __init__(
        self, 
        base_model_name_or_path="Qwen/Qwen3-4B-Instruct-2507", 
        num_latents=16, 
        buffer_capacity=512,
        lora_r=64,
        lora_alpha=128,
        local_files_only=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self._name_or_path = base_model_name_or_path
        self.num_latents = num_latents
        self.buffer_capacity = buffer_capacity
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.local_files_only = local_files_only


class EvoRAGModel(PreTrainedModel):
    supports_gradient_checkpointing = True
    def __init__(self, config: EvoRAGConfig, tokenizer=None):
        super().__init__(config)
        self.config = config
        print(f"Loading base model: {config.base_model_name_or_path}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            local_files_only=config.local_files_only,
        )
        self.mem_tokenizer = tokenizer

        emb_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            local_files_only=config.local_files_only,
        )

        for param in base_model.parameters():
            param.requires_grad = False
        for param in emb_model.parameters():
            param.requires_grad = False

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r, 
            lora_alpha=config.lora_alpha, 
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        self.model = get_peft_model(base_model, peft_config, adapter_name="memory_generator")
        self.model.add_adapter("retriever", peft_config)

        self.emb_model = emb_model
        self.hidden_size = base_model.config.hidden_size

        # memory generation
        self.memory_queries = nn.Parameter(
            torch.randn(1, config.num_latents, self.hidden_size, dtype=torch.bfloat16)
        )

        self.latent_to_mem = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU())
        self.mem_to_latent = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size))
        
        self.emb_hidden_size = emb_model.config.hidden_size      
        self.mem_to_emb = nn.Sequential(
            nn.Linear(self.hidden_size, self.emb_hidden_size), 
            nn.ReLU(),
            nn.Linear(self.emb_hidden_size, self.emb_hidden_size)
        )

        for module in [self.latent_to_mem, self.mem_to_latent, self.mem_to_emb]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.eye_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.normal_(self.memory_queries, std=0.02)

        for name, param in self.model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        self.memory_queries.requires_grad = True
        for param in self.latent_to_mem.parameters():
            param.requires_grad = True
        for param in self.mem_to_latent.parameters():
            param.requires_grad = True

        self.retrieval_loss_weight = 1.0 

    def _load_emb_model(self):
        emb_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )        
        for param in emb_model.parameters():
            param.requires_grad = False
        self.emb_model = emb_model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_emb_input_embeddings(self):
        return self.emb_model.get_input_embeddings()

    def _run_with_positions(self, model_obj, embeds_, mask_, adapter=None):
        pos_ids = (torch.cumsum(mask_, dim=1).long() - 1).masked_fill_(mask_ == 0, 1)
        if adapter:
            model_obj.set_adapter(adapter)
        out = model_obj(
            inputs_embeds=embeds_,
            attention_mask=mask_,
            position_ids=pos_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        return out.hidden_states[-1]

    @torch.no_grad()
    def build_latent_buffer(self, history_turns=None, tokenizer=None):
        self.eval()
        if tokenizer is None:
            tokenizer = self.mem_tokenizer
        if tokenizer is None:
            raise ValueError("tokenizer is required for build_latent_buffer")

        history_turns = history_turns or []
        device = self.model.device
        IM_END_ID = 151643

        history_token_ids = []
        for turn in history_turns:
            ids = tokenizer(str(turn), add_special_tokens=False)["input_ids"]
            if not ids or ids[-1] != IM_END_ID:
                ids = ids + [IM_END_ID]
            history_token_ids.append(ids)

        current_buffer = None
        global_raw_buffer = None
        if history_token_ids:
            t = 0
            target_len = 2048
            chunk_size_target = 16
            while t < len(history_token_ids):
                chunk_size = 0
                accumulated_len = 0
                for i in range(t, len(history_token_ids)):
                    turn_len = len(history_token_ids[i])
                    if chunk_size > 0 and (accumulated_len + turn_len) > target_len:
                        break
                    accumulated_len += turn_len
                    chunk_size += 1
                    if chunk_size >= chunk_size_target:
                        break

                chunk_embeds, chunk_masks, query_positions = [], [], []
                offset = 0
                if current_buffer is not None:
                    buf_latent = self.mem_to_latent(current_buffer)
                    chunk_embeds.append(buf_latent)
                    chunk_masks.append(torch.ones((1, buf_latent.shape[1]), device=device, dtype=torch.long))
                    offset += buf_latent.shape[1]

                for ids in history_token_ids[t:t + chunk_size]:
                    turn_tensor = torch.tensor([ids], device=device)
                    turn_emb = self.get_input_embeddings()(turn_tensor)
                    chunk_embeds.append(turn_emb)
                    chunk_masks.append(torch.ones((1, turn_emb.shape[1]), device=device, dtype=torch.long))
                    offset += turn_emb.shape[1]
                    chunk_embeds.append(self.memory_queries)
                    chunk_masks.append(torch.ones((1, self.config.num_latents), device=device, dtype=torch.long))
                    query_positions.append((offset, offset + self.config.num_latents))
                    offset += self.config.num_latents

                gen_out = self._run_with_positions(
                    self.model,
                    torch.cat(chunk_embeds, dim=1),
                    torch.cat(chunk_masks, dim=1),
                    adapter="memory_generator",
                )
                for q_start, q_end in query_positions:
                    new_latent = self.latent_to_mem(gen_out[:, q_start:q_end, :])
                    global_raw_buffer = new_latent if global_raw_buffer is None else torch.cat([global_raw_buffer, new_latent], dim=1)
                    current_buffer = global_raw_buffer[:, -self.config.buffer_capacity:, :]
                del gen_out
                t += chunk_size

        return None if current_buffer is None else current_buffer.detach().cpu()

    @torch.no_grad()
    def encode_texts_with_latent_buffer(self, texts, latent_buffer=None, tokenizer=None, native=False, max_batch_tokens=2048):
        self.eval()
        if tokenizer is None:
            tokenizer = self.mem_tokenizer
        if tokenizer is None:
            raise ValueError("tokenizer is required for encode_texts_with_latent_buffer")

        text_list = [texts] if isinstance(texts, str) else list(texts)
        device = self.model.device
        IM_END_ID = 151643

        current_buffer = None
        if latent_buffer is not None and not native:
            current_buffer = latent_buffer.to(device=device, dtype=self.memory_queries.dtype)

        encoded = []
        tokenized_texts = []
        for text in text_list:
            ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
            if not ids or ids[-1] != IM_END_ID:
                ids = ids + [IM_END_ID]
            tokenized_texts.append(ids)

        curr_idx = 0
        while curr_idx < len(tokenized_texts):
            batch_seqs = []
            batch_max_len = 0
            while curr_idx < len(tokenized_texts):
                ids = tokenized_texts[curr_idx]
                if native:
                    seq = self.get_emb_input_embeddings()(torch.tensor([ids], device=device))[0]
                else:
                    text_emb = self.get_input_embeddings()(torch.tensor([ids], device=device))[0]
                    if current_buffer is not None:
                        buf_latent = self.mem_to_latent(current_buffer)[0]
                        seq = torch.cat([buf_latent, text_emb], dim=0)
                    else:
                        seq = text_emb
                new_max_len = max(batch_max_len, seq.size(0))
                if batch_seqs and (len(batch_seqs) + 1) * new_max_len > max_batch_tokens:
                    break
                batch_seqs.append(seq)
                batch_max_len = new_max_len
                curr_idx += 1

            batched_embeds = torch.stack([
                torch.cat([torch.zeros(batch_max_len - s.size(0), s.shape[-1], device=device, dtype=s.dtype), s])
                for s in batch_seqs
            ])
            batched_masks = torch.stack([
                torch.cat([torch.zeros(batch_max_len - s.size(0), device=device), torch.ones(s.size(0), device=device)])
                for s in batch_seqs
            ]).long()

            if native:
                out = self._run_with_positions(self.emb_model, batched_embeds, batched_masks, adapter=None)
                vec = out[:, -1, :]
            else:
                out = self._run_with_positions(self.model, batched_embeds, batched_masks, adapter="retriever")
                vec = self.mem_to_emb(self.latent_to_mem(out[:, -1:, :])).mean(dim=1)
            encoded.append(vec.float().cpu())

        embeddings = torch.cat(encoded, dim=0).numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)
        return embeddings

    @torch.no_grad()
    def encode_texts_with_memory(self, texts, history_turns=None, tokenizer=None, native=False, max_batch_tokens=2048):
        latent_buffer = None if native else self.build_latent_buffer(history_turns=history_turns, tokenizer=tokenizer)
        return self.encode_texts_with_latent_buffer(
            texts,
            latent_buffer=latent_buffer,
            tokenizer=tokenizer,
            native=native,
            max_batch_tokens=max_batch_tokens,
        )


    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        if batch_size != 1:
            raise ValueError("Recurrent training currently only supports batch_size=1 due to dynamic slicing complexity.")
            
        IM_END_ID = 151645
        USER_ID = 872
        INSERT_IDX = 3

        meta = kwargs.get("meta", [{}])[0]
        evidence_indices = meta.get("evidence_turns", [])
        turns = meta.get("turns", [])

        # 1. 格式化历史 turns，补充 ChatML 的 role tokens
        for idx, turn in enumerate(turns):
            if idx == len(turns)-1:
                turns[idx] = [641, 1235, 25, 16246, 264, 3482, 2711, 3239, 11, 17179,
                                9760, 46769, 429, 4226, 279, 3239, 198, 2859, 25] + turn + [151643]
            else:
                turns[idx] = turn + [151643]

        end_indices = (input_ids[0] == IM_END_ID).nonzero(as_tuple=True)[0]
        split_points = [0]
        for idx in end_indices:
            if idx == seq_len-2 or input_ids[0][idx+3]==USER_ID:
                split_points.append(idx.item() + 2)
        
        if split_points[-1] != seq_len:
            raise ValueError("Input sequence wrongly formatted.")
        num_turns = len(split_points) - 1

        # ====== 核心包装器：处理并行拼装和动态位置编码 ======
        def forward_with_cp(model_obj, embeds, mask, adapter=None):
            pos_ids = (torch.cumsum(mask, dim=1).long() - 1).masked_fill_(mask == 0, 1)
            def run_transformer(embeds_, mask_, pos_ids_):
                if adapter: 
                    model_obj.set_adapter(adapter)
                out = model_obj(
                    inputs_embeds=embeds_, 
                    attention_mask=mask_, 
                    position_ids=pos_ids_, 
                    output_hidden_states=True, 
                    use_cache=False
                )
                return out.hidden_states[-1]
                
            if self.training and embeds.requires_grad:
                return checkpoint(run_transformer, embeds, mask, pos_ids, use_reentrant=False)
            return run_transformer(embeds, mask, pos_ids)

        # ================== 第一阶段：自适应 Chunk 构建与历史记忆生成 ==================
        current_buffer = None
        global_raw_buffer = None   
        buffer_end_indices = []    # 用于后续检索记录对应的记忆游标
        total_history = num_turns - 1
        t = 0

        target_len = 2048
        chunk_size_target = 16

        while t < total_history:
            chunk_size = 0
            accumulated_len = 0
            
            for i in range(t, total_history):
                turn_len = split_points[i+1] - split_points[i]
                if chunk_size > 0 and (accumulated_len + turn_len) > target_len:
                    break
                accumulated_len += turn_len
                chunk_size += 1
                if chunk_size >= chunk_size_target:
                    break

            chunk_embeds, chunk_masks, query_positions = [], [], []
            offset = 0
            
            # 注入之前的 Buffer
            if current_buffer is not None:
                buf_latent = self.mem_to_latent(current_buffer)
                chunk_embeds.append(buf_latent)
                chunk_masks.append(torch.ones((1, buf_latent.shape[1]), device=device, dtype=attention_mask.dtype))
                offset += buf_latent.shape[1]
                
            # 拼装当前 Chunk 内部的文本与 Queries
            for i in range(t, t + chunk_size):
                s_idx, e_idx = split_points[i], split_points[i+1]
                turn_emb = self.get_input_embeddings()(input_ids[:, s_idx:e_idx])
                
                chunk_embeds.append(turn_emb)
                chunk_masks.append(attention_mask[:, s_idx:e_idx])
                offset += turn_emb.shape[1]
                
                chunk_embeds.append(self.memory_queries)
                chunk_masks.append(torch.ones((1, self.config.num_latents), device=device, dtype=attention_mask.dtype))
                query_positions.append((offset, offset + self.config.num_latents))
                offset += self.config.num_latents
                
            gen_out = forward_with_cp(
                self.model, 
                torch.cat(chunk_embeds, dim=1), 
                torch.cat(chunk_masks, dim=1), 
                adapter="memory_generator"
            )
            
            for q_start, q_end in query_positions:
                new_latent = self.latent_to_mem(gen_out[:, q_start:q_end, :])
                buffer_end_indices.append(0 if global_raw_buffer is None else global_raw_buffer.shape[1])
                
                if global_raw_buffer is None:
                    global_raw_buffer = new_latent
                else:
                    global_raw_buffer = torch.cat([global_raw_buffer, new_latent], dim=1)
                    
                if global_raw_buffer.shape[1] > self.config.buffer_capacity:
                    current_buffer = global_raw_buffer[:, -self.config.buffer_capacity:, :]
                else:
                    current_buffer = global_raw_buffer  
            del gen_out
            t += chunk_size
        
        buffer_end_indices.append(global_raw_buffer.shape[1] if global_raw_buffer is not None else 0)

        # ================== 第二阶段：最后一轮的预测与 Loss 计算 (LM Task) ==================
        start_idx, end_idx = split_points[-2], split_points[-1]
        text_embeds = self.get_input_embeddings()(input_ids[:, start_idx:end_idx])
        cur_attention_mask = attention_mask[:, start_idx:end_idx]
        cur_labels = labels[:, start_idx:end_idx] if labels is not None else None

        if current_buffer is not None:
            actual_idx = min(INSERT_IDX, text_embeds.shape[1])
            buf_latent = self.mem_to_latent(current_buffer)
            buf_len = buf_latent.shape[1]
            
            combined_embeds = torch.cat([text_embeds[:, :actual_idx], buf_latent, text_embeds[:, actual_idx:]], dim=1)
            combined_mask = torch.cat([cur_attention_mask[:, :actual_idx], torch.ones((1, buf_len), device=device, dtype=cur_attention_mask.dtype), cur_attention_mask[:, actual_idx:]], dim=1)
            
            if cur_labels is not None:
                combined_labels = torch.cat([cur_labels[:, :actual_idx], torch.full((1, buf_len), -100, device=device, dtype=cur_labels.dtype), cur_labels[:, actual_idx:]], dim=1)
            else:
                combined_labels = None
        else:
            combined_embeds, combined_mask, combined_labels = text_embeds, cur_attention_mask, cur_labels

        with self.model.disable_adapter():
            outputs = self.model(
                inputs_embeds=combined_embeds, 
                attention_mask=combined_mask, 
                labels=combined_labels, 
                use_cache=False
            )

        # ================== 第三阶段：Batched Retriever 特征生成 ==================
        turn_embeddings = []
        if evidence_indices and turns:
            N_turns = len(turns)
            retrieval_indices = list(buffer_end_indices) if len(turns) == num_turns else [global_raw_buffer.shape[1] if global_raw_buffer is not None else 0] * N_turns

            prompt_vec_list = []
            max_batch_tokens = 2048 
            
            curr_idx = 0
            while curr_idx < N_turns:
                batch_seqs = []
                batch_max_len = 0
                
                while curr_idx < N_turns:
                    turn_tokens = turns[curr_idx]
                    end_ptr = retrieval_indices[curr_idx]
                    
                    t_emb = self.get_input_embeddings()(torch.tensor([turn_tokens], device=device))[0]
                    b_emb = self.mem_to_latent(global_raw_buffer[:, max(0, end_ptr-self.config.buffer_capacity):end_ptr, :])[0] if end_ptr > 0 else torch.empty(0, self.hidden_size, device=device)
                    seq = torch.cat([b_emb, t_emb], dim=0)
                    
                    new_max_len = max(batch_max_len, seq.size(0))
                    potential_tokens = (len(batch_seqs) + 1) * new_max_len
                    
                    if len(batch_seqs) > 0 and potential_tokens > max_batch_tokens:
                        break
                    
                    batch_seqs.append(seq)
                    batch_max_len = new_max_len
                    curr_idx += 1

                batched_ret_embeds = torch.stack([
                    torch.cat([torch.zeros(batch_max_len - s.size(0), self.hidden_size, device=device, dtype=s.dtype), s]) 
                    for s in batch_seqs
                ])
                batched_ret_masks = torch.stack([
                    torch.cat([torch.zeros(batch_max_len - s.size(0), device=device), torch.ones(s.size(0), device=device)]) 
                    for s in batch_seqs
                ]).to(attention_mask.dtype)

                ret_out = forward_with_cp(self.model, batched_ret_embeds, batched_ret_masks, adapter="retriever")
                p_vec = self.mem_to_emb(self.latent_to_mem(ret_out[:, -1:, :])).mean(dim=1)
                prompt_vec_list.append(p_vec)
            turn_embeddings = list(torch.cat(prompt_vec_list, dim=0))

        # ================== 第四阶段：对比学习 Loss 计算 ==================
        if evidence_indices and len(turn_embeddings) >= 2:
            query_emb = turn_embeddings[-1]
            history_keys = torch.stack(turn_embeddings[:-1], dim=0)

            query_emb_norm = torch.nn.functional.normalize(query_emb.unsqueeze(0).float(), p=2, dim=-1)
            history_keys_norm = torch.nn.functional.normalize(history_keys.unsqueeze(0).float(), p=2, dim=-1)

            valid_evidence_indices = [idx for idx in evidence_indices if idx < history_keys_norm.shape[1]]
            all_idx = list(range(history_keys_norm.shape[1]))
            neg_idx = [i for i in all_idx if i not in valid_evidence_indices]

            if valid_evidence_indices:
                pos_embs = history_keys_norm[:, valid_evidence_indices]
                neg_embs = history_keys_norm[:, neg_idx]

                pos_scores = torch.bmm(query_emb_norm.unsqueeze(1), pos_embs.transpose(1, 2)).squeeze(1) / 0.1
                neg_scores = torch.bmm(query_emb_norm.unsqueeze(1), neg_embs.transpose(1, 2)).squeeze(1) / 0.1
                print(1111, evidence_indices, len(turns), pos_scores, neg_scores)
                loss_fct = torch.nn.CrossEntropyLoss()
                total_loss = 0.0
                target = torch.zeros(1, dtype=torch.long, device=device)

                for i in range(pos_scores.shape[1]):
                    logits = torch.cat([pos_scores[:, i:i+1], neg_scores], dim=1)
                    total_loss += loss_fct(logits, target)
                contrastive_loss = total_loss / pos_scores.shape[1] * torch.log(torch.tensor(neg_scores.shape[1] + 1.0, device=device)) 
                print('debug111', outputs["logits"].shape, num_turns, len(turn_embeddings), input_ids.shape, outputs.loss, contrastive_loss)
                
                outputs.loss = outputs.loss + self.retrieval_loss_weight * contrastive_loss
        else:
            print('debug2', outputs["logits"].shape, num_turns, len(turn_embeddings), input_ids.shape, outputs.loss)

        # Trick: 保持 Dummy Loss 防止 DDP 报错
        if self.training:
            dummy_loss = 0.0 * sum(p.sum() for p in self.parameters() if p.requires_grad)
            outputs.loss = outputs.loss + dummy_loss
            
        return outputs


    @torch.no_grad()
    def generate_raw(self, input_ids, attention_mask=None, **kwargs):
        self.eval()
        device = self.model.device
        input_ids = input_ids.to(device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            attention_mask = attention_mask.to(device)

        with self.model.disable_adapter():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                **kwargs
            )
        return output_ids
    

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, **kwargs):
        self.eval() 
        batch_size, seq_len = input_ids.shape
        if batch_size != 1: 
            raise ValueError("Batch size 1 only")
        
        device = self.model.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        IM_END_ID = 151645
        USER_ID = 872
        INSERT_IDX = 3

        # ================== 1. 拆分对话轮次 ==================
        end_indices = (input_ids[0] == IM_END_ID).nonzero(as_tuple=True)[0]
        split_points = [0]
        for idx in end_indices:
            if idx == seq_len-2 or (idx+3 < seq_len and input_ids[0][idx+3]==USER_ID):
                split_points.append(idx.item() + 2)
        
        if split_points[-1] != seq_len:
            split_points.append(seq_len)
            
        num_turns = len(split_points) - 1
        total_history = num_turns - 1

        # ================== 2. 并行化历史记忆生成 (Chunked) ==================
        current_buffer = None
        global_raw_buffer = None   
        t = 0

        target_len = 2048
        chunk_size_target = 16

        # 启用 memory_generator Adapter
        self.model.set_adapter("memory_generator")

        while t < total_history:
            chunk_size = 0
            accumulated_len = 0
            
            # 动态计算当前 Chunk 的大小
            for i in range(t, total_history):
                turn_len = split_points[i+1] - split_points[i]
                if chunk_size > 0 and (accumulated_len + turn_len) > target_len:
                    break
                accumulated_len += turn_len
                chunk_size += 1
                if chunk_size >= chunk_size_target:
                    break

            chunk_embeds, chunk_masks, query_positions = [], [], []
            offset = 0
            
            # 注入之前的 Buffer
            if current_buffer is not None:
                buf_latent = self.mem_to_latent(current_buffer)
                chunk_embeds.append(buf_latent)
                chunk_masks.append(torch.ones((1, buf_latent.shape[1]), device=device, dtype=attention_mask.dtype))
                offset += buf_latent.shape[1]
                
            # 拼装当前 Chunk 内部的文本与 Queries
            for i in range(t, t + chunk_size):
                s_idx, e_idx = split_points[i], split_points[i+1]
                turn_emb = self.get_input_embeddings()(input_ids[:, s_idx:e_idx])
                
                chunk_embeds.append(turn_emb)
                chunk_masks.append(attention_mask[:, s_idx:e_idx])
                offset += turn_emb.shape[1]
                
                chunk_embeds.append(self.memory_queries)
                chunk_masks.append(torch.ones((1, self.config.num_latents), device=device, dtype=attention_mask.dtype))
                query_positions.append((offset, offset + self.config.num_latents))
                offset += self.config.num_latents
                
            combined_embeds = torch.cat(chunk_embeds, dim=1)
            combined_masks = torch.cat(chunk_masks, dim=1)
            
            # 动态计算 position_ids (与 forward 保持一致)
            pos_ids = (torch.cumsum(combined_masks, dim=1).long() - 1).masked_fill_(combined_masks == 0, 1)

            # 单次前向传播处理整个 Chunk
            gen_out = self.model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_masks,
                position_ids=pos_ids,
                output_hidden_states=True,
                use_cache=False
            ).hidden_states[-1]
            
            # 提取 Queries 位置的隐藏状态并更新 Buffer
            for q_start, q_end in query_positions:
                new_latent = self.latent_to_mem(gen_out[:, q_start:q_end, :])
                
                if global_raw_buffer is None:
                    global_raw_buffer = new_latent
                else:
                    global_raw_buffer = torch.cat([global_raw_buffer, new_latent], dim=1)
                    
                if global_raw_buffer.shape[1] > self.config.buffer_capacity:
                    current_buffer = global_raw_buffer[:, -self.config.buffer_capacity:, :]
                else:
                    current_buffer = global_raw_buffer  
            
            t += chunk_size

        # ================== 3. 最后一轮预测与文本生成 ==================
        last_start, last_end = split_points[-2], split_points[-1]
        last_ids = input_ids[:, last_start:last_end]
        cur_attention_mask = attention_mask[:, last_start:last_end]
        text_embeds = self.get_input_embeddings()(last_ids)

        if current_buffer is not None:
            actual_insert_idx = min(INSERT_IDX, text_embeds.shape[1])
            buffer_latent = self.mem_to_latent(current_buffer)
            buf_len = buffer_latent.shape[1]
            
            final_embeds = torch.cat([
                text_embeds[:, :actual_insert_idx, :],
                buffer_latent, 
                text_embeds[:, actual_insert_idx:, :]
            ], dim=1)
            
            final_mask = torch.cat([
                cur_attention_mask[:, :actual_insert_idx],
                torch.ones((1, buf_len), device=device, dtype=cur_attention_mask.dtype),
                cur_attention_mask[:, actual_insert_idx:]
            ], dim=1)
        else:
            final_embeds = text_embeds
            final_mask = cur_attention_mask

        with self.model.disable_adapter():
            output_ids = self.model.generate(
                inputs_embeds=final_embeds,
                attention_mask=final_mask,
                use_cache=True,
                **kwargs 
            )
            
        return output_ids

    
    @torch.no_grad()
    def generate_retrieve_idx(self, input_ids, attention_mask=None, meta=None, native=False, save_dir='./output/model_cache', **kwargs):
        cache_path = None
        if save_dir is not None:
            model_hash = hashlib.md5(str(self.config.base_model_name_or_path).encode('utf-8')).hexdigest()
            save_dir = os.path.join(save_dir, model_hash)
            os.makedirs(save_dir, exist_ok=True)
            input_bytes = input_ids.cpu().numpy().tobytes()
            meta_str = json.dumps(meta, sort_keys=True) if meta else "{}"
            hash_obj = hashlib.md5(input_bytes)
            hash_obj.update(meta_str.encode('utf-8'))
            hash_obj.update(str(native).encode('utf-8'))
            cache_key = hash_obj.hexdigest()
            cache_path = os.path.join(save_dir, f"cache_{cache_key}.pt")
            
            if os.path.exists(cache_path):
                print("Cache hit! Loading results from cache.")
                device = self.model.device
                try:
                    cached_result = torch.load(cache_path, map_location=device)
                    return cached_result
                except Exception as e:
                    print(f"Failed to load cache due to: {e}. Recomputing.")
                cache_path = None
        # =======================================================
        
        self.eval()
        batch_size, seq_len = input_ids.shape
        device = self.model.device
        input_ids = input_ids.to(device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)
        
        if batch_size != 1:
            raise ValueError("Inference only supports batch_size=1")

        meta = meta if meta else {}
        turns = meta.get("turns", [])
        
        IM_END_ID = 151645
        USER_ID = 872
        
        # 1. 格式化历史 turns，补充 ChatML 的 role tokens
        processed_turns = []
        for idx, turn in enumerate(turns):
            if idx == len(turns) - 1:
                processed_turn = [641, 1235, 25, 16246, 264, 3482, 2711, 3239, 11, 17179,
                                9760, 46769, 429, 4226, 279, 3239, 198, 2859, 25] + turn + [151643]
            else:
                processed_turn = turn + [151643]
            processed_turns.append(processed_turn)

        end_indices = (input_ids[0] == IM_END_ID).nonzero(as_tuple=True)[0]
        split_points = [0]
        for idx in end_indices:
            if idx == seq_len - 2 or (idx + 3 < seq_len and input_ids[0][idx+3] == USER_ID):
                split_points.append(idx.item() + 2)
        
        if split_points[-1] != seq_len:
            split_points.append(seq_len)
        
        num_turns = len(split_points) - 1
        N_turns = len(processed_turns)
        
        if N_turns != num_turns:
            print(f"Warning: processed_turns({N_turns}) != num_turns({num_turns})")

        # ====== 核心包装器：处理并行拼装和动态位置编码 (去除了 checkpoint) ======
        def run_transformer(model_obj, embeds_, mask_, adapter=None):
            pos_ids = (torch.cumsum(mask_, dim=1).long() - 1).masked_fill_(mask_ == 0, 1)
            if adapter: 
                model_obj.set_adapter(adapter)
            out = model_obj(
                inputs_embeds=embeds_, 
                attention_mask=mask_, 
                position_ids=pos_ids, 
                output_hidden_states=True, 
                use_cache=False
            )
            return out.hidden_states[-1]

        # ================== 第一阶段：自适应 Chunk 构建与历史记忆生成 ==================
        current_buffer = None
        global_raw_buffer = None   
        buffer_end_indices = []
        total_history = num_turns - 1 # 最后一次 turn 是 query，无需生成下一轮 memory
        t = 0

        target_len = 2048
        chunk_size_target = 16

        if not native: # Native 模式无需生成历史 Memory
            while t < total_history:
                chunk_size = 0
                accumulated_len = 0
                
                for i in range(t, total_history):
                    turn_len = split_points[i+1] - split_points[i]
                    if chunk_size > 0 and (accumulated_len + turn_len) > target_len:
                        break
                    accumulated_len += turn_len
                    chunk_size += 1
                    if chunk_size >= chunk_size_target:
                        break

                chunk_embeds, chunk_masks, query_positions = [], [], []
                offset = 0
                
                if current_buffer is not None:
                    buf_latent = self.mem_to_latent(current_buffer)
                    chunk_embeds.append(buf_latent)
                    chunk_masks.append(torch.ones((1, buf_latent.shape[1]), device=device, dtype=attention_mask.dtype))
                    offset += buf_latent.shape[1]
                    
                for i in range(t, t + chunk_size):
                    s_idx, e_idx = split_points[i], split_points[i+1]
                    turn_emb = self.get_input_embeddings()(input_ids[:, s_idx:e_idx])
                    
                    chunk_embeds.append(turn_emb)
                    chunk_masks.append(attention_mask[:, s_idx:e_idx])
                    offset += turn_emb.shape[1]
                    
                    chunk_embeds.append(self.memory_queries)
                    chunk_masks.append(torch.ones((1, self.config.num_latents), device=device, dtype=attention_mask.dtype))
                    query_positions.append((offset, offset + self.config.num_latents))
                    offset += self.config.num_latents
                    
                gen_out = run_transformer(
                    self.model, 
                    torch.cat(chunk_embeds, dim=1), 
                    torch.cat(chunk_masks, dim=1), 
                    adapter="memory_generator"
                )
                
                for q_start, q_end in query_positions:
                    new_latent = self.latent_to_mem(gen_out[:, q_start:q_end, :])
                    buffer_end_indices.append(0 if global_raw_buffer is None else global_raw_buffer.shape[1])
                    
                    if global_raw_buffer is None:
                        global_raw_buffer = new_latent
                    else:
                        global_raw_buffer = torch.cat([global_raw_buffer, new_latent], dim=1)
                        
                    if global_raw_buffer.shape[1] > self.config.buffer_capacity:
                        current_buffer = global_raw_buffer[:, -self.config.buffer_capacity:, :]
                    else:
                        current_buffer = global_raw_buffer  
                del gen_out
                t += chunk_size
            
            buffer_end_indices.append(global_raw_buffer.shape[1] if global_raw_buffer is not None else 0)

        # ================== 第二阶段：Batched Retriever 特征生成 ==================
        if N_turns == num_turns:
            retrieval_indices = buffer_end_indices
        else:
            final_mem_len = global_raw_buffer.shape[1] if global_raw_buffer is not None else 0
            retrieval_indices = [final_mem_len] * N_turns

        turn_embeddings = []
        prompt_vec_list = []
        max_batch_tokens = 2048 

        curr_idx = 0
        while curr_idx < N_turns:
            batch_seqs = []
            batch_max_len = 0
            
            while curr_idx < N_turns:
                turn_tokens = processed_turns[curr_idx]
                
                if native:
                    seq = self.get_emb_input_embeddings()(torch.tensor([turn_tokens], device=device))[0]
                else:
                    end_ptr = retrieval_indices[curr_idx]
                    t_emb = self.get_input_embeddings()(torch.tensor([turn_tokens], device=device))[0]
                    b_emb = self.mem_to_latent(global_raw_buffer[:, max(0, end_ptr-self.config.buffer_capacity):end_ptr, :])[0] if end_ptr > 0 else torch.empty(0, self.hidden_size, device=device, dtype=t_emb.dtype)
                    seq = torch.cat([b_emb, t_emb], dim=0)
                
                new_max_len = max(batch_max_len, seq.size(0))
                potential_tokens = (len(batch_seqs) + 1) * new_max_len
                
                if len(batch_seqs) > 0 and potential_tokens > max_batch_tokens:
                    break
                
                batch_seqs.append(seq)
                batch_max_len = new_max_len
                curr_idx += 1

            batched_embeds = torch.stack([
                torch.cat([torch.zeros(batch_max_len - s.size(0), seq.shape[-1], device=device, dtype=s.dtype), s]) 
                for s in batch_seqs
            ])
            batched_masks = torch.stack([
                torch.cat([torch.zeros(batch_max_len - s.size(0), device=device), torch.ones(s.size(0), device=device)]) 
                for s in batch_seqs
            ]).to(attention_mask.dtype)

            if native:
                out = run_transformer(self.emb_model, batched_embeds, batched_masks, adapter=None)
                p_vec = out[:, -1, :] # Emb model 取最后一个 Token
            else:
                ret_out = run_transformer(self.model, batched_embeds, batched_masks, adapter="retriever")
                p_vec = self.mem_to_emb(self.latent_to_mem(ret_out[:, -1:, :])).mean(dim=1)
                
            prompt_vec_list.append(p_vec)
            
        turn_embeddings = list(torch.cat(prompt_vec_list, dim=0))

        # ================== 第三阶段：计算对比得分与概率 ==================
        if len(turn_embeddings) < 2:
            return {"scores": None, "probs": None, "sorted_indices": None, "sorted_scores": None, "msg": "Insufficient turns for retrieval"}

        query_emb = turn_embeddings[-1]
        history_keys = torch.stack(turn_embeddings[:-1], dim=0)

        # Normalization (与 forward 完全对齐)
        query_emb_norm = torch.nn.functional.normalize(query_emb.unsqueeze(0).float(), p=2, dim=-1)
        history_keys_norm = torch.nn.functional.normalize(history_keys.unsqueeze(0).float(), p=2, dim=-1)

        # 计算得分矩阵 (除以温度 0.1)
        scores = torch.bmm(query_emb_norm.unsqueeze(1), history_keys_norm.transpose(1, 2)).squeeze(1).squeeze(0)
        scores = scores / 0.1

        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        probs = torch.softmax(scores, dim=-1) 
        result = {
            "scores": scores,               
            "probs": probs,                 
            "sorted_indices": sorted_indices, 
            "sorted_scores": sorted_scores
        }

        # ================== cache ==================
        if cache_path is not None:
            save_data = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in result.items()}
            torch.save(save_data, cache_path)
        # =======================================================

        return result
