import asyncio
import copy
import hashlib
import os
import time

import httpx
import numpy as np
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from model.evo_embedding import EvoRAGConfig, EvoRAGModel


DEFAULT_EVOEMBEDDING_MODEL = "ClareNie/EvoEmbedding-4B"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


class OpenAIClient:
    def __init__(
        self,
        api_key=None,
        base_url=None,
        model_name=None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY or pass api_key explicitly.")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"Loading model {self.model_name}...")

    def send_message(self, message, retry_times=5):
        instructions = None
        for _ in range(retry_times):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.prepare_input(message),
                )
                instructions = response.choices[0].message.content
                if instructions:
                    break
            except Exception as exc:
                print(f"OpenAI API error: {exc}. Retrying...")
                time.sleep(1)
        if not instructions:
            raise ValueError("Failed to get a valid response after multiple retries.")
        return instructions

    def prepare_input(self, message):
        return [
            {
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}],
            }
            for msg in message
        ]


async def llm_openai_api(
    messages,
    model_name,
    ip="localhost",
    host="8080",
    temperature=0.0,
    max_tokens=128,
    top_p=None,
    n=1,
    **kwargs,
):
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "n": n,
    }
    payload.update(kwargs)

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        resp = await client.post(
            f"http://{ip}:{host}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return [choice["message"]["content"] for choice in data["choices"]]


class qwen3_client:
    def __init__(self, model_name=DEFAULT_BASE_MODEL):
        if "4b" in model_name.lower():
            model_name = DEFAULT_BASE_MODEL
        elif "30b" in model_name.lower():
            model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        self.model_name = model_name
        self.use_api = model_name == "Qwen/Qwen3-30B-A3B-Instruct-2507"

        if self.use_api:
            print(f"Using OpenAI-compatible API for {model_name}.")
        else:
            print(f"Loading local model {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )

    def send_message(self, message):
        if self.use_api:
            return asyncio.run(
                llm_openai_api(
                    messages=message,
                    model_name=self.model_name,
                    max_tokens=1024,
                )
            )[0]

        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)


_emb_model_cache = {}


def get_text_embedding(
    text,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    is_query=False,
    save_dir="./output/eval_results/emb_cache",
):
    cache_path = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        hash_obj = hashlib.md5()
        hash_obj.update(str(text).encode("utf-8"))
        hash_obj.update(model_name.encode("utf-8"))
        hash_obj.update(str(is_query).encode("utf-8"))
        cache_path = os.path.join(save_dir, f"cache_{hash_obj.hexdigest()}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

    if model_name not in _emb_model_cache:
        print(f"Loading embedding model: {model_name}...")
        _emb_model_cache[model_name] = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2"
                if "Qwen" in model_name
                else "sdpa",
            },
        )
    model = _emb_model_cache[model_name]

    text_list = [text] if isinstance(text, str) else text
    if is_query and "Qwen" in model_name:
        task = "Given a web search query, retrieve relevant passages that answer the query"
        text_list = [f"Instruct: {task}\nQuery:{t}" for t in text_list]

    encode_kwargs = {"convert_to_numpy": True, "normalize_embeddings": True}
    if model_name == "jinaai/jina-embeddings-v5-text-small":
        embeddings = model.encode(text_list, batch_size=4, task="retrieval", **encode_kwargs)
    else:
        embeddings = model.encode(text_list, batch_size=4, **encode_kwargs)

    result = embeddings if not isinstance(text, str) else embeddings.reshape(1, -1)
    if cache_path is not None:
        np.save(cache_path, result)
    return result


class EvoEmbeddingClient:
    def __init__(
        self,
        model_name="EvoEmbedding",
        model_path=DEFAULT_EVOEMBEDDING_MODEL,
        tokenizer_name=DEFAULT_BASE_MODEL,
        device="cuda",
    ):
        print(f"Loading EvoEmbedding model from {model_path}...")
        self.tokenizer = AutoProcessor.from_pretrained(tokenizer_name)
        config = EvoRAGConfig.from_pretrained(model_path)
        self.model = EvoRAGModel.from_pretrained(
            model_path,
            config=config,
            tokenizer=self.tokenizer,
        )
        self.model.to(dtype=torch.bfloat16, device=device)

        self.use_qwen_eval = "30b" in model_name.lower()
        if self.use_qwen_eval:
            self.eval_client = qwen3_client(model_name="Qwen/Qwen3-30B-A3B-Instruct-2507")

    def send_message(self, message):
        model_inputs = self._tokenize(message)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=False,
        )
        return self.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)

    def send_message_raw(self, message):
        if self.use_qwen_eval:
            return self.eval_client.send_message(message)
        model_inputs = self._tokenize(message)
        generated_ids = self.model.generate_raw(
            **model_inputs,
            max_new_tokens=256,
            do_sample=False,
        )
        input_len = model_inputs["input_ids"].shape[1]
        output_ids = generated_ids[0][input_len:].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def send_message_retrieve(self, message, rag_sentence_num, native=False, _sorted=True):
        model_inputs = self._tokenize(message)
        turns = []
        for turn_idx, msg in enumerate(message[:-1]):
            if turn_idx % 2 != 0:
                continue
            if turn_idx + 1 >= len(message):
                turns.append(f"User: {msg['content']}")
                continue
            assistant_msg = message[turn_idx + 1]
            if not msg["content"]:
                content = f"Assistant: {assistant_msg['content']}"
            elif not assistant_msg["content"]:
                content = f"User: {msg['content']}"
            else:
                content = f"User: {msg['content']}\nAssistant: {assistant_msg['content']}"
            turns.append(content)

        turns.append(message[-1]["content"].strip())
        meta = {"turns": [self.tokenizer(turn)["input_ids"] for turn in turns]}
        retrieve_results = self.model.generate_retrieve_idx(
            **copy.deepcopy(model_inputs),
            meta=meta,
            native=native,
        )
        retrieve_indices = retrieve_results["sorted_indices"].tolist()[:rag_sentence_num]
        return sorted(retrieve_indices) if _sorted else retrieve_indices

    def _tokenize(self, message):
        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer([text], return_tensors="pt").to(self.model.device)


EvoRAGClient = EvoEmbeddingClient
