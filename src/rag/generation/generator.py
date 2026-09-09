# src/rag/generation/generator.py
import os
import yaml
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ManualGenerator:
    """基于本地 LLM 的生成器，支持 4-bit 量化以节省显存。"""

    def __init__(self, model_name="Qwen2.5-7B-Instruct", device="cuda"):
        self.model_path = self._get_model_path(model_name)
        self.device = device
        self.tokenizer = None
        self.model = None

    def _get_model_path(self, model_name: str):
        with open('configs/path.yaml', 'r') as f:
            path = yaml.safe_load(f)
        
        return path['models'][model_name]
        

    def load_model(self):
        if self.model is not None:
            return

        print(f"Loading model: {self.model_path} (4-bit quantized)...")

        # 配置 4-bit 量化，极大节省显存 (12GB 显卡必备)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        print("Model loaded successfully.")

    def format_context(self, retrieval_results):
        """将检索结果格式化为 LLM 易读的 Context 字符串。"""
        context_parts = []
        for i, res in enumerate(retrieval_results):
            part = (
                f"[Document {i + 1}]\n"
                f"Source: {res['metadata'].get('manual_id', 'Unknown')} | "
                f"Section: {res['metadata'].get('section', 'Unknown')} | "
                f"Pages: {res['metadata'].get('page_start', '?')}-{res['metadata'].get('page_end', '?')}\n"
                f"Content: {res['text']}\n"
            )
            context_parts.append(part)
        return "\n---\n".join(context_parts)

    def generate(self, query, retrieval_results, max_new_tokens=512):
        if self.model is None:
            self.load_model()

        context = self.format_context(retrieval_results)

        # 精心设计的 Prompt，强制 LLM 基于 Context 回答并带上引用
        system_prompt = (
            "You are an expert aircraft maintenance engineer specializing in A320 air conditioning systems. "
            "Your task is to answer the user's query based *strictly* on the provided manual excerpts. "
            "If the answer is not in the context, say 'I cannot find the specific procedure in the provided excerpts.' "
            "Always cite the source using the format [Doc X] at the end of each key step."
        )

        user_prompt = (
            f"Query: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Please provide a structured response including:\n"
            "1. Probable Cause (if applicable)\n"
            "2. Recommended Remedial Action / Procedure\n"
            "3. Reference Manuals and Sections"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 使用 Qwen 的 chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        print("Generating response...")
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,  # 降低温度，使维修建议更严谨
            top_p=0.9,
        )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response
