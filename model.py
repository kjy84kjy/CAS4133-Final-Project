# model.py

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class HFGenerator:
    def __init__(self, model_name, torch_dtype=torch.float16):
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()

        try:
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None
        except Exception:
            pass

        print("[Info] Model device map:")
        if hasattr(self.model, "hf_device_map"):
            print(self.model.hf_device_map)
        else:
            print("[Info] hf_device_map is not available.")

    def build_chat_prompt(self, user_prompt, system_prompt=None):
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. Follow the user's instruction exactly."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if getattr(self.tokenizer, "chat_template", None) is not None:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback for non-chat-template models
        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

    def generate(
        self,
        prompts,
        max_new_tokens=160,
        temperature=0.0,
        batch_size=8,
        show_progress=True,
        desc="Generating",
        system_prompt=None,
        use_chat_template=True,
    ):
        responses = []

        if use_chat_template:
            prompts = [
                self.build_chat_prompt(p, system_prompt=system_prompt)
                for p in prompts
            ]

        batches = [
            prompts[i:i + batch_size]
            for i in range(0, len(prompts), batch_size)
        ]

        iterator = batches

        if show_progress:
            iterator = tqdm(batches, desc=desc, total=len(batches))

        for batch_prompts in iterator:
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )

            inputs = {
                key: value.to(self.model.device)
                for key, value in inputs.items()
            }

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            if temperature > 0:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = 0.9
            else:
                generation_kwargs["do_sample"] = False
                generation_kwargs["temperature"] = None
                generation_kwargs["top_p"] = None
                generation_kwargs["top_k"] = None

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )

            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:]

            batch_responses = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )

            responses.extend([r.strip() for r in batch_responses])

        return responses