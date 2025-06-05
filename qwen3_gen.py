from numpy import inf
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from queue import PriorityQueue


class LLM:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", think: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.think = think
        self.model.eval()

    def generate(self, prompt: str, samples: int = 5) -> list[tuple[str, float]]:
        # Step 1: Prepare chat-formatted input
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.think,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]

        # Step 2: Generate multiple samples
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            max_new_tokens=100,
            num_return_sequences=samples,
            return_dict_in_generate=True,
            output_scores=True,
        )

        sequences = outputs.sequences  # [samples, total_len]
        generated_texts = []

        # Step 3: Get log-probs using model forward pass
        with torch.no_grad():
            logits = self.model(sequences).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = sequences[:, 1:]

            log_probs_tensor = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs_tensor.gather(
                2, shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            # Remove prompt logprobs
            gen_token_log_probs = []
            for i in range(samples):
                gen_logprob = token_log_probs[i, prompt_len - 1 :]  # -1 for shift
                gen_token_log_probs.append(gen_logprob)

        # Step 4: Decode sequences and collect results
        for i in range(samples):
            generated = sequences[i, prompt_len:]  # exclude prompt
            text_out = self.tokenizer.decode(generated, skip_special_tokens=True)
            total_log_prob = gen_token_log_probs[i].sum().item()
            generated_texts.append((text_out.strip(), total_log_prob))

        return generated_texts


def example():
    """
    simple LLM guided search algorithm example
    """

    llm = LLM()

    prompt = 'Directly generate a number between 0 and 100 without any other text'

    q: PriorityQueue[tuple[float, int, int]] = PriorityQueue()

    # (score, state, steps)
    q.put((0, 0, 0))
    while not q.empty():
        acc_score, state, steps = q.get()
        print(f"currently at (state={state}, score={acc_score}, steps={steps})")

        if state > 100:
            print("a path ending a state greater than 100 is found")
            return

        outputs = llm.generate(prompt)
        for step, score in outputs:
            try:
                new_score = acc_score - score
                new_state = state + int(step.splitlines()[0].strip())
                new_steps = steps + 1
                next_node = (new_score, new_state, new_steps)
                if new_score != inf:
                    print(
                        f"    take {step} with score {score} => new state {next_node}"
                    )
                    q.put(next_node)
            except:
                pass


if __name__ == "__main__":
    example()
