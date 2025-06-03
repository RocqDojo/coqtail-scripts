from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch

import os
from colorama import Fore, init
from peft import PeftModel, PeftConfig
init(autoreset=True)


base_model_path = os.environ.get("BASE_MODEL")
lora_model_path = os.environ.get("LORA_MODEL")
retriver_model_path = os.environ.get("RETRIVER_MODEL")
vector_db_path = os.environ.get("VECTOR_DB")
device = os.environ.get("TORCH_DEVICE")


class RandomMock:
    def __init__(self, *args, **kwargs) -> None:
        _ = args
        _ = kwargs
        import random

        self.choice = random.choice

    def query(self, _hypothesis: str, _conclusion: str):
        """
        get tactic hint for a proof step
        """
        return self.choice(
            [
                "constructor.",
                "split.",
                "left.",
                "right.",
                "assumption.",
                "reflexivity.",
                "f_equal.",
                "discriminate.",
                "contradiction.",
                "congruence.",
                "trivial.",
                "intuition.",
                "lia.",
            ]
        )


class InputMock:
    def __init__(self, *args, **kwargs) -> None:
        _ = args
        _ = kwargs

    def query(self, hypothesis: str, conclusion: str):
        """
        get tactic hint for a proof step
        """
        print()
        print(Back.YELLOW + "hypothesis:", hypothesis)
        print(Back.YELLOW + "conclusion:", conclusion)
        tactic = input(Back.YELLOW + "tactic hint > ")
        return tactic

rag_prompt_template = """
You are assisting in proving a theorem in Coq. Below is the current state of the proof, including the proof context, the goal, and related definitions of theorems or lemmas that might be useful. Your task is to generate the next step to apply in the proof script. Provide only a single step of valid Coq proof tactic, without any additional text or explanations.
### Proof Context:
{context}
### Proof Goal:
{goal}
### Related Definitions:
{retrived}
### Next Proof Step:
"""

# no RAG
norag_prompt_template = """
You are assisting in proving a theorem in Coq. Below is the current state of the proof, including the proof context and the goal. Your task is to generate the next step to apply in the proof script. Provide only a single step of valid Coq proof tactic, without any additional text or explanations.
### Proof Context:
{context}
### Proof Goal:
{goal}
### Next Proof Step:
"""

new_guided_prompt_template = """
## Backgrounds

You are an exprt in Coq theorem proving.
After reading the current proof state, including the proof context and the goal, your have to generate the next tactic.

## Instructions

Only provide a single tactic.
Do not repeat the provided input.
Do not add any prefix before the response.
Do not add suffix after the response.
Do not quote the response.
Do not enclose the response markdown code block.
Do not output in markdown syntax.
The response should have exactly one line.
The response should end with a period.


## Examples

Example1:
### Proof Context:
Hmatch1 : s1 =~ Char t
Hmatch2 : s2 =~ re2
IH1 : s1 =~ re_opt_e (Char t)
IH2 : s2 =~ re_opt_e re2
### Proof Goal:
s1 =~ re_opt_e (Char t)
### Reponse:
apply IH1.

Example2:
### Proof Context:

### Proof Goal:
0 = 0 \\/ (exists n' : nat, 0 = S (S n') /\\ ev n')
### Reponse:
left.

Example3:
### Proof Context:

### Proof Goal:
0 = 0
### Reponse:
reflexivity.

## Task

Now follow the instructions and work on this problem:

### Proof Context:
{context}
### Proof Goal:
{goal}
### Reponse:
"""

def extract_tactic(output: str) -> str:
    # print('[[model output]] BEGIN')
    # print(Fore.GREEN + output)
    # print('[[model output]] END')
    r = output.splitlines()
    return r[0] if len(r)>0 else 'LLM GIVES EMPTY OUTPUT'


class SftLlmQuery:
    # device: cuda:0, cuda:1, cuda:2, cuda:3, cpu
    def __init__(self):

        # 加载 FAISS 向量数据库
        print('loading retrieve model', retriver_model_path)
        embeddings = HuggingFaceEmbeddings(model_name=retriver_model_path)
        print('loading FAISS', vector_db_path)
        self.db_model = FAISS.load_local(
            vector_db_path, embeddings, allow_dangerous_deserialization=True
        )

        # 加载大语言模型
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16).to(device)
        lora_model = PeftModel.from_pretrained(base_model, lora_model_path, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.llm_model = lora_model

    def retrieve(self, proof_text: str):
        # 进行查询
        query = (
            ",".join(proof_text.split("\n"))
            .replace("context:", "")
            .replace("goal:", "")
        )

        # 你要证明的定理
        retrieved_docs = self.db_model.similarity_search_with_score(
            query, k=2
        )  # 进行相似性搜索

        # 输出检索结果（带相似度分数）
        # for idx, (doc, score) in enumerate(retrieved_docs, start=1):
        #    print(f"\n--- 相关证明 {idx} (相似度: {score:.4f}) ---")
        #    print(doc.page_content)

        retrieved_docs = [
            doc.page_content for (doc, score) in retrieved_docs if score < 15
        ]
        return retrieved_docs

    def query(self, proof_context: str, proof_goal: str):
        # with RAG
        proof_text = proof_context + proof_goal
        retrieved_docs = self.retrieve(proof_text)
        retrieved_info = "\n".join(retrieved_docs)

        new_prompt = rag_prompt_template.format(
            context=proof_context, goal=proof_goal, retrived=retrieved_info
        )
        '''
        # without RAG
        new_prompt = norag_prompt_template.format(
            context=proof_context, goal=proof_goal,
        )
        '''

        inputs = self.tokenizer(new_prompt, return_tensors="pt").to(device)
        output = self.llm_model.generate(
            **inputs,
            max_new_tokens=100,
            # if we set beams = 1, we will be using multi-sampling
            num_beams=1,  # Must be ≥ num_return_sequences
            num_return_sequences=10,
            do_sample=True,  # Enables stochastic beam search
            top_p=0.8,
            temperature=0.7,
        )

        return [
            extract_tactic(
                self.tokenizer.decode(o, skip_special_tokens=True)[len(new_prompt):].strip()
                )
            for o in output
        ]

        return extract_tactic(decoded_output)
