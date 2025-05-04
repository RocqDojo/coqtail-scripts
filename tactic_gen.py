import os
from colorama import Back

llm_model_path = os.environ.get("LLM_MODEL")
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


prompt_template = """
You are assisting in proving a theorem in Coq. Below is the current state of the proof, including the proof context, the goal, and related definitions of theorems or lemmas that might be useful. Your task is to generate the next step to apply in the proof script. Provide only a single step of valid Coq proof tactic, without any additional text or explanations.
### Proof Context:
{context}
### Proof Goal:
{goal}
### Related Definitions:
{retrived}
### Next Proof Step:
"""


def extract_tactic(output: str) -> str:
    prompt_endings = "### Next Proof Step:"

    real_output = output.split(prompt_endings, 1)[1]
    filtered_output = real_output.strip().split(".")[0] + "."
    return filtered_output


class SftLlmQuery:
    # device: cuda:0, cuda:1, cuda:2, cuda:3, cpu
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        import torch

        # 加载 FAISS 向量数据库
        embeddings = HuggingFaceEmbeddings(model_name=retriver_model_path)
        self.db_model = FAISS.load_local(
            vector_db_path, embeddings, allow_dangerous_deserialization=True
        )

        # 加载大语言模型
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_path, torch_dtype=torch.float16
        ).to(device)

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
        proof_text = proof_context + proof_goal
        retrieved_docs = self.retrieve(proof_text)
        retrieved_info = "\n".join(retrieved_docs)

        new_prompt = prompt_template.format(
            context=proof_context, goal=proof_goal, retrived=retrieved_info
        )

        inputs = self.tokenizer(new_prompt, return_tensors="pt").to(device)
        output = self.llm_model.generate(
            **inputs,
            max_new_tokens=100,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.85,
            temperature=0.3,
        )

        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return extract_tactic(decoded_output)
