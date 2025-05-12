import os
from colorama import Back

llm_model_path = os.environ.get("LLM_MODEL")
retriver_model_path = os.environ.get("RETRIVER_MODEL")
vector_db_path = os.environ.get("VECTOR_DB")
device = os.environ.get("TORCH_DEVICE")
max_samples = int(os.environ.get("MAX_SAMPLES", "10"))


class RandomMock:
    def __init__(self, *args, **kwargs) -> None:
        _ = args
        _ = kwargs
        import random

        self.choice = random.choice

    def query_one(self, _hypothesis: str, _conclusion: str):
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

    def query(self, _hypothesis: str, _conclusion: str):
        return [self.query_one(_hypothesis, _conclusion) for _ in range(max_samples)]


class InputMock:
    def __init__(self, *args, **kwargs) -> None:
        _ = args
        _ = kwargs

    def query_one(self, hypothesis: str, conclusion: str):
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
    return output.splitlines()[0]


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

        # decode to get 10 reponses at once
        # https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=1,  # do not use beam search
            num_return_sequences=10,
            do_sample=True,  # generate multiple reponses
            top_p=0.8,
            temperature=0.7,
        )

        def decode(out):
            decoded = self.tokenizer.decode(o, skip_special_tokens=True)
            return decoded[len(new_prompt) :].strip()

        return [extract_tactic(decode(out)) for out in outputs]
