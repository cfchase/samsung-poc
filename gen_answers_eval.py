from dotenv import load_dotenv
from datasets import Dataset, load_dataset
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.llms import VLLMOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
)

def generate_answers(testset, prompt, llm):
    dataset_df = testset.to_pandas()
    dataset_df["answer"] = None

    for index, row in dataset_df.iterrows():
        conversation = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=False
        )
        query_str = row["question"]
        context_str = "\n\n".join(row["contexts"])
        answer = conversation.predict(query_str=query_str, context_str=context_str)
        dataset_df.loc[index, "answer"] = answer

    return Dataset.from_pandas(dataset_df)


load_dotenv()

data_dir = os.getenv('DATA_DIR', "data")
testset_json = os.getenv("TESTSET_JSON", "testset.jsonl")

infer_endpoint = os.getenv("INFER_ENDPOINT")
model_name = os.getenv("MODEL_NAME")
api_key = os.getenv("LLM_API_KEY")

loader = DirectoryLoader(data_dir, loader_cls=TextLoader, glob="**/*.md")
documents = loader.load()
testset = Dataset.from_json(testset_json)


# LLM definition
llm = VLLMOpenAI(           # we are using the vLLM OpenAI-compatible API client. But the Model is running on OpenShift, not OpenAI.
    openai_api_key=api_key,   # and that is why we don't need an OpenAI key for this.
    openai_api_base= f"{infer_endpoint}/v1",
    model_name=f"{model_name}",
    temperature=0.00,
    max_tokens=2048
)

TEXT_QA_TEMPLATE = (
    "<|system|>\n"
    "You are, Red Hat Instruct Model based on Granite 7B, an AI language model developed by Red Hat and IBM Research, based on the Granite-7b-base language model. My primary function is to be a chat assistant.\n"
    "<|user|>\n"
    "{context_str}\n"
    "Given the above context information, answer the query.\n"
    "{query_str}\n"
    "<|assistant|>\n"
)


PROMPT = PromptTemplate(input_variables=["query_str", "context_str"], template=TEXT_QA_TEMPLATE)

answers_ds = generate_answers(testset, PROMPT, llm)

metrics = [
    faithfulness,
    answer_relevancy,
    answer_correctness,
]

eval_result = evaluate(
    answers_ds,
    metrics=metrics,
)

eval_result.to_pandas().to_csv("eval_result.csv", index=False, header=True)