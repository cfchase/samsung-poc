from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import os
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# load OPENAI_API_KEY and DATA_DIR from .env file
load_dotenv()
data_dir = os.getenv('DATA_DIR', "data")
test_size = int(os.getenv('TEST_SIZE', "20"))

loader = DirectoryLoader(data_dir, loader_cls=TextLoader, glob="**/*.md")
documents = loader.load()

# generator with openai models
generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

distributions = {
    simple: 0.5,
    multi_context: 0.25,
    reasoning: 0.25
}

testset = generator.generate_with_langchain_docs(
    documents,
    test_size=test_size,
    distributions=distributions
)

testset.to_pandas().to_json("testset.jsonl")
