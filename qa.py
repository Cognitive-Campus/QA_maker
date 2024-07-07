import time
import logging
import sys
import pandas as pd
import nest_asyncio
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

import os

# GOOGLE_API_KEY = "AIzaSyDre4NzFItmCRbWJ6Z5ki3A6-ZcUC7kcoc"  # add your GOOGLE API key here
GOOGLE_API_KEY = "AIzaSyCrDt5iXSHAyHOYvzv4IBTRTkxaXIxeMpg"  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# Setup logging
def setup_logging():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load documents from directory
def load_documents(input_dir):
    reader = SimpleDirectoryReader(input_dir=input_dir)
    return reader.load_data()

# Initialize LLM and Evaluator
def initialize_evaluator(api_key):
    # llm = Groq(model="mixtral-8x7b-32768", api_key=api_key)
    # llm = Groq(model="llama3-8b-8192", api_key=api_key)
    # llm = Gemini(model="models/gemini-pro")
    llm = HuggingFaceInferenceAPI(
    model_name="mistralai/Mistral-7B-Instruct-v0.2", token="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"
    )
    evaluator = RelevancyEvaluator(llm=llm)
    return llm, evaluator

# Generate evaluation questions from documents
def generate_questions(documents, llm):
    data_generator = DatasetGenerator.from_documents(documents, llm=llm)
    nest_asyncio.apply()  # Apply nest_asyncio
    return data_generator.generate_questions_from_nodes()

# Create vector index from documents
def create_vector_index(documents):
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    # embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

    return VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Evaluate responses
def evaluate_responses(questions, query_engine, evaluator):
    results = []
    count=0
    questions = questions
    for question in questions:

        if count >= 15:
            return results

        response_vector = query_engine.query(question)
        eval_result = evaluator.evaluate_response(query=question, response=response_vector)
        
        result_dict = {
            "Query": question,
            "Response": str(response_vector),
            "Source": response_vector.source_nodes[0].node.get_content()[:1000] + "...",
            "Eval_result_feedback": eval_result.feedback,
            "Score": eval_result.score,
            "Pairwise source": eval_result.pairwise_source,
        }
        results.append(result_dict)
        count = count + 1
    return results

def main():
    print("starrttt....")
    start_time = time.time()  # Record the start time
    
    setup_logging()
    input_dir = "./data"
    api_key = "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"
    
    documents = load_documents(input_dir)
    llm, evaluator = initialize_evaluator(api_key)
    eval_questions = generate_questions(documents, llm)

    eval_questions_filtered = []


    eval_questions_df = pd.DataFrame(eval_questions)

    vector_index = create_vector_index(documents)
    query_engine = vector_index.as_query_engine(llm=llm)
    
    results = evaluate_responses(eval_questions, query_engine, evaluator)
    
    eval_df = pd.DataFrame(results)
    eval_df.to_json("eval_results.json", orient='records', lines=True)
    eval_df.to_csv("eval_results.csv", index=False)
    
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Execution time: {elapsed_time:.2f} seconds")

# if __name__ == "__main__":
main()
