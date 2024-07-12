import logging
import sys
import pandas as pd
# import nest_asyncio
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding, HuggingFaceInferenceAPIEmbedding
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
from typing import List
import tempfile
from llama_index.llms.gemini import Gemini
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

import os

GOOGLE_API_KEY = "AIzaSyCrDt5iXSHAyHOYvzv4IBTRTkxaXIxeMpg"  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


#! GROQ API KEY
API_KEY = "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"

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
    # nest_asyncio.apply()  # Apply nest_asyncio
    return data_generator.generate_questions_from_nodes()

# Create vector index from documents
def create_vector_index(documents):
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", token="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")
    return VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Evaluate responses
def evaluate_responses(questions, query_engine, evaluator):
    results = []
    count = 0
    for question in questions:
        if count >= 5:
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
        count += 1
    return results

# Main processing function
def process_documents(files, api_key):
    with tempfile.TemporaryDirectory() as input_dir:
        for file in files:
            file_path = os.path.join(input_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

        documents = load_documents(input_dir)
        llm, evaluator = initialize_evaluator(api_key)
        eval_questions = generate_questions(documents, llm)
        
        vector_index = create_vector_index(documents)
        query_engine = vector_index.as_query_engine(llm=llm)
        
        results = evaluate_responses(eval_questions, query_engine, evaluator)
    return results

# Create FastAPI app
app = FastAPI()

# API endpoint for uploading files and processing documents
@app.post("/evaluate")
def evaluate(files: List[UploadFile] = File(...)):
    try:
        results = process_documents(files, api_key=API_KEY)
        return JSONResponse(content={"results": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run this code to start the FastAPI app
# uvicorn api:app --reload 

