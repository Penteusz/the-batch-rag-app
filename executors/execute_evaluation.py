import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langsmith import traceable

from config import (
    OPENAI_MODEL_NAME,
    MODEL_TEMPERATURE,
    VECTORSTORE_PATH,
    OPENAI_API_KEY,
    PROMPT_TEMPLATE,
    TOKEN_LIMIT,
    RETRIEVED_DOCS_COUNT
)

from evaluation.correctness import evaluate_correctness
from evaluation.groundedness import evaluate_groundedness
from evaluation.relevance import evaluate_relevance
from evaluation.retrieval_relevance import evaluate_retrieval_relevance
from evaluation.build_dataset import EvaluationDatasetManager

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
    model_name=OPENAI_MODEL_NAME,
    temperature=MODEL_TEMPERATURE,
    openai_api_key=OPENAI_API_KEY
)

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVED_DOCS_COUNT})

@traceable(name="rag_evaluation")
def rag_bot(question: str) -> Dict:
    """
    Process a question through the RAG pipeline and return the answer and retrieved documents.
    
    Args:
        question: User's question
        
    Returns:
        Dict containing answer and retrieved documents
    """

    docs = retriever.invoke(question)
    
    context_parts = []
    total_tokens = 0
    
    for doc in docs:
        text = doc.page_content
        tokens = len(llm.get_token_ids(text))
        
        if total_tokens + tokens > TOKEN_LIMIT:
            break
            
        context_parts.append(text)
        total_tokens += tokens
    
    context = "\n\n".join(context_parts)
    
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "query"])
    chain = prompt | llm | (lambda x: x.content)
    
    response = chain.invoke({"context": context, "query": question})
    
    return {"answer": response, "documents": docs}

def evaluate_rag_system():
    """Run comprehensive evaluation of the RAG system"""
    results = []
    
    dataset_manager = EvaluationDatasetManager()
    dataset = dataset_manager.get_or_create_dataset()
    dataset_manager.create_examples(dataset.id)
    
    examples = dataset_manager.get_examples()
    
    for example in examples:
        rag_output = rag_bot(example["question"])
        
        eval_inputs = {"question": example["question"]}
        eval_outputs = rag_output
        eval_reference = {"expected_answer": example["expected_answer"]}
        
        correctness_score = evaluate_correctness(eval_inputs, eval_outputs, eval_reference)
        groundedness_score = evaluate_groundedness(eval_inputs, eval_outputs)
        relevance_score = evaluate_relevance(eval_inputs, eval_outputs)
        retrieval_score = evaluate_retrieval_relevance(eval_inputs, eval_outputs)
        
        result = {
            "question": example["question"],
            "generated_answer": rag_output["answer"],
            "expected_answer": example["expected_answer"],
            "metrics": {
                "correctness": correctness_score,
                "groundedness": groundedness_score,
                "relevance": relevance_score,
                "retrieval_relevance": retrieval_score
            }
        }
        results.append(result)
        
        print(f"\nEvaluating: {example['question']}")
        print(f"Generated Answer: {rag_output['answer']}")
        print(f"Expected Answer: {example['expected_answer']}")
        print("Metrics:")
        print(f"- Correctness: {correctness_score}")
        print(f"- Groundedness: {groundedness_score}")
        print(f"- Relevance: {relevance_score}")
        print(f"- Retrieval Relevance: {retrieval_score}")
    
    return results

if __name__ == "__main__":
    print("Starting RAG system evaluation...")
    results = evaluate_rag_system()
    
    total_examples = len(results)
    overall_metrics = {
        "correctness": sum(r["metrics"]["correctness"] for r in results) / total_examples,
        "groundedness": sum(r["metrics"]["groundedness"] for r in results) / total_examples,
        "relevance": sum(r["metrics"]["relevance"] for r in results) / total_examples,
        "retrieval_relevance": sum(r["metrics"]["retrieval_relevance"] for r in results) / total_examples
    }
    
    print("\nOverall Evaluation Results:")
    for metric, score in overall_metrics.items():
        print(f"{metric.capitalize()}: {score:.2%}")