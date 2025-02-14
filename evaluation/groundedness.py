from typing import Dict
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from config import OPENAI_API_KEY, OPENAI_MODEL_NAME, MODEL_TEMPERATURE

class GroundednessEvaluation(BaseModel):
    explanation: str = Field(description="Explain your reasoning for the score")
    grounded: bool = Field(description="True if the answer is supported by the source documents")

SYSTEM_TEMPLATE = """You are evaluating if an AI assistant's answer is grounded in the provided source documents.

Grade based on these criteria:
1. All factual claims in the answer must be supported by the source documents
2. The answer should not contain information that cannot be derived from the sources
3. The answer can combine or rephrase information from sources, but cannot introduce new facts

A groundedness score of True means the answer is fully supported by the sources.
A groundedness score of False means the answer contains unsupported claims.

{format_instructions}

You must respond with a valid JSON object containing 'explanation' and 'grounded' fields."""

HUMAN_TEMPLATE = """SOURCE DOCUMENTS: {source_docs}
ASSISTANT'S ANSWER: {assistant_answer}

Analyze the answer and sources step by step, then provide your evaluation in the required JSON format."""

def evaluate_groundedness(inputs: Dict, outputs: Dict) -> bool:
    """
    Evaluate if the RAG system's answer is grounded in the source documents.
    
    Args:
        inputs: Dict containing the question
        outputs: Dict containing the answer and source documents
        
    Returns:
        bool: True if answer is grounded, False otherwise
    """
    parser = PydanticOutputParser(pydantic_object=GroundednessEvaluation)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)
    ])
    
    grader = ChatOpenAI(
        model=OPENAI_MODEL_NAME,
        temperature=MODEL_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    source_text = " ".join(doc.page_content for doc in outputs["documents"])
    
    formatted_prompt = prompt.format_messages(
        format_instructions=parser.get_format_instructions(),
        source_docs=source_text,
        assistant_answer=outputs['answer']
    )
    
    result = grader.invoke(formatted_prompt)
    evaluation = parser.parse(result.content)
    
    return evaluation.grounded