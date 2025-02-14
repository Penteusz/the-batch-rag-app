from typing import Dict
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from config import OPENAI_API_KEY, OPENAI_MODEL_NAME, MODEL_TEMPERATURE

class RetrievalRelevanceEvaluation(BaseModel):
    explanation: str = Field(description="Explain your reasoning for the score")
    relevant: bool = Field(description="True if the retrieved documents are relevant to the question")

SYSTEM_TEMPLATE = """You are evaluating if the retrieved documents are relevant to the user's question.

Grade based on these criteria:
1. The documents should contain information relevant to answering the question
2. The documents should cover the main aspects of the question
3. The documents should not be primarily about unrelated topics
4. The combined documents should provide sufficient context to answer the question

A relevance score of True means the retrieved documents are helpful for answering the question.
A relevance score of False means the documents are not helpful or are off-topic.

{format_instructions}

You must respond with a valid JSON object containing 'explanation' and 'relevant' fields."""

HUMAN_TEMPLATE = """QUESTION: {question}
RETRIEVED DOCUMENTS: {source_docs}

Analyze the question and documents step by step, then provide your evaluation in the required JSON format."""

def evaluate_retrieval_relevance(inputs: Dict, outputs: Dict) -> bool:
    """
    Evaluate if the retrieved documents are relevant to the question.
    
    Args:
        inputs: Dict containing the question
        outputs: Dict containing the retrieved documents
        
    Returns:
        bool: True if documents are relevant, False otherwise
    """
    parser = PydanticOutputParser(pydantic_object=RetrievalRelevanceEvaluation)
    
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
        question=inputs['question'],
        source_docs=source_text
    )
    
    result = grader.invoke(formatted_prompt)
    evaluation = parser.parse(result.content)
    
    return evaluation.relevant