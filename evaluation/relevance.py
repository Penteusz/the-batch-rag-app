from typing import Dict
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from config import OPENAI_API_KEY, OPENAI_MODEL_NAME, MODEL_TEMPERATURE

class RelevanceEvaluation(BaseModel):
    explanation: str = Field(description="Explain your reasoning for the score")
    relevant: bool = Field(description="True if the answer is relevant to the question")

SYSTEM_TEMPLATE = """You are evaluating if an AI assistant's answer is relevant to the user's question.

Grade based on these criteria:
1. The answer should directly address the main points of the question
2. The answer should provide information that helps solve the user's query
3. The answer should stay focused on the question topic
4. The answer should not include irrelevant or off-topic information

A relevance score of True means the answer is relevant and helpful.
A relevance score of False means the answer is off-topic or unhelpful.

{format_instructions}

You must respond with a valid JSON object containing 'explanation' and 'relevant' fields."""

HUMAN_TEMPLATE = """QUESTION: {question}
ASSISTANT'S ANSWER: {assistant_answer}

Analyze the question and answer step by step, then provide your evaluation in the required JSON format."""

def evaluate_relevance(inputs: Dict, outputs: Dict) -> bool:
    """
    Evaluate if the RAG system's answer is relevant to the question.
    
    Args:
        inputs: Dict containing the question
        outputs: Dict containing the answer
        
    Returns:
        bool: True if answer is relevant, False otherwise
    """
    parser = PydanticOutputParser(pydantic_object=RelevanceEvaluation)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)
    ])
    
    grader = ChatOpenAI(
        model=OPENAI_MODEL_NAME,
        temperature=MODEL_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    formatted_prompt = prompt.format_messages(
        format_instructions=parser.get_format_instructions(),
        question=inputs['question'],
        assistant_answer=outputs['answer']
    )
    
    result = grader.invoke(formatted_prompt)
    evaluation = parser.parse(result.content)
    
    return evaluation.relevant