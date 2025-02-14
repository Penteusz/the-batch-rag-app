from typing import Dict
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from config import OPENAI_API_KEY, OPENAI_MODEL_NAME, MODEL_TEMPERATURE


class CorrectnessEvaluation(BaseModel):
    explanation: str = Field(description="Explain your reasoning for the score")
    correct: bool = Field(description="True if the answer is factually correct")


SYSTEM_TEMPLATE = """You are evaluating the factual correctness of an AI assistant's answer compared to a reference answer.

Grade based on these criteria:
1. All factual claims in the assistant's answer must be accurate according to the reference answer
2. The assistant's answer should not contradict any information in the reference answer
3. The assistant's answer can include additional correct information not in the reference answer
4. The assistant's answer can use different wording as long as the meaning is preserved

You must respond with a valid JSON object containing 'explanation' and 'correct' fields.
Do not include any text before or after the JSON object.
The JSON must be properly formatted with double quotes around strings.

{format_instructions}"""

HUMAN_TEMPLATE = """QUESTION: {question}
REFERENCE ANSWER: {reference_answer}
ASSISTANT'S ANSWER: {assistant_answer}

Analyze the answers step by step, then provide your evaluation in the required JSON format."""


def evaluate_correctness(inputs: Dict, outputs: Dict, reference_outputs: Dict) -> bool:
    """
    Evaluate if the RAG system's answer is factually correct compared to the reference answer.
    
    Args:
        inputs: Dict containing the question
        outputs: Dict containing the generated answer
        reference_outputs: Dict containing the reference answer
        
    Returns:
        bool: True if answer is correct, False otherwise
    """
    parser = PydanticOutputParser(pydantic_object=CorrectnessEvaluation)
    
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
        reference_answer=reference_outputs['expected_answer'],
        assistant_answer=outputs['answer']
    )
    
    result = grader.invoke(formatted_prompt)
    try:
        evaluation = parser.parse(result.content)
        return evaluation.correct
    except Exception as e:
        print(f"Error parsing evaluation result: {str(e)}")
        print(f"Raw result: {result.content}")
        return False