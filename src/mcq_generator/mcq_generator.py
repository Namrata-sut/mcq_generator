import os
import json
import importlib.resources as pkg_resources
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
import config

# Load environment variables
load_dotenv()

with pkg_resources.open_text(config, "response.json") as f:
    response_json = json.load(f)

# Access api key
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize model
model = init_chat_model("gemini-2.5-pro", model_provider="google_genai")

# Define the template for quiz generation
template1 = """
    Text:{text}
    You are an expert MCQ maker. Given the above text, it is your job to \
    create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. 
    Make sure the questions are not repeated and check all the questions to be conforming to the text as well.
    Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
    Ensure to make {number} MCQs
    QUIZ:
        # RESPONSE_JSON
        {response_json}
"""
quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=template1
)

# Create parser
str_parser = StrOutputParser()

# Create quiz generation chains
quiz_chain = (quiz_generation_prompt | model | str_parser).with_config(verbose= True)

# Define the template for quiz evaluation/review
template2= """
    You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
    You need to evaluate the complexity of the question and give a complete analysis of the quiz. Use at most 50 words for complexity analysis. 
    If the quiz is not at par with the cognitive and analytical abilities of the students,\
    update the quiz questions which need to be changed and adjust the tone such that it perfectly fits the student abilities.
    QUIZ MCQs:
    {quiz}

    Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=template2
)

# Create evaluation chains
review_chain = (quiz_evaluation_prompt | model | str_parser).with_config(verbose= True)

# Runnable wrapper to rename quiz_chain output key to 'quiz' so it matches review_chain input
class RenameOutputRunnableWithSubject(Runnable):
    def __init__(self, runnable, output_key, extra_vars: dict):
        self.runnable = runnable
        self.output_key = output_key
        self.extra_vars = extra_vars  # e.g. {"subject": SUBJECT}

    def invoke(self, inputs, config=None, **kwargs):
        output = self.runnable.invoke(inputs, config=config, **kwargs)
        # Put output string under output_key, add extra_vars
        data = {self.output_key: output}
        data.update(self.extra_vars)
        return data

SUBJECT ="General"
# use this wrapper instead of just RenameOutputRunnable
quiz_chain_mapped = RenameOutputRunnableWithSubject(
    quiz_chain,
    output_key="quiz",
    extra_vars={"subject": SUBJECT})

# Final chain: quiz generation -> quiz review
generate_evaluate_chain = quiz_chain_mapped | review_chain




