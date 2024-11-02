import streamlit as st
from openai import OpenAI
import os
import langchain
from langchain.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch

st.title("‍✈️ Airline Customer Service")
prompt = st.text_input("Share with us your experience of the latest trip.", "It was awesome!")


### Load your API Key
my_secret_key = st.secrets['MyOpenAIKey']
os.environ["OPENAI_API_KEY"] = my_secret_key

### Create the LLM API object
llm = OpenAI(openai_api_key=my_secret_key)
#llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4")

### Create a template to handle the case where customer feedback is given.
airline_template = """You are an expert at handling airline customer feedback regarding their flight experiences.
From the following text, determine whether the flight experience is positive or negative. If negative, determine if it was caused by the airline (negative controlled) or beyond the airline's control (negative uncontrolled).

Do not respond with more than two words.

Text:
{request}

"""

### Create the decision-making chain

flight_type_chain = (
    PromptTemplate.from_template(airline_template)
    | llm
    | StrOutputParser()
)

negativecontrolled_chain = PromptTemplate.from_template(
    """You are a airline customer service officer that is experienced with handling customer feedback after trips. \
    Given the text below, offer sympathies and inform the user that customer service will contact them soon to resolve the issue or provide compensation.
    Do not respond with any reasoning. Just respond professionally as a customer service officer. Respond in first-person mode.

    Your response should follow these guidelines:
    1. Do not provide any reasoning behind the negative experience. Just respond professionally as a customer service officer in one short paragraph.
    2. Address the customer directly in first-person.
    3. Sign off the response simply as Andy, no role or title.

Text:
{text}

"""
) | llm


negativeuncontrolled_chain = PromptTemplate.from_template(
    """You are a airline customer service officer that is experienced with handling customer feedback after trips. \
    Given the text below, offer sympathies but explain that the airline is not liable in such situations.
    Do not respond with any reasoning. Just respond professionally as a customer service officer. Respond in first-person mode.

    Your response should follow these guidelines:
    1. Do not provide any reasoning behind the negative experience. Just respond professionally as a customer service officer in one short paragraph.
    2. Address the customer directly in first-person.
    3. Sign off the response simply as Bobbie, no role or title.

Text:
{text}

"""
) | llm


positive_chain = PromptTemplate.from_template(
    """You are a airline customer service officer.
    Given the text below, thank the customer for their feedback and for choosing to fly with the airline.

    Your response should follow these guidelines:
    1. You will thank the customer for their feedback and for choosing to fly with the airline.
    2. Do not respond with any reasoning. Just respond professionally as a airline customer service officer in one short paragraph.
    3. Address the customer directly.
    4. Sign off the response simply as Celeste, no role or title.

Text:
{text}

"""
) | llm


### Routing/Branching chain
branch = RunnableBranch(
    (
        lambda x: "negative controlled" in x["flight_type"].lower(),
        negativecontrolled_chain,
    ),
    (
        lambda x: "negative uncontrolled" in x["flight_type"].lower(),
        negativeuncontrolled_chain,
    ),
    positive_chain,
)

### Put all the chains together
full_chain = {"flight_type": flight_type_chain, "text": lambda x: x["request"]} | branch

langchain.debug = False

response = full_chain.invoke({"request": prompt})


### Display
st.write(response)
