import os

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(
    page_title="GenAI Code Assistant",
    page_icon="💻",
    layout="wide"
)

st.title("💻 GenAI Code Assistant")
st.caption("Generate, explain, debug, and optimize Python code.")

# Read the key from local .env or cloud secrets.
def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


api_key = get_api_key()

if not api_key:
    st.error(
        "GROQ_API_KEY is missing. Add it to your local .env "
        "file or the hosting platform's Secrets settings."
    )
    st.stop()


@st.cache_resource
def load_llm(key):
    return ChatGroq(
        api_key=key,
        model="openai/gpt-oss-20b",
        temperature=0
    )


try:
    llm = load_llm(api_key)
except Exception as e:
    st.error(f"Could not initialize the language model: {e}")
    st.stop()


PROMPTS = {
    "Generate Code": """
You are an expert Python developer.

Requirement:
{user_input}

Generate readable, well-structured Python code.
Include a brief explanation and an example.
""",

    "Explain Code": """
You are a Python instructor.

Explain the following code in simple language:
{user_input}

Explain its purpose, important lines, and example output
where appropriate.
""",

    "Debug Code": """
You are a Python debugging assistant.

Review this code:
{user_input}

Identify actual errors or likely bugs, explain their causes,
and provide corrected code. If no definite bug is visible,
explain potential edge cases instead of inventing errors.
""",

    "Optimize Code": """
You are a Python code reviewer.

Review this code:
{user_input}

Suggest improvements to readability, performance, and
maintainability. Provide an improved version and explain
the changes. Preserve the original behavior where possible.
"""
}

task = st.selectbox(
    "Choose a task",
    [
        "Generate Code",
        "Explain Code",
        "Debug Code",
        "Optimize Code"
    ]
)

user_input = st.text_area(
    "Enter a requirement or paste your code",
    height=240,
    placeholder="Enter your Python requirement or code here..."
)

if st.button("Submit", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a requirement or code.")
    else:
        prompt = PROMPTS[task].format(user_input=user_input)

        try:
            with st.spinner("Generating response..."):
                response = llm.invoke(prompt)

            st.subheader("Assistant Response")
            st.markdown(response.content)

        except Exception:
            st.error(
                "The model request failed. Check your API key, "
                "model access, network connection, and provider limits."
            )

with st.sidebar:
    st.header("About")
    st.write("**Application:** GenAI Code Assistant")
    st.write("**Interface:** Streamlit")
    st.write("**Model provider:** Groq")
    st.write("**Features:** Generate, explain, debug, optimize")
    st.info("Review generated code before running it.")
