from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

model = ChatOpenAI(
    model = "openai/gpt-oss-120b",
    openai_api_key = os.getenv("GROQ_API_KEY"),
    openai_api_base = "https://api.groq.com/openai/v1"
)

# UI
st.header("Research Tool")

paper_input = st.selectbox("Select research paper name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explaination style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explaination length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

#load the template.json file
template = load_prompt('template.json')

# Invoke and Result
if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)