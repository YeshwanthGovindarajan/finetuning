import streamlit as st
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time

st.title("Twitter Bio Generator 👇")

st.markdown(
    """Welcome to the Twitter Bio Generator! 
    To get started, enter your profession from the choices given below,
    and let the AI generate a unique Twitter bio for you."""
)

col1, col2 = st.columns((3, 1))
with col1:
    user_data = st.text_input('Enter Your Choice:', placeholder='E.g., Blockchain Developer')
with col2:
    generate_button = st.button('Generate Bio')

if generate_button and user_data:
    st.write("Please wait, generating the bio...")
    model_name = "Yeshwanth-03-06-2004/gpt2-tweetgen"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    text2text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = f"Give me a twitter bio for: {user_data}"

    generated_text = text2text_generator(prompt, max_length=280)

    with st.expander("Generated Tweet", expanded=True):
        st.markdown("#### Bio:")
        st.write(generated_text[0]['generated_text'])
else:
    st.info('Enter a profession above and click "Generate Bio" to see the AI-generated Twitter bio.')
