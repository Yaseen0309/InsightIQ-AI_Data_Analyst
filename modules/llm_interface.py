import re
from collections import deque
import pandas as pd
from modules.data_utils import describe_dataframe
import together  # or your preferred LLaMA backend
import streamlit as st

# Optional: for debugging
import logging


# Global variables
loaded_text = ""
loaded_df = None
chat_history = deque(maxlen=5)  # To keep track of recent Q&A

def ask_about_file(user_question: str, loaded_text=None, loaded_df=None):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = deque(maxlen=5)

    if loaded_df is not None:
        df_sample = loaded_df.head(5).to_string(index=False)
        schema = str(loaded_df.dtypes)
        context = f"The data has the following schema:\n{schema}\n\nHere is a preview:\n{df_sample}"
    elif loaded_text:
        context = loaded_text[:3000]
    else:
        return "⚠️ No data loaded to analyze."

    history = "\n".join([f"<|user|>{q}\n<|assistant|>{a}" for q, a in st.session_state.chat_history])
    full_prompt = f"<|system|>\nYou are a helpful analyst.\n{history}\n<|user|>{user_question}\nContext:\n{context}\n<|assistant|>"

    answer = llama_query(full_prompt)
    st.session_state.chat_history.append((user_question, answer))
    return answer

query_cache = {}

def llama_query(user_prompt: str, df: pd.DataFrame = None, system_prompt="You're a smart data analyst.", is_code=False, debug=False):
    model_name = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

    if is_code:
        system_prompt = "You are a Python data analyst. Answer only in valid Python code using `df`."
        if isinstance(df, pd.DataFrame):
            sample = df.head(5).to_string(index=False)
            system_prompt += f"\n\nSample:\n{sample}"

    elif isinstance(df, pd.DataFrame):
        system_prompt += f"\nThe dataset has the following columns:\n{', '.join(df.columns)}"

    full_prompt = f"<|system|>\n{system_prompt.strip()}\n<|user|>\n{user_prompt.strip()}\n<|assistant|>"

    if debug:
        print("\n======= PROMPT =======\n", full_prompt)

    try:
        response = together.Complete.create(
            model=model_name,
            prompt=full_prompt,
            max_tokens=512,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            stop=["<|user|>", "<|endoftext|>"]
        )

        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['text'].strip()
        else:
            return "⚠️ No valid response from the model."

    except Exception as e:
        return f"❌ Error during API call: {str(e)}"
    

def format_prompt(question, df_summary, df_columns, sample_rows):
    """
    Improved prompt for better LLaMA reasoning and code generation.
    Includes column names, summary, and sample rows.
    """
    prompt = f"""
You are a Python data analyst.

The dataset is available in a DataFrame called `df`.

Here is a summary:
{df_summary}

Here are the first 5 rows:
{sample_rows}

Column names: {', '.join(df_columns)}

Now answer the following question using valid Python code (wrapped in triple backticks):

{question}
"""
    return prompt.strip()


def extract_code_block(response, debug=False):
    """
    Extracts Python code block from LLaMA response.
    Handles markdown formatting and natural language mixing.
    Filters out non-ASCII characters and validates syntax.
    """
    import ast

    def is_valid_python(code):
        try:
            ast.parse(code)
            return True
        except:
            return False

    code = ""

    # 1. Try to extract from triple backticks
    if "```" in response:
        try:
            code_parts = response.split("```")
            for part in code_parts:
                if any(token in part for token in ['df', 'plot', '=', 'print']):
                    code = part.strip()
                    if code.startswith("python"):
                        code = code[len("python"):].strip()
                    break
        except Exception as e:
            print("⚠️ Error parsing markdown block:", e)

    # 2. Fallback: scan for code-like lines
    if not code:
        code_lines = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if any(kw in line for kw in ["=", "df[", "groupby(", "plot(", "print(", "sort_values("]):
                code_lines.append(line)
        if code_lines:
            code = "\n".join(code_lines)

    # 3. Strip emojis/non-ASCII characters
    code = code.encode("ascii", errors="ignore").decode().strip()

    # 4. Validate & return
    if code and is_valid_python(code):
        if debug:
            print("✅ Extracted Code:\n", code)
        return code
    else:
        if debug:
            print("❌ Invalid or no code found in response:\n", response)
        return None
