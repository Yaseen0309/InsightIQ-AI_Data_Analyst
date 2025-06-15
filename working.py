# 🚀 Core Libraries
import os
import io
import re
import ast
import time
import contextlib
import warnings
from typing import Union
from collections import deque
from io import BytesIO

# 🧠 Data & Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 📂 File Handling
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import pdfplumber
import docx
from docx import Document

# 🌐 API Interaction
import requests
import together
from dotenv import load_dotenv

# 🧹 Utilities
import difflib


# Clean environment setup
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Proper key loading
os.environ['TOGETHER_API_KEY'] = "ce8d4c04ff7b5d7f2f0b3415584561546b7fa53a0fb0ba9e9baa00106b161599"


def load_file(file_path):
    ext = os.path.splitext(file_path)[1].lower().strip('.')

    if ext == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            return '\n'.join([page.extract_text() or '' for page in pdf.pages]), None

    elif ext == 'txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), None
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='ISO-8859-1') as f:
                return f.read(), None

    elif ext in ['doc', 'docx']:
        return extract_text_from_docx(file_path), None

    elif ext in ['csv', 'xlsx']:
        try:
            if ext == 'csv':
                try:
                    df = pd.read_csv(file_path)
                except UnicodeDecodeError:
                    print("⚠️ UTF-8 failed, trying ISO-8859-1...")
                    df = pd.read_csv(file_path, encoding='ISO-8859-1')
            else:
                df = pd.read_excel(file_path)

            df, col_map = normalize_columns(df)
            return df, col_map

        except Exception as e:
            print("❌ Error reading tabular file:", e)
            raise e

    elif ext in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
        return extract_text_from_image(file_path), None

    else:
        print(f"🚨 DEBUG: File extension = .{ext}")
        raise ValueError(f"Unsupported file type: .{ext}")



def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        return '\n'.join(full_text)
    except Exception as e:
        return f"❌ Failed to read .docx file: {e}"


def extract_text_from_image(file_path):
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f"❌ Failed to extract text from image: {e}"
    


#Data analysis Function
def describe_dataframe(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    buffer.write("📊 Dataset Overview:\n")
    buffer.write(f"- Shape: {df.shape}\n")
    buffer.write(f"- Columns: {list(df.columns)}\n\n")
    buffer.write("🔍 Data Types:\n")
    buffer.write(str(df.dtypes))
    buffer.write("\n\n🧮 Descriptive Statistics:\n")
    buffer.write(str(df.describe(include='all')))
    
    return buffer.getvalue()

def extract_columns_from_prompt(prompt: str, df: pd.DataFrame):
    cols = df.columns.str.lower()
    found = [col for col in cols if any(word in prompt.lower() for word in col.split())]
    return list(set(found))


# Global variables
loaded_text = ""
loaded_df = None
chat_history = deque(maxlen=5)  # To keep track of recent Q&A

def ask_about_file(user_question: str, loaded_text=None, loaded_df=None):
    global chat_history

    if loaded_df is not None:
        df_sample = loaded_df.head(5).to_string(index=False)
        schema = str(loaded_df.dtypes)
        context = f"The data has the following schema:\n{schema}\n\nHere is a preview:\n{df_sample}"
    elif loaded_text:
        context = loaded_text[:3000]
    else:
        return "⚠️ No data loaded to analyze."

    prompt = "\n".join([f"<|user|>{q}\n<|assistant|>{a}" for q, a in chat_history])
    full_query = f"{prompt}\n<|user|>{user_question}\nContext:\n{context}\n<|assistant|>"

    answer = llama_query(full_query)
    chat_history.append((user_question, answer))
    return answer


query_cache = {}

def llama_query(user_prompt: str, df: pd.DataFrame = None, system_prompt="You're a smart data analyst.", is_code=False):
    """
    Sends a prompt to LLaMA or a specialized model, with optional DataFrame context.
    If is_code=True, routes to a code-optimized LLM like CodeLlama or Starcoder2.
    """
    model_name = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    
    

    if isinstance(df, pd.DataFrame):
        columns_summary = ", ".join(df.columns)
        system_prompt += f"\nThe dataset has the following columns:\n{columns_summary}"

    full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>"

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




def extract_plot_info_from_llama(query, df_columns):
    chart_types = [
        "line", "bar", "stacked_bar", "grouped_bar", "pie", "scatter", "histogram", "box", "violin",
        "heatmap", "area", "count", "pairplot"
    ]

    # Add special keywords detection (fallback)
    fallback_chart = None
    lowered_query = query.lower()
    if "stacked" in lowered_query and "bar" in lowered_query:
        fallback_chart = "stacked_bar"
    elif "grouped" in lowered_query and "bar" in lowered_query:
        fallback_chart = "grouped_bar"

    prompt = f"""
<|system|>
You are a helpful data analyst. From the user’s chart request, extract:
1. chart_type (line, bar, stacked_bar, grouped_bar, pie, scatter, histogram, box, violin, heatmap, area, count, pairplot)
2. x-axis column name (choose from: {list(df_columns)} or "none")
3. y-axis column name (choose from: {list(df_columns)} or "none")
4. Optional aggregation (sum, mean, count) if grouping is implied, otherwise "none".

Format:
chart_type: <type>
x: <x_column or none>
y: <y_column or none>
agg: <aggregation or none>
<|user|>
{query}
<|assistant|>
"""

    response = llama_query(prompt)

    try:
        chart_type = re.search(r'chart_type:\s*(\w+)', response, re.IGNORECASE)
        x_col = re.search(r'x:\s*(.+)', response, re.IGNORECASE)
        y_col = re.search(r'y:\s*(.+)', response, re.IGNORECASE)
        agg_func = re.search(r'agg:\s*(\w+)', response, re.IGNORECASE)

        chart_type = chart_type.group(1).strip().lower() if chart_type else fallback_chart
        x_col = x_col.group(1).strip() if x_col else None
        y_col = y_col.group(1).strip() if y_col else None
        agg_func = agg_func.group(1).strip().lower() if agg_func else None

        if chart_type and chart_type.lower() == "none":
            chart_type = fallback_chart

        return chart_type, x_col, y_col, agg_func

    except Exception as e:
        print(f"⚠️ Parsing failed: {e}")
        print("🧠 Raw response from LLaMA:\n", response)
        return fallback_chart, None, None, None

    
def handle_plot_query(query, df):
    chart_type, x_col, y_col, agg_func = extract_plot_info_from_llama(query, df.columns)

    # Fallback chart type detection
    lowered_query = query.lower()
    if not chart_type:
        if "stacked" in lowered_query and "bar" in lowered_query:
            chart_type = "stacked_bar"
        elif "grouped" in lowered_query and "bar" in lowered_query:
            chart_type = "grouped_bar"
        elif "trend" in lowered_query and "monthly" in lowered_query:
            chart_type = "line"

    if not chart_type:
        return "❌ Couldn't confidently interpret your chart request. Try rephrasing."

    df_plot = df.copy()

    # Ensure datetime is parsed for any time-aware plotting
    if x_col and 'date' in x_col.lower():
        df_plot[x_col] = pd.to_datetime(df_plot[x_col], errors='coerce')

    plt.figure(figsize=(10, 6))

    try:
        if chart_type == 'stacked_bar':
            if 'region' in df_plot.columns and 'category' in df_plot.columns and 'sales' in df_plot.columns:
                pivot_df = df_plot.pivot_table(index='region', columns='category', values='sales', aggfunc='sum').fillna(0)
                pivot_df.plot(kind='bar', stacked=True)
            else:
                return "❌ Required columns for stacked bar not found."

        elif chart_type == 'grouped_bar':
            if 'region' in df_plot.columns and 'category' in df_plot.columns and 'sales' in df_plot.columns:
                pivot_df = df_plot.pivot_table(index='region', columns='category', values='sales', aggfunc='sum').fillna(0)
                pivot_df.plot(kind='bar')
            else:
                return "❌ Required columns for grouped bar not found."

        elif chart_type == 'line' and x_col and y_col and pd.api.types.is_datetime64_any_dtype(df_plot[x_col]):
            df_plot[x_col] = pd.to_datetime(df_plot[x_col], errors='coerce')
            df_plot = df_plot.dropna(subset=[x_col, y_col])

            # Group by month
            df_plot['period'] = df_plot[x_col].dt.to_period('M').dt.to_timestamp()

            # Smart priority grouping
            group_col = None
            priority_cols = ["region", "segment", "category", "ship_mode"]
            for col in priority_cols:
                if col in df_plot.columns and df_plot[col].nunique() <= 10:
                    group_col = col
                    break
            if not group_col:
                for col in df_plot.columns:
                    if col not in [x_col, y_col] and df_plot[col].nunique() <= 10 and df_plot[col].dtype == 'object':
                        group_col = col
                        break

            if group_col:
                summary = df_plot.groupby(['period', group_col])[y_col].sum().reset_index()
                for key in summary[group_col].unique():
                    subset = summary[summary[group_col] == key]
                    plt.plot(subset['period'], subset[y_col], label=str(key))
                plt.legend()
            else:
                summary = df_plot.groupby('period')[y_col].sum().reset_index()
                plt.plot(summary['period'], summary[y_col], marker='o')

        elif chart_type == 'bar' and x_col and y_col:
            sns.barplot(x=x_col, y=y_col, data=df_plot)

        elif chart_type == 'scatter' and x_col and y_col:
            plt.scatter(df_plot[x_col], df_plot[y_col])

        elif chart_type == 'heatmap':
            corr = df_plot.select_dtypes(include='number').corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm')

        elif chart_type == 'box' and x_col and y_col:
            sns.boxplot(x=x_col, y=y_col, data=df_plot)

        elif chart_type == 'violin' and x_col and y_col:
            sns.violinplot(x=x_col, y=y_col, data=df_plot)

        elif chart_type == 'pie' and x_col and y_col:
            df_plot.set_index(x_col)[y_col].plot.pie(autopct='%1.1f%%')

        else:
            return f"❌ Unsupported or unimplemented chart type: {chart_type}"

        plt.title(f'{chart_type.replace("_", " ").title()} Chart')
        if x_col: plt.xlabel(x_col)
        if y_col: plt.ylabel(y_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True)

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf

    except Exception as e:
        return f"❌ Plotting failed: {e}"



import re

import ast

def extract_code_block(response):
    """
    Extracts Python code block from LLaMA response.
    Handles markdown formatting and natural language mixing.
    Filters out non-ASCII characters and validates syntax.
    """
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
                if 'df' in part or 'plot' in part or 'print' in part or '=' in part:
                    code = part
                    break
            if code.startswith("python"):
                code = code[len("python"):].strip()
        except Exception as e:
            print("⚠️ Error parsing markdown block:", e)

    # 2. Fallback: scan for lines that look like code
    if not code:
        code_lines = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if any(keyword in line for keyword in ["=", "df[", "groupby(", "plot(", "print(", "sort_values("]):
                code_lines.append(line)
        if code_lines:
            code = "\n".join(code_lines)

    # 3. Clean up non-ASCII characters (like emojis)
    code = code.encode("ascii", errors="ignore").decode().strip()

    # 4. Validate syntax
    if code and is_valid_python(code):
        return code
    else:
        print("❌ Invalid or no code found in response.")
        return None
    
def normalize_columns(df: pd.DataFrame):
    """
    Normalize column names and return original-to-normalized mapping.
    """
    original_cols = df.columns
    normalized_cols = [col.strip().lower().replace(" ", "_") for col in original_cols]
    col_map = dict(zip(normalized_cols, original_cols))  # normalized → original
    df.columns = normalized_cols
    return df, col_map


def map_column_names(code: str, df: pd.DataFrame) -> str:
    """
    Replace model-generated column names in the code with actual column names from the dataset.
    Handles both exact and fuzzy matches.
    """
    # Create mapping: normalized → actual column names
    col_map = {col.strip().lower().replace(" ", "_"): col for col in df.columns}

    # Extract column names used in the code like df['colname']
    used_cols = re.findall(r"df\[['\"](.*?)['\"]\]", code)

    for used_col in used_cols:
        norm_col = used_col.strip().lower().replace(" ", "_")

        # 1. Exact match
        if norm_col in col_map and used_col != col_map[norm_col]:
            correct_col = col_map[norm_col]
            code = re.sub(rf"(['\"])({re.escape(used_col)})\1", f"'{correct_col}'", code)

        # 2. Fuzzy fallback
        elif norm_col not in col_map:
            close_matches = difflib.get_close_matches(norm_col, col_map.keys(), n=1, cutoff=0.6)
            if close_matches:
                correct_col = col_map[close_matches[0]]
                code = re.sub(rf"(['\"])({re.escape(used_col)})\1", f"'{correct_col}'", code)

    return code


def run_generated_code(code: str, df: pd.DataFrame):
    """Run the generated code with the actual DataFrame, after fixing column names."""
    code = map_column_names(code, df)

    for col in df.columns:
        if 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass

    local_vars = {'df': df.copy(), "pd": pd}
    stdout = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {}, local_vars)
        output = stdout.getvalue().strip()
        if 'result' in local_vars:
            return local_vars['result']
        return output if output else "✅ Code executed successfully."
    except Exception as e:
        return f"❌ Error during code execution: {e}"
    
def respect_rate_limit(seconds=105):
    for remaining in range(seconds, 0, -5):
        print(f"⏳ Waiting {remaining} seconds to respect LLaMA rate limits...", end='\r')
        time.sleep(5)
    print("\n✅ Ready for next query.")
    

def main():
    print("📂 Uploading and Reading the File...")
    file_path = input("📂 Enter path to your file (CSV/XLSX/TXT/DOCX/PDF/Image): ")

    df_or_text, col_map = load_file(file_path)

    is_tabular = isinstance(df_or_text, pd.DataFrame)
    is_textual = isinstance(df_or_text, str)

    if is_tabular:
        df = df_or_text

        if df.empty:
            print("❌ The dataset appears to be empty.")
            return

        print("✅ Tabular dataset loaded successfully.")
        #print("📊 Preview:\n", df.head())
        #print("🧾 Summary:\n", df.describe(include='all'))

        df_summary = df.describe(include='all').to_string()
        df_columns = df.columns.tolist()

    elif is_textual:
        loaded_text = df_or_text
        if not loaded_text.strip():
            print("❌ No textual content extracted from the file.")
            return

        print("📝 Text file loaded successfully. You can now ask questions about the content.")
    else:
        print("❌ Unsupported file or failed to extract content.")
        return

    plot_keywords = [
        "plot", "graph", "chart", "visualize", "visualisation", "visualization",
        "line", "line chart", "line plot",
        "bar", "bar chart", "bar graph",
        "histogram", "hist", "distribution",
        "scatter", "scatter plot",
        "pie", "pie chart",
        "box", "boxplot", "box plot",
        "violin", "violin plot",
        "area", "area chart", "area plot",
        "heatmap", "heat map",
        "density", "density plot", "kde",
        "pairplot", "pair plot",
        "countplot", "count plot",
        "bubble", "bubble chart",
        "donut", "donut chart",
        "correlation", "correlation matrix",
        "timeseries", "time series", "trend", "timeline",
        "facet", "facet plot", "multiplot", "multiple plots", "subplots",
        "map", "geoplot", "choropleth", "treemap", "sunburst",
    ]

    while True:
        user_question = input("\n💬 Your Question (or type 'exit'): ").strip()
        if user_question.lower() in ['exit', 'quit']:
            print("👋 Exiting the assistant. Goodbye!")
            break

        is_plot_request = any(re.search(rf'\b{kw}\b', user_question.lower()) for kw in plot_keywords)

        try:
            if is_tabular and is_plot_request:
                plot_response = handle_plot_query(user_question, df)
                '''if plot_response:
                    print(f"📊 {plot_response}")
                else:
                    print("❌ Plotting failed or not recognized.")'''
                if isinstance(plot_response, BytesIO):
                    img = Image.open(plot_response)
                    img.show()  # For CLI
                else:
                    print(plot_response)

            elif is_tabular:
                sample_rows = df.head(5).to_string()
                formatted_prompt = format_prompt(user_question, df_summary, df_columns, sample_rows)


                retries = 3
                llama_response = ""
                while retries > 0 and not llama_response.strip():
                    print("✅ Querying LLaMA on tabular data...")
                    llama_response = llama_query(formatted_prompt, df, is_code=True)
                    retries -= 1
                    if not llama_response.strip():
                        print("⚠️ Empty response from LLaMA. Retrying...")


                code_block = extract_code_block(llama_response)
                if not code_block:
                    print("⚠️ No valid Python code found. Try simplifying your question.")
                    continue

                #print("🤖 Generated code:\n", code_block)
                result = run_generated_code(code_block, df)
                print("📊 Answer:\n", result)

            elif is_textual:
                answer = ask_about_file(user_question)
                print("🧠 Answer:\n", answer)

            else:
                print("⚠️ No valid content loaded.")

        except Exception as e:
            print(f"❌ An unexpected error occurred:\n{e}")

        respect_rate_limit()


if __name__ == "__main__":
    main()