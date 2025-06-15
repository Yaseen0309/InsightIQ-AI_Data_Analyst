from io import BytesIO
import os
import re
import warnings
import pandas as pd
from PIL import Image
from modules.env_loader import load_environment
load_environment()
from modules import (
    load_file,
    llama_query,
    handle_plot_query,
    format_prompt,
    extract_code_block,
    run_generated_code,
    respect_rate_limit
)
from modules.llm_interface import ask_about_file


# Clean environment setup
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)



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
        print("📊 Preview:\n", df.head())
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