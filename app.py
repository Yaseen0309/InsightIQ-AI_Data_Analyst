import streamlit as st
import os, io, time
import pandas as pd
from PIL import Image
from script import (
    load_file, run_generated_code,
    format_prompt, llama_query, extract_code_block,
    handle_plot_query, respect_rate_limit
)

def initialize_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "df" not in st.session_state:
        st.session_state.df = None
        st.session_state.col_map = None
        st.session_state.sample_rows = None
    if "preview_shown" not in st.session_state:
        st.session_state.preview_shown = False
    if "dev_mode" not in st.session_state:
        st.session_state.dev_mode = False

# App config and setup
st.set_page_config(page_title="InsightIQ — AI Data Analyst", layout="wide")
st.title("📊 InsightIQ — Your AI Data Analyst")
initialize_session()

# Sidebar upload
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/InsightIQ_logo_icon.png/240px-InsightIQ_logo_icon.png", width=120)
    st.header("📂 Upload File")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "pdf", "txt", "docx", "jpg", "jpeg", "png"])
    st.checkbox("🚧 Skip LLaMA Rate Limit (Dev Mode)", key="dev_mode")

# File upload logic
if uploaded_file:
    file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    df, col_map = load_file(file_path)

    if isinstance(df, pd.DataFrame):
        st.session_state.df = df
        st.session_state.col_map = col_map
        st.session_state.sample_rows = df.head().to_string()
        st.success("✅ Tabular data loaded successfully.")

        if not st.session_state.preview_shown:
            st.markdown("Here are the first 5 rows of your data:")
            st.dataframe(df.head())
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": "Here are the first 5 rows of your data:"})
            st.session_state.messages.append({"role": "assistant", "type": "dataframe", "content": df.head()})
            st.session_state.preview_shown = True
    else:
        st.warning("⚠️ This file is not tabular. Only text extraction is supported.")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "text":
            content = msg["content"]
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(to right, #667eea, #764ba2); color:white;
                                padding:10px 15px;
                                border-radius: 18px 0px 18px 18px;
                                margin: 6px 0; max-width: 65%;
                                word-wrap: break-word;
                                overflow-wrap: break-word;
                                text-align: left; box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                                margin-left:auto;'>
                        {content}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(
                    f"""
                    <div style='background: var(--secondary-background-color); color:var(--text-color);
                                padding:10px 15px;
                                border-radius: 0px 18px 18px 18px;
                                margin: 6px 0; max-width: 65%;
                                word-wrap: break-word;
                                overflow-wrap: break-word;
                                text-align: left; box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                                margin-right:auto;'>
                        {content}
                    </div>
                    """, unsafe_allow_html=True)

        elif msg.get("type") == "dataframe":
            st.dataframe(msg["content"])
        elif msg.get("type") == "image":
            st.image(Image.open(io.BytesIO(msg["content"])))

# Chat input
user_prompt = st.chat_input("Ask your data analyst...")

if user_prompt and st.session_state.df is not None:
    df = st.session_state.df
    col_map = st.session_state.col_map
    sample_rows = st.session_state.sample_rows

    with st.chat_message("user"):
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; align-items: flex-start; margin: 6px 0;">
                <div style='background: linear-gradient(to right, #667eea, #764ba2); color: white; padding: 10px 15px;
                            border-radius: 18px 0px 18px 18px; max-width: 65%;
                            text-align: left; box-shadow: 0 2px 6px rgba(0,0,0,0.2);'>
                    {user_prompt}
                </div>
                
            </div>
            """, unsafe_allow_html=True
        )


    st.session_state.messages.append({"role": "user", "type": "text", "content": user_prompt})

    with st.chat_message("assistant"):
        
        with st.spinner("🤖 Thinking..."):
            start_time = time.time()

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
            is_plot = any(k in user_prompt.lower() for k in plot_keywords)

            if is_plot:
                plot_result = handle_plot_query(user_prompt, df)
                if isinstance(plot_result, (bytes, io.BytesIO)):
                    image_bytes = plot_result.getvalue() if isinstance(plot_result, io.BytesIO) else plot_result
                    st.image(Image.open(io.BytesIO(image_bytes)), caption="📊 Chart")
                    st.session_state.messages.append({"role": "assistant", "type": "image", "content": image_bytes})
                else:
                    st.markdown(plot_result)
                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": plot_result})
            else:
                prompt = format_prompt(user_prompt, df.describe(include='all').to_string(), df.columns.tolist(), sample_rows)
                llama_response = llama_query(prompt, df)
                code_block = extract_code_block(llama_response)

                if code_block:
                    result = run_generated_code(code_block, df)
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                        st.session_state.messages.append({"role": "assistant", "type": "dataframe", "content": result})
                    elif isinstance(result, (bytes, io.BytesIO)):
                        image_bytes = result.getvalue() if isinstance(result, io.BytesIO) else result
                        st.image(Image.open(io.BytesIO(image_bytes)), caption="📊 Chart")
                        st.session_state.messages.append({"role": "assistant", "type": "image", "content": image_bytes})
                    else:
                        st.markdown(result)
                        st.session_state.messages.append({"role": "assistant", "type": "text", "content": result})
                else:
                    msg = "⚠️ No valid Python code generated."
                    st.markdown(msg)
                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": msg})

            elapsed = time.time() - start_time
            st.markdown(f"🕐 Response time: {elapsed:.2f} seconds")
        
        respect_rate_limit(ui_mode="streamlit", dev_mode=st.session_state.dev_mode)

# Styling
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
    }

    /* Align user messages to right */
    [data-testid="chat-message-user"] {
        flex-direction: row-reverse !important;
    }

    /* Align assistant messages to left */
    [data-testid="chat-message-assistant"] {
        flex-direction: row !important;
    }

    /* Chat bubble styling - USER */
    [data-testid="chat-message-user"] .stMarkdown {
        background: var(--primary-color);
        color: var(--text-color);
        padding: 10px 15px;
        border-radius: 18px 0px 18px 18px;
        max-width: 65%;
        margin-left: auto;
        margin-right: 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }

    /* Chat bubble styling - ASSISTANT */
    [data-testid="chat-message-assistant"] .stMarkdown {
        background: var(--secondary-background-color);
        color: var(--text-color);
        padding: 10px 15px;
        border-radius: 0px 18px 18px 18px;
        max-width: 65%;
        margin-right: auto;
        margin-left: 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }

    /* Optional background (uses Streamlit's default theme colors) */
    .stApp {
        background-color: var(--background-color);
    }

    /* Adjust image/avatar spacing */
    [data-testid="chat-message-user"] img,
    [data-testid="chat-message-assistant"] img {
        margin: 0 10px;
    }

    /* Optional: Remove max-width constraint on entire app */
    .main .block-container {
        max-width: 100%;
    }
</style>
""", unsafe_allow_html=True)

