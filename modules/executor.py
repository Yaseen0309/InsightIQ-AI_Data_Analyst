import io
import re
import time
import contextlib
import pandas as pd
import builtins
import streamlit as st

from modules.data_utils import map_column_names


def run_generated_code(code: str, df: pd.DataFrame, debug: bool = False):
    """Run the generated code and return result as object (DataFrame, Series, str, etc.)."""

    import builtins
    import re
    import contextlib

    code = map_column_names(code, df)

    # Convert date columns
    for col in df.columns:
        if 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass

    local_vars = {'df': df.copy(), 'pd': pd, '__builtins__': builtins}
    stdout = io.StringIO()

    # Replace print() with result =
    code = re.sub(r"print\((.*)\)", r"result = \1", code)

    # Wrap last line
    lines = code.strip().split('\n')
    last_line = lines[-1].strip()
    has_result = any(line.strip().startswith("result") for line in lines)

    if not has_result and last_line and not last_line.startswith(("result", "#")):
        lines[-1] = f"result = {last_line}"
        code = '\n'.join(lines)

    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {}, local_vars)

        result = local_vars.get('result', None)
        output = stdout.getvalue().strip()

        # Debug logging
        if debug:
            print("🔍 Final executed code:\n", code)
            print("🧪 Stdout:", output)
            print("📦 Result:", type(result), result)

        # Smart return order
        if isinstance(result, (pd.DataFrame, pd.Series, list, dict)):
            return result
        elif isinstance(result, (int, float, str)):
            return str(result)
        elif output:
            return output
        else:
            return "✅ Code executed successfully."

    except Exception as e:
        error_trace = f"❌ Error during code execution: {e}"
        if debug:
            print(error_trace)
            print("💥 Code that caused it:\n", code)
        return error_trace


    
def respect_rate_limit(seconds=105, ui_mode="cli", dev_mode=False):
    """Wait for LLaMA rate limit with CLI or Streamlit animation."""

    if dev_mode:
        return  # Skip wait during testing

    if ui_mode == "streamlit":
        import streamlit as st
        import random

        # Emoji animation frames
        frames = ["⏳", "⌛"]
        placeholder = st.empty()

        with placeholder.container():
            for remaining in range(seconds, 0, -1):
                mins, secs = divmod(remaining, 60)
                emoji = frames[remaining % len(frames)]  # rotate emojis

                timer_text = f"{emoji} Respecting LLaMA Rate Limit: {mins:02d}:{secs:02d} remaining"

                placeholder.markdown(
                    f"""
                    <style>
                        .llama-timer {{
                            text-align: center;
                            font-size: 1.2rem;
                            font-weight: bold;
                            background-color: #f4f6f9;
                            padding: 1rem;
                            border-radius: 12px;
                            box-shadow: 2px 2px 10px #ddd;
                            color: #003366;
                            animation: pulse 2s infinite;
                        }}
                        @keyframes pulse {{
                            0% {{ transform: scale(1); opacity: 1; }}
                            50% {{ transform: scale(1.03); opacity: 0.85; }}
                            100% {{ transform: scale(1); opacity: 1; }}
                        }}
                        @media (prefers-color-scheme: dark) {{
                            .llama-timer {{
                                background-color: #5b5c5c !important;
                                color: #f0f0f0 !important;
                                box-shadow: 2px 2px 10px #000;
                            }}
                        }}
                    </style>
                    <div class='llama-timer'>
                        {timer_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                time.sleep(1)
            placeholder.empty()
        st.success("✅ Ready for next query!")

    else:
        # CLI fallback with 5-sec steps
        for remaining in range(seconds, 0, -5):
            print(f"⏳ Waiting {remaining} seconds to respect LLaMA rate limits...", end='\r')
            time.sleep(5)
        print("\n✅ Ready for next query.")

