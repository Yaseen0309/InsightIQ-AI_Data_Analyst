import pandas as pd
import difflib
import io
import re


def normalize_columns(df: pd.DataFrame):
    """
    Normalize column names and return original-to-normalized mapping.
    """
    original_cols = df.columns
    normalized_cols = [col.strip().lower().replace(" ", "_") for col in original_cols]
    col_map = dict(zip(normalized_cols, original_cols))  # normalized → original
    df.columns = normalized_cols
    return df, col_map


def map_column_names(code: str, df: pd.DataFrame, debug=False) -> str:
    """
    Replace LLM-generated column names in code with actual DataFrame column names.
    Handles normalization and fuzzy matching.
    """
    import difflib
    import re

    col_map = {col.strip().lower().replace(" ", "_"): col for col in df.columns}
    used_cols = re.findall(r"df\[['\"](.*?)['\"]\]", code)

    replaced = set()

    for used_col in used_cols:
        norm_col = used_col.strip().lower().replace(" ", "_")

        if norm_col in col_map and used_col != col_map[norm_col]:
            correct_col = col_map[norm_col]
            if used_col not in replaced:
                code = re.sub(rf"(['\"]){re.escape(used_col)}(['\"])", f"'{correct_col}'", code)
                replaced.add(used_col)
                if debug:
                    print(f"🔁 Exact match replaced: {used_col} → {correct_col}")

        elif norm_col not in col_map:
            close_matches = difflib.get_close_matches(norm_col, col_map.keys(), n=1, cutoff=0.6)
            if close_matches:
                correct_col = col_map[close_matches[0]]
                if used_col not in replaced:
                    code = re.sub(rf"(['\"]){re.escape(used_col)}(['\"])", f"'{correct_col}'", code)
                    replaced.add(used_col)
                    if debug:
                        print(f"🔁 Fuzzy match replaced: {used_col} → {correct_col}")

    return code


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
    buffer.write("\n\n❓ Missing Values:\n")
    buffer.write(str(df.isnull().sum()))

    
    return buffer.getvalue()

def extract_columns_from_prompt(prompt: str, df: pd.DataFrame):
    prompt_words = set(prompt.lower().split())
    matched_cols = [
        col for col in df.columns
        if any(word in prompt_words for word in col.lower().split('_'))
    ]
    return list(set(matched_cols))
