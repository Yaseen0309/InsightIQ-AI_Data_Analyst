# modules/__init__.py
# === File Handling ===
from .file_loader import load_file

# === LLM Interaction ===
from .llm_interface import (
    llama_query,
    ask_about_file,
    format_prompt,
    extract_code_block
)

# === Plot Utilities ===
from .plot_utils import (
    handle_plot_query,
    extract_plot_info_from_llama
)

# === Data Utilities ===
from .data_utils import (
    normalize_columns,
    map_column_names,
    describe_dataframe,
    extract_columns_from_prompt
)

# === Code Execution ===
from .executor import (
    run_generated_code,
    respect_rate_limit
)
