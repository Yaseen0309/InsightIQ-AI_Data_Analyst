import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO

from modules.llm_interface import llama_query


def extract_plot_info_from_llama(query, df_columns):
    chart_types = [
        "line", "bar", "stacked_bar", "grouped_bar", "pie", "scatter", "histogram", "box", "violin",
        "heatmap", "area", "count", "pairplot"
    ]

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

        # Handle "none" cases
        if chart_type == "none":
            chart_type = fallback_chart
        if x_col == "none":
            x_col = None
        if y_col == "none":
            y_col = None
        if agg_func == "none":
            agg_func = None

        return chart_type, x_col, y_col, agg_func

    except Exception as e:
        print(f"⚠️ Failed to parse chart info: {e}")
        print("🧠 Raw LLaMA response:\n", response)
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
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close('all') 
        return buf

    except Exception as e:
        return f"❌ Plotting failed: {e}"