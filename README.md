# 📊 InsightIQ — Your AI Data Analyst Copilot

**InsightIQ** is your intelligent AI copilot for data analysis. Upload files, ask questions in natural language, and get charts, tables, and insights in return — powered by LLaMA 4 from Together API.

🚀 Works in **CLI** & **Streamlit Web App** modes.

---

## 🧠 Features

- 🔍 Natural language Q&A over CSV, Excel, PDF, TXT, DOCX, and image files
- 📊 Auto-detects and renders relevant plots: line, bar, scatter, heatmaps, etc.
- 🤖 Multi-modal I/O: works in both CLI & Streamlit UI
- 📦 Modular backend with reusable logic
- 📚 Extracts text from images and PDFs using OCR
- 🧵 Maintains chat history in UI (and shows tables/plots in stream)
- 🎨 Aesthetic light/dark responsive UI

---

## 🖥️ Modes of Use

### 1️⃣ Web App (Streamlit)

```bash
streamlit run app.py
```

> Upload your data file and interact with your AI analyst in chat form. See plots, ask questions, and explore visually.

---

### 2️⃣ CLI Mode (Terminal)

```bash
python script.py
```

> The terminal-based version of InsightIQ — best for devs and quick analyses.

---

## 🗃️ Supported File Types

- `.csv`, `.xlsx`, `.txt`, `.pdf`, `.docx`
- `.jpg`, `.jpeg`, `.png` (text extraction using OCR)

---

## 📁 Project Structure

```
InsightIQ/
├── app.py                 # Streamlit UI
├── script.py              # CLI version
├── modules/               # Modular logic
│   ├── file_loader.py     # File reading + extraction
│   ├── plot_utils.py      # Visualization handlers
│   ├── data_utils.py      # Column matching, normalization
│   ├── llm_interface.py   # LLaMA query + formatting
│   └── executor.py        # Code generation + execution
├── assets/                # Logo, icons, future reports
├── temp/                  # Uploaded files
├── .env                   # Environment config
├── requirements.txt
└── README.md
```

---

## 🧪 Example Queries to Try

- `Show top 5 orders with highest profit`
- `Plot sales and profit by month for each region`
- `What is the correlation between discount and profit?`
- `List orders with discount > 0.2 and sales > 1000`
- `Show the top 3 sub-categories in each region`

---

## 🔧 Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your Together API key:

Create a `.env` file:

```
TOGETHER_API_KEY=your_key_here
```

---

## 🔐 Dev Mode

Enable in Streamlit sidebar to skip long rate limits for testing.

---

## 🌍 Technologies

| Tool        | Role                        |
|-------------|-----------------------------|
| Python 3.11 | Core backend                |
| Streamlit   | UI for chat-based analyst   |
| Together API | LLaMA 4 for analysis/coding |
| Pandas      | Data analysis               |
| Matplotlib / Seaborn | Plots and visuals |
| pdfplumber, pytesseract | File extraction |

---

## 📄 License

MIT License © 2025 — Built by [@Yaseen](https://github.com/Yaseen0309)

---

