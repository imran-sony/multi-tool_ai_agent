# Multi-Tool AI Agent

## Setup
1. `pip install -r requirements.txt`
2. Download Kaggle CSVs and put them in `data/` as `heart.csv`, `cancer.csv`, `diabetes.csv`
3. Create .env with API keys:
   OPENAI_API_KEY=...
   SERPAPI_API_KEY=...
4. Run:
   python convert_csvs.py --data-dir data
5. Start agent:
   python main_agent.py
