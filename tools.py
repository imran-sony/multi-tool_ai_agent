# tools.py
import re
import sqlite3
from typing import Any, Dict, List, Optional
import os
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.llms import OpenAI as LLM
from langchain.tools import BaseTool
from serpapi import GoogleSearch
import json


SELECT_ONLY_RE = re.compile(r'^\s*select\b', re.I | re.S)

def is_select_only(sql: str) -> bool:

    forbidden = re.compile(r'\b(insert|update|delete|drop|alter|create|attach|detach|replace|pragma)\b', re.I)
    if not SELECT_ONLY_RE.match(sql):
        return False
    if forbidden.search(sql):
        return False

    if ';' in sql.strip().rstrip(';'):

        parts = [p.strip() for p in sql.split(';') if p.strip()]
        if len(parts) > 1:
            return False
    return True


class BaseDBTool(BaseTool):
    name = "BaseDBTool"
    description = "Base DB tool"

    def __init__(self, db_path: str, table_name: str, llm=None, max_rows=50):
        self.db_path = db_path
        self.table_name = table_name
        self.max_rows = max_rows
  
        self.llm = llm or ChatOpenAI(temperature=0)
        super().__init__()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def run_select(self, sql: str) -> List[tuple]:
        """Execute a SELECT statement and return rows (limit enforced)."""
        if not is_select_only(sql):
            raise ValueError("Only safe SELECT queries are allowed.")

        sql_clean = sql.strip().rstrip(';')
        if re.search(r'\blimit\b', sql_clean, re.I) is None:
            sql_clean = f"{sql_clean} LIMIT {self.max_rows}"
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(sql_clean)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            return cols, rows
        finally:
            conn.close()

    def format_results_nl(self, cols: List[str], rows: List[tuple], user_question: str) -> str:

        if not rows:
            return "No rows matched your query."


        header = " | ".join(cols)
        sample_lines = []
        for r in rows[:10]:
            sample_lines.append(" | ".join(str(x) for x in r))
        sample_text = "\n".join(sample_lines)
        total = len(rows)
        return (f"Query: {user_question}\n"
                f"Returned {total} row(s). Columns: {header}\n"
                f"Sample rows:\n{sample_text}")


class HeartDiseaseDBTool(BaseDBTool):
    name = "HeartDiseaseDBTool"
    description = ("Use this tool to answer numeric/statistical questions about the Heart Disease dataset. "
                   "Input should be a natural-language question about heart_disease table. Returns results from SQL.")
    def __init__(self, db_path='heart_disease.db', table_name='heart_disease', llm=None):
        super().__init__(db_path, table_name, llm=llm)

    def _run(self, query: str) -> str:
 
        prompt = (
            "You are a helpful assistant that MUST output a single SQL SELECT statement (no explanation) "
            "that answers the user's question against a SQLite table named "
            f"'{self.table_name}'. The table columns exist as in the DB. "
            "You must not produce any non-SELECT statement. If the question is not answerable with SQL, "
            "just respond: 'CANNOT_ANSWER_WITH_SQL'.\n\nUser question:\n"
            f"{query}\n\nSQL:"
        )
        llm_resp = self.llm.call_as_llm(HumanMessage(content=prompt)) if hasattr(self.llm, 'call_as_llm') else self.llm(prompt)
        sql = None
        if isinstance(llm_resp, str):
            sql = llm_resp.strip()
        else:

            sql = llm_resp.content if hasattr(llm_resp, 'content') else str(llm_resp)
            sql = sql.strip()
    
        if '```' in sql:
            sql = sql.split('```')[-2].strip()
    
        if sql.upper().startswith("CANNOT_ANSWER_WITH_SQL"):
            return "I couldn't convert that question into a SQL query. Try asking for statistics or a specific aggregation."
        if not is_select_only(sql):
            return ("The generated SQL is not safe (non-SELECT or multi-statement). "
                    "I will not execute it. Try rephrasing your question to require a SELECT aggregation or filter.")

        try:
            cols, rows = self.run_select(sql)
            return self.format_results_nl(cols, rows, query)
        except Exception as e:
            return f"Error executing query: {e}"

class CancerDBTool(BaseDBTool):
    name = "CancerDBTool"
    description = ("Use this tool to answer numeric/statistical questions about the Cancer dataset. "
                   "Input should be a natural-language question about cancer table. Returns results from SQL.")
    def __init__(self, db_path='cancer.db', table_name='cancer', llm=None):
        super().__init__(db_path, table_name, llm=llm)

    def _run(self, query: str) -> str:
        prompt = (
            "You are a helpful assistant that MUST output a single SQL SELECT statement (no explanation) "
            f"that answers the user's question against a SQLite table named '{self.table_name}'. "
            "If you cannot answer with SQL, return CANNOT_ANSWER_WITH_SQL.\n\nUser question:\n"
            f"{query}\n\nSQL:"
        )
        llm_resp = self.llm.call_as_llm(HumanMessage(content=prompt)) if hasattr(self.llm, 'call_as_llm') else self.llm(prompt)
        sql = llm_resp.strip() if isinstance(llm_resp, str) else (llm_resp.content if hasattr(llm_resp, 'content') else str(llm_resp))
        if '```' in sql:
            sql = sql.split('```')[-2].strip()
        if sql.upper().startswith("CANNOT_ANSWER_WITH_SQL"):
            return "I couldn't convert that question into a SQL query. Try asking for statistics or a specific aggregation."
        if not is_select_only(sql):
            return "Refusing to run unsafe SQL. Please ask for a SELECT query."
        try:
            cols, rows = self.run_select(sql)
            return self.format_results_nl(cols, rows, query)
        except Exception as e:
            return f"Error executing query: {e}"

class DiabetesDBTool(BaseDBTool):
    name = "DiabetesDBTool"
    description = ("Use this tool to answer numeric/statistical questions about the Diabetes dataset. "
                   "Input should be a natural-language question about diabetes table. Returns results from SQL.")
    def __init__(self, db_path='diabetes.db', table_name='diabetes', llm=None):
        super().__init__(db_path, table_name, llm=llm)

    def _run(self, query: str) -> str:
        prompt = (
            "You are a helpful assistant that MUST output a single SQL SELECT statement (no explanation) "
            f"that answers the user's question against a SQLite table named '{self.table_name}'. "
            "If you cannot answer with SQL, return CANNOT_ANSWER_WITH_SQL.\n\nUser question:\n"
            f"{query}\n\nSQL:"
        )
        llm_resp = self.llm.call_as_llm(HumanMessage(content=prompt)) if hasattr(self.llm, 'call_as_llm') else self.llm(prompt)
        sql = llm_resp.strip() if isinstance(llm_resp, str) else (llm_resp.content if hasattr(llm_resp, 'content') else str(llm_resp))
        if '```' in sql:
            sql = sql.split('```')[-2].strip()
        if sql.upper().startswith("CANNOT_ANSWER_WITH_SQL"):
            return "I couldn't convert that question into a SQL query. Try asking for statistics or a specific aggregation."
        if not is_select_only(sql):
            return "Refusing to run unsafe SQL. Please ask for a SELECT query."
        try:
            cols, rows = self.run_select(sql)
            return self.format_results_nl(cols, rows, query)
        except Exception as e:
            return f"Error executing query: {e}"


class MedicalWebSearchTool(BaseTool):
    name = "MedicalWebSearchTool"
    description = ("Use this tool for general medical knowledge: definitions, symptoms, cures. "
                   "It performs a web search and summarizes results.")
    def __init__(self, serpapi_key: Optional[str] = None, llm=None):
        self.serpapi_key = serpapi_key or os.getenv('SERPAPI_API_KEY')
        self.llm = llm or ChatOpenAI(temperature=0)
        super().__init__()

    def _search(self, query: str, num_results:int=5):
        if not self.serpapi_key:
            raise ValueError("SERPAPI_API_KEY is not set.")
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
            "num": num_results,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        items = []
        for r in results.get('organic_results', [])[:num_results]:
            items.append({
                "title": r.get('title'),
                "snippet": r.get('snippet'),
                "link": r.get('link'),
            })
        return items

    def _run(self, query: str) -> str:
        """Perform search and summarize with LLM"""
        try:
            items = self._search(query, num_results=5)
        except Exception as e:
            return f"Search failed: {e}"
        
        combined = "\n\n".join([f"{i+1}. {it['title']}\n{it['snippet']}\nLink: {it['link']}" for i,it in enumerate(items)])
        prompt = (
            "You are a medical-assistant summarizer (use general medical knowledge only). "
            "Given these search results, produce a brief, accurate answer to the user's question and include short references to the result titles. "
            "Be concise and avoid making novel medical claims — only summarize what's in the results. "
            "If results are inconclusive, say so and recommend consulting trusted medical sources or a professional.\n\n"
            f"Search results:\n{combined}\n\nUser question:\n{query}\n\nAnswer:"
        )
        llm_resp = self.llm.call_as_llm(HumanMessage(content=prompt)) if hasattr(self.llm, 'call_as_llm') else self.llm(prompt)
        text = llm_resp.content if hasattr(llm_resp, 'content') else str(llm_resp)
        return text.strip()
