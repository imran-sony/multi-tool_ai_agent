# main_agent.py
import os
from tools import HeartDiseaseDBTool, CancerDBTool, DiabetesDBTool, MedicalWebSearchTool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain import OpenAI

def build_tools():

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")  
  
    heart_tool = HeartDiseaseDBTool(db_path='heart_disease.db', table_name='heart_disease', llm=llm)
    cancer_tool = CancerDBTool(db_path='cancer.db', table_name='cancer', llm=llm)
    diabetes_tool = DiabetesDBTool(db_path='diabetes.db', table_name='diabetes', llm=llm)
    web_tool = MedicalWebSearchTool(serpapi_key=os.getenv('SERPAPI_API_KEY'), llm=llm)

    tools = [
        Tool.from_function(func=heart_tool._run, name="HeartDiseaseDBTool", description=heart_tool.description),
        Tool.from_function(func=cancer_tool._run, name="CancerDBTool", description=cancer_tool.description),
        Tool.from_function(func=diabetes_tool._run, name="DiabetesDBTool", description=diabetes_tool.description),
        Tool.from_function(func=web_tool._run, name="MedicalWebSearchTool", description=web_tool.description),
    ]
    return tools, llm

def build_agent():
    tools, llm = build_tools()
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    return agent

def repl():
    agent = build_agent()
    print("Multi-Tool Medical Agent. Type 'exit' to quit.")
    while True:
        q = input("\nUser question: ").strip()
        if q.lower() in ("exit","quit"):
            break
        resp = agent.run(q)
        print("\nAgent response:\n", resp)

if __name__ == '__main__':
    import dotenv
    dotenv.load_dotenv()
    repl()
