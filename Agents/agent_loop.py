from ollama import chat, ChatResponse
import json
import sqlite3

"""
Please download the netflix sqlite db from here:
https://github.com/lerocha/netflixdb/releases

This script shows how to use ollama's qwen3:8b model in an agentic loop with tool calls.
The agent is given a function to execute queries against an sqlite db and another function to inform itself about its schema
"""


# Connect to the Netflix sample database
conn = sqlite3.connect("netflixdb.sqlite")
cursor = conn.cursor()

# Define the tool execution function that Ollama will call
def run_sqlite_query(query: str):
    """
    Executes a SQL query against the connected netflixdb.sqlite database and returns the rows.
    You have direct access to the database through this tool. Pass valid SQLite syntax here.
    """
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_database_schema():
    """Returns the names and table creation schemas of all tables in the database."""
    conn = sqlite3.connect("netflixdb.sqlite")
    cursor = conn.cursor()
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    schema = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return json.dumps(schema)

system_prompt = """
You are a helpful data assistant with SQLite expertise. 
Use the available tools to inspect the database schema first, then run SQL queries to answer user requests accurately.
"""


available_functions = {
    "get_database_schema": get_database_schema,
    "run_sqlite_query": run_sqlite_query
}


messages = [
    {'role': 'system', 'content': system_prompt}
]

print("Chat session started with Qwen3:8b. Type 'exit' or 'quit' to stop.\n")

while True:
    # 1. Get input from the user dynamically
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Ending chat session.")
        break
        
    # Append the user's message to the conversation history
    messages.append({'role': 'user', 'content': user_input})

    # 2. Inner agentic loop to handle tool calls and multi-step reasoning
    while True:
        response: ChatResponse = chat(
            model='qwen3:8b',
            messages=messages,
            tools=[get_database_schema, run_sqlite_query],
            think=True,
        )
        
        # Append the model's response (including thought/content/tool calls) to history
        messages.append(response.message)
        
        if response.message.thinking:
            print("Thinking: ", response.message.thinking)
        if response.message.content:
            print("Assistant: ", response.message.content)
            
        if response.message.tool_calls:
            for tc in response.message.tool_calls:
                if tc.function.name in available_functions:
                    print(f"Calling {tc.function.name} with arguments {tc.function.arguments}")
                    result = available_functions[tc.function.name](**tc.function.arguments)
                    print(f"Result: {result}")
                    
                    # Add the tool execution result back into the message history
                    messages.append({'role': 'tool', 'tool_name': tc.function.name, 'content': str(result)})
        else:
            # Exit the inner loop once the model finishes calling tools and gives a final text answer
            break