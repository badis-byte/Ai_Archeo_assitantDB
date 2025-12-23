from dotenv import load_dotenv
import os
from typing import Optional

from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_groq import ChatGroq
from supabase import create_client, Client
from langchain_core.runnables import RunnableLambda
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import initialize_agent

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="SQL Chatbot API",
    description="A chatbot API that can answer questions about archaeological findings in Algeria using SQL queries",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    
class ChatResponse(BaseModel):
    response: str
    status: str = "success"

class ErrorResponse(BaseModel):
    error: str
    status: str = "error"

# Global variables for the chatbot components
llm = None
agent = None
chain = None

def initialize_chatbot():
    """Initialize the chatbot components"""
    global llm, agent, chain
    
    try:
        # Get environment variables
        supa = os.getenv("Supabase_Project_Password")
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        ai = os.getenv("GROQ_API_KEY")
        
        if not all([supa, url, key, ai]):
            raise ValueError("Missing required environment variables")
        
        # Initialize Supabase client
        supabase: Client = create_client(url, key)
        
        # Initialize LLM
        llm = ChatGroq(model="gemma2-9b-it", api_key=ai)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
             You are an SQL agent that answers questions about archaeological findings in Algeria.
             Always check the schema of the database and all relations present in it using the tools you have.
             When you find a row or column that might contain the answer, check all foreign keys from or to it 
             and examine the data there to get more detailed and accurate answers.
             Never run a query that alters or modifies the database in any way - you are only a retriever.
             Be helpful and provide detailed responses about the archaeological findings not the database itself
             as you're talking to normal users.
            """),
            ("user", "{input}")
        ])
        
        # Initialize database
        db = SQLDatabase.from_uri(f"postgresql://postgres.hioxzjseqyevsstvntvu:{supa}@aws-0-us-east-2.pooler.supabase.com:6543/postgres")
        
        # Define tools
        @tool
        def show_relationships(dummy: str = ''):
            """Show foreign key relationships between tables."""
            query = """
            SELECT conrelid::regclass AS table_name, conname AS constraint_name,
                   pg_get_constraintdef(c.oid) AS definition
            FROM pg_constraint c
            WHERE contype = 'f';
            """
            return db.run(query)

        @tool
        def preview_table(table_name: str):
            """Preview the first few rows of a table."""
            try:
                return db.run(f"SELECT * FROM {table_name} LIMIT 5;")
            except Exception as e:
                return f"Error previewing table {table_name}: {str(e)}"

        @tool
        def get_data_dictionary(dummy: str = ''):
            """Return information about tables, their columns, primary keys, and foreign keys"""
            return """
            --List all tables:
| table_name |
| ---------- |
| sites      |
| details    |
| mobil      |
| immobil    |

            --List columns for each table:
| table_name | column_name    | data_type                   |
| ---------- | -------------- | --------------------------- |
| details    | id             | bigint                      |
| details    | created_at     | timestamp with time zone    |
| details    | info           | text                        |
| immobil    | id             | bigint                      |
| immobil    | found          | timestamp with time zone    |
| immobil    | surface        | double precision            |
| immobil    | more           | bigint                      |
| immobil    | name           | character varying           |
| immobil    | location       | character varying           |
| mobil      | id             | bigint                      |
| mobil      | found          | timestamp without time zone |
| mobil      | name           | character varying           |
| mobil      | weight         | double precision            |
| mobil      | place          | bigint                      |
| mobil      | more           | bigint                      |
| sites      | id             | bigint                      |
| sites      | created_at     | timestamp with time zone    |
| sites      | wilaya         | character varying           |
| sites      | number of lots | integer                     |

--List all foreign keys:
| table_name | column_name | foreign_table_name | foreign_column_name |
| ---------- | ----------- | ------------------ | ------------------- |
| mobil      | place       | sites              | id                  |
| immobil    | more        | details            | id                  |
| mobil      | more        | details            | id                  |
| immobil    | location    | sites              | wilaya              |
            """

        tools = [show_relationships, preview_table, get_data_dictionary]
        
        # Create SQL agent
        sql_agent = create_sql_agent(llm=llm, db=db, verbose=False)  # Set verbose=False for API
        
        # Create combined agent
        agent = initialize_agent(
            tools=tools + sql_agent.tools,
            llm=llm,
            agent_type="zero-shot-react-description",
            verbose=False,  # Set verbose=False for API
            handle_parsing_errors=True
        )
        
        # Create chain
        chain = prompt | agent
        
        return True
        
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot when the API starts"""
    success = initialize_chatbot()
    if not success:
        print("Failed to initialize chatbot")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "SQL Chatbot API",
        "description": "Ask questions about archaeological findings in Algeria",
        "endpoints": {
            "/chat": "POST - Send a message to the chatbot",
            "/health": "GET - Check API health status",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if chain is None:
        return {"status": "error", "message": "Chatbot not initialized"}
    return {"status": "healthy", "message": "API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        if chain is None:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Process the message
        response = chain.invoke({"input": request.message})
        
        # Extract the response text
        if isinstance(response, dict) and "output" in response:
            response_text = response["output"]
        else:
            response_text = str(response)
        
        return ChatResponse(response=response_text)
        
    except Exception as e:
        print(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing your message: {str(e)}")

@app.get("/examples")
async def get_examples():
    """Get example queries that users can try"""
    return {
        "examples": [
            "Can you give me more details about the immobil that was found in the wilaya of blida?",
            "What are all the tables in the database?",
            "Show me the foreign key relationships",
            "What mobile artifacts were found?",
            "List all archaeological sites",
            "What details are available for site ID 1?",
            "How many artifacts were found in each wilaya?",
            "What is the largest immobile artifact by surface area?"
        ]
    }

# For local development
if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0",
        port=8000,
        reload=True
    )