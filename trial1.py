from dotenv import load_dotenv
import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_groq import ChatGroq
from supabase import create_client, Client
from langchain_core.runnables import RunnableLambda
from fastapi import FastAPI
from pydantic import BaseModel



load_dotenv()

app = FastAPI()


url: str = os.getenv("SUPABASE_URL")
skey: str = os.getenv("SUPABASE_KEY")
key = os.getenv("GROQ_API_KEY")
supabase: Client = create_client(url, skey)


class QueryRequest(BaseModel):
    user_query: str

@app.post("/chat")
def chat_endpoint(req: QueryRequest):



    main_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a costumer service assistant for an algerian archeological website, you need to identify whether the user's query is a general question, or whether it does actually need for us to check the database, to know whether we need to go to the datanase know that the data in it contains all the sites discovered and all the info about what was discovered in it, answer with 'question is general' if no need to check the database, else 'question is specific' and after that reinclude the user's question starting with: 'the user question is:(and write it again exactly how it was)'"),
            ("user", "here is the user's question: {query}"),
        ]
    )

    model = ChatGroq(
        model="gemma2-9b-it",
        api_key=key
        )


    generalPrompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a helpful archeology assistant, respond to the user short and consise if the user shifts away from your domain knowledge away from archeology from algeria answer shotly and remind him of what you are able to answer about better, what you should also ignore is the first part that says:(question is general) in the biginning as it is not part of the users question it gets added automatically, so the user's input actually starts after that"),
            ("user", "here is the users input: {query}"),
        ]
    )

    ALLOWED_OPERATORS = {"eq", "gt", "lt", "gte", "lte", "neq", "like", "ilike"}
    ALLOWED_TABLES = {"sites", "mobil", "immobil", "details"}
    prompt = f'''you are a helpful text to sql interpreter, interperet the user's demand to a python query to supabase api. the difference between mobil and immobil tablesis that mobil are the stuff that can be carried, immobil are the big sites like a house or a city, both of these tables link to the forth table(details) which has further decription and details about the discovery in question
            JSON query
    for the Supabase client.

    Output only valid JSON with this structure:
    {{
    "table": "<table_name>",
    "select": ["*"],
    "filters": [
        {{"column": "<col>", "operator": "<op>", "value": <value>}}
    ],
    "order": {{"column": "<col>", "ascending": true}},
    "limit": <number>
    }}

    Rules:
    - Only use these tables: ("sites", "mobil", "immobil", "details")
    - Operators must be one of: ("eq", "gt", "lt", "gte", "lte", "neq", "like", "ilike")
    - If no filters, return an empty list for "filters".
    - If no ordering, return null for "order".
    - Always include "limit" (default 10 if not specified).'''

    specificPrompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{prompt}"),
            ("user", "{query}"),
        ]
    )

    dafaultPrompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a helpful assistant"),
            ("user", "{query}"),
        ]
    )

    supaprompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"""you are a translator from json form to supabase api in python content to query the data needed by the sql nprovided, Do NOt use the method .from() use .from_() or .table() insted of it just replace when needed, store it in a variable (response) then do as such: results = response.data.. and do not add any formats just these two lines of code nothing else, For each foreign key column, YOU MUSTreplace it with the full related table by expanding it in .select(), Example: .select('*, related_table(*)') following this relations
            [
    [    
        "source_table": "mobil",
        "source_column": "place",
        "target_table": "sites",
        "target_column": "id"
    ],
    [
        "source_table": "immobil",
        "source_column": "more",
        "target_table": "details",
        "target_column": "id"
    ],
    [
        "source_table": "mobil",
        "source_column": "more",
        "target_table": "details",
        "target_column": "id"
        ],
    [
        "source_table": "immobil",
        "source_column": "location",
        "target_table": "sites",
        "target_column": "id"
        ]
    ]
             AND START with just supabase. not supabase_client or such
            AND DO NOT USE .from() use : .table()
            AND DO NOT forget the .execute() at the end"""),
            ("user", "{query}"),
        ]
    )

    formatprompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a text format corrector, the given query is passed from an llm and your job is to make sure it is well coded to be run, what you should do is: remove any text formatting or comments and language specification, and say nothing and explain nothing, the ONLY thing needed from you to output is the two lines from code redirected directly to run so no additional talking no formatting just clean two lines code that are NOT commented so remove any ''' ''' you find -- make sure he is not using .filter() in the wrong, if it is replace it with the .eq() as such ex: .eq('name', 'tipaza')-- AND NOT COMMENTED remove that '''python''' shit, MaKE sure it is not using supabase.from() it is Dangerous.. make it supabase.table() instead"),
            ("user", "here is the previous llm's output do not forget to not use the ```python``` and just output pure code, literally what you should just output is two lines: line1(response = supabase.from()..(rest of code).execute()), line2(results=response.data) thats it no more no less bitch: {query}"),
        ]
    )

    stringtodict = RunnableLambda( lambda x: {"query": x, "prompt": prompt})

    branch = RunnableBranch(
        (
            lambda x: "general" in x,
            generalPrompt | model | StrOutputParser()
        ),
        (
            lambda x: "specific" in x,
            stringtodict | specificPrompt | model | StrOutputParser() | supaprompt | model | StrOutputParser() | formatprompt | model | StrOutputParser()
        ),
        dafaultPrompt | model | StrOutputParser()

    )



    #question = "what was found in tipaza as immobil"
    question = req.user_query

    main_chain = main_prompt | model | StrOutputParser() 


    the_chain = main_chain | branch

    result = the_chain.invoke({"query": question, "prompt": prompt})
    print("***************************************************************")
    print("***************************************************************")
    print(result)
    print("***************************************************************")

    local_vars = {"supabase": supabase}

    # --- 3. Run the AI's code ---
    exec(result, {}, local_vars)

    # --- 4. Retrieve results ---
    results = local_vars.get("results")
    #print("Query results:", results)
    return results









