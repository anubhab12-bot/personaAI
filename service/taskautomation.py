import os
from dotenv import load_dotenv
from langchain_google_community.search import GoogleSearchAPIWrapper
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# from service.settings import GROQ_MODEL

load_dotenv()

SERPAPI_API_KEY = "cf06593be79873c12d15ed580eae58171a68f5707515ede3692c376fc7f9ee83"
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GROQ_MODEL = os.environ.get('GROQ_MODEL')

def search_google(query):
    """Fetches top Google search results."""
    search = GoogleSearchAPIWrapper(
        google_api_key="AIzaSyDdOxEiY24S6_BPJ_rkztAImXRRtdkGBoE",
        google_cse_id="f5fd0a57f552f4c59"
    )
    results = search.run(query)
    return results

def process_with_groq(query, search_results):
    """Processes search results using Groq LLM."""
    llm = ChatGroq(api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL)

    prompt = PromptTemplate(
        input_variables=["search_results", "query"],
        template="You are an intelligent AI. Based on the following search results:\n\n{search_results}\n\nAnswer this query concisely: {query}"
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"search_results": search_results, "query": query})

    return response

# def get_answer(query):
#     """Main function to search and generate an answer."""
#     search_results = search_google(query)
#     answer = process_with_groq(query, search_results)
#     return answer

# if __name__ == "__main__":
#     user_query = input("Enter your question: ")
#     result = get_answer(user_query)
#     print("\nAI Answer:", result)
    