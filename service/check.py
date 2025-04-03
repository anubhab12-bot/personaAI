from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool
import os
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Initialize the model and tokenizer for summarization
summarizer = pipeline(
    "summarization",
    model="facebook/bart-base",  # Using smaller base model instead of large
    device=-1,  # Use CPU
    model_kwargs={"low_cpu_mem_usage": True}  # Optimize memory usage
)


# Your API credentials
GOOGLE_CSE_ID = "f5fd0a57f552f4c59"
GOOGLE_API_KEY = "AIzaSyDdOxEiY24S6_BPJ_rkztAImXRRtdkGBoE"

class EnhancedGoogleSearch:
    def __init__(self):
        self.search = GoogleSearchAPIWrapper(
            google_api_key=GOOGLE_API_KEY,
            google_cse_id=GOOGLE_CSE_ID,
            k=3  # Number of results to fetch
        )

    def format_results(self, query: str) -> str:
        try:
            # Get search results
            raw_results = self.search.results(query, num_results=2)
            
            if not raw_results:
                return f"No information found about '{query}'"

            # Combine content from results
            combined_content = " ".join(
                [f"{result.get('title', '')}. {result.get('snippet', '')}" 
                 for result in raw_results]
            )

            # Generate summary
            summary = summarizer(
                combined_content,
                max_length=150,
                min_length=50,
                do_sample=False
            )[0]['summary_text']

            # Format the response
            response = f"""
Search Results for: {query}
-------------------------
Summary:
{summary}

Detailed Information:
"""
            # Add individual results with source attribution
            for i, result in enumerate(raw_results, 1):
                title = result.get('title', 'No title')
                snippet = result.get('snippet', 'No content available')
                link = result.get('link', '')
                
                response += f"\n{i}. {title}"
                response += f"\n   {snippet}"
                response += f"\n   Source: {link}\n"

            return response

        except Exception as e:
            return f"Error performing search: {str(e)}"

def search_and_summarize(query: str) -> str:
    """Main function to perform search and return formatted results"""
    search_tool = EnhancedGoogleSearch()
    return search_tool.format_results(query)

# Example usage
if __name__ == "__main__":
    # Test queries
    queries = ["Lamine Yamal"]
    
    
    for query in queries:
        print("\n" + "="*50)
        print(f"Searching for: {query}")
        print("="*50)
        result = search_and_summarize(query)
        print(result)