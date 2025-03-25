# Welcome message template
WELCOME_TEMPLATE = """
Welcome! You can ask questions about:
1. Your personal information (examples):
   - What are my skills?
   - Tell me about myself
   - What are my interests?
   - Where do I work?
2. Or any other general questions!
"""
COMPANY_QUERY_TEMPLATE = """
You are an AI assistant with access to personal information about the user. 
Here is the available information about the user's company:
{personal_context}

The user is asking: {query}

Please provide a comprehensive response about their company, including relevant details from the context. 
Ensure to mention their role and company name in the response. 
Format the response in JSON as follows:
{
    "response": "Your AI-generated response about the company",
    "links": [
        {"title": "Company Website", "url": "https://example.com"}
    ]
}
ONLY return the JSON response. No additional text.
"""

IDENTITY_QUERY_TEMPLATE = """
        Here is the available information about the user:
        {personal_context}

        The user is asking: {query}

        Please provide a comprehensive response that:
        1. Introduces who they are (name, occupation)
        2. Describes their background and experience
        3. Mentions their skills and interests
        4. Includes any other relevant personal details

        Format the response in a natural, conversational way while maintaining the required JSON structure.
"""

# System message template
SYSTEM_TEMPLATE = """You are a smart AI assistant with access to personal information about the user. 
Use this information when relevant to provide personalized responses.
Always return responses in JSON format like this:
{
    "response": "Your AI-generated response",
    "links": [
        {"title": "Relevant site 1", "url": "https://example.com"},
        {"title": "Relevant site 2", "url": "https://example2.com"}
    ]
}
ONLY return JSON responses. No additional text."""

# Personal query template
PERSONAL_QUERY_TEMPLATE = """
Context about the user:
{personal_context}

Question about the user: {query}

Please provide a detailed response based on the user's personal information.
"""

# General query template
GENERAL_QUERY_TEMPLATE = """
Context about the user that might be relevant:
{personal_context}

User query: {query}

Please provide a response, incorporating personal context when relevant.
"""