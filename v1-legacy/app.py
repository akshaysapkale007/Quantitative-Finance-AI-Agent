"""
Finance AI Chatbot - Version 1.0 (Legacy)
Original implementation from early 2025

NOTE: This code no longer runs due to breaking changes in LangChain and OpenBB APIs.
It is preserved here to demonstrate the evolution of the project.

Key differences from v2:
- Used deprecated langchain.chat_models import path
- Required Tavily API for web search (paid)
- Required OpenBB API key (now uses free yfinance provider)
- Used create_tool_calling_agent (now uses ReAct pattern)
- Used Gemini 1.5 Flash (now uses Gemini 3 Flash Preview)
"""

import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import logging

# Langchain imports
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import TavilySearchTool

# OpenBB imports
from openbb import obb

# Logging setup
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class FinanceChatbot:
    def __init__(self):
        # Fetch API keys
        google_api_key = os.getenv('GOOGLE_API_KEY')
        tavily_api_key = os.getenv('TAVILY_API_KEY')
        openbb_api_key = os.getenv('OPENBB_API_KEY')

        # Validate API keys
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set in the .env file.")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY is not set in the .env file.")
        if not openbb_api_key:
            raise ValueError("OPENBB_API_KEY is not set in the .env file.")

        # Initialize Gemini 1.5 Flash model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=google_api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        # Setup OpenBB
        try:
            obb.account.login(api_key=openbb_api_key)
        except Exception as e:
            logger.error(f"OpenBB login failed: {e}")
        
        # Create tools
        self.tools = self.setup_tools(tavily_api_key)
        
        # Create agent
        self.agent = self.create_finance_agent()
    
    def setup_tools(self, tavily_api_key: str):
        """
        Setup tools for the finance chatbot
        """
        tools = [
            # Tavily Search Tool
            TavilySearchTool(api_key=tavily_api_key),
            
            # OpenBB Stock Price Tool
            Tool(
                name="get_stock_price",
                func=self.get_stock_price,
                description="Retrieve current stock price for a given ticker symbol"
            ),
            
            # OpenBB Company Overview Tool
            Tool(
                name="get_company_overview",
                func=self.get_company_overview,
                description="Get detailed company overview and financial information"
            ),
            
            # OpenBB Financial Statements Tool
            Tool(
                name="get_financial_statements",
                func=self.get_financial_statements,
                description="Retrieve financial statements for a company"
            ),
            
            # Market News Tool
            Tool(
                name="get_market_news",
                func=self.get_market_news,
                description="Fetch recent market news and headlines"
            )
        ]
        return tools
    
    def create_finance_agent(self):
        """
        Create a tool-calling agent for financial queries
        """
        # Custom prompt template for financial context
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI financial assistant. "
             "Use the available tools to provide accurate and detailed financial information. "
             "Always cite your sources and be transparent about the data you're using. "
             "If a tool cannot provide the exact information, use web search or explain limitations. "
             "Provide concise, actionable insights tailored to the user's query."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Create tool-calling agent
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True,
            max_iterations=5
        )
        
        return agent_executor
    
    def get_stock_price(self, ticker: str):
        """
        Retrieve current stock price
        """
        try:
            stock_data = obb.equity.price(ticker)
            return f"Current stock price for {ticker}: ${stock_data.price}. " \
                   f"Change: {stock_data.change} ({stock_data.change_percent}%)"
        except Exception as e:
            logger.error(f"Stock price error for {ticker}: {e}")
            return f"Could not retrieve stock price for {ticker}. {str(e)}"
    
    def get_company_overview(self, ticker: str):
        """
        Get company overview
        """
        try:
            overview = obb.equity.profile(ticker)
            return f"Company: {overview.name}\n" \
                   f"Sector: {overview.sector}\n" \
                   f"Industry: {overview.industry}\n" \
                   f"Market Cap: {overview.market_cap}\n" \
                   f"Description: {overview.description}"
        except Exception as e:
            logger.error(f"Company overview error for {ticker}: {e}")
            return f"Could not retrieve company overview for {ticker}. {str(e)}"
    
    def get_financial_statements(self, ticker: str, statement_type: str = "income"):
        """
        Retrieve financial statements
        """
        try:
            if statement_type == "income":
                financials = obb.equity.financials.income(ticker)
            elif statement_type == "balance":
                financials = obb.equity.financials.balance(ticker)
            elif statement_type == "cashflow":
                financials = obb.equity.financials.cashflow(ticker)
            else:
                return "Invalid statement type. Choose 'income', 'balance', or 'cashflow'."
            
            return f"Financial Statements ({statement_type}) for {ticker}:\n{financials}"
        except Exception as e:
            logger.error(f"Financial statement error for {ticker}: {e}")
            return f"Could not retrieve financial statements for {ticker}. {str(e)}"
    
    def get_market_news(self, query: str = None):
        """
        Fetch recent market news
        
        :param query: Optional search query to filter news
        :return: Recent market news as a string
        """
        try:
            # If no specific query, fetch general market news
            if not query:
                news = obb.news.headlines(provider="google_finance")
            else:
                # Search for news related to a specific query
                news = obb.news.search(query=query, provider="google_finance")
            
            # Format news into a readable string
            news_summary = "Recent Market News:\n"
            for article in news[:5]:  # Limit to 5 most recent articles
                news_summary += f"- {article.title}\n  Source: {article.source}\n  Link: {article.link}\n\n"
            
            return news_summary
        except Exception as e:
            logger.error(f"Market news error: {e}")
            return f"Could not retrieve market news. {str(e)}"
    
    def process_query(self, query: str):
        """
        Process user query using the finance agent
        
        :param query: User's input query
        :return: Agent's response
        """
        try:
            response = self.agent.invoke({"input": query})
            return response['output']
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return f"An error occurred while processing your query: {str(e)}"

# Flask App Setup
app = Flask(__name__)
finance_chatbot = FinanceChatbot()

@app.route('/')
def index():
    """
    Render the main chatbot interface
    """
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat interactions
    """
    try:
        # Get user query from request
        data = request.get_json()
        query = data.get('query', '')
        
        # Validate query
        if not query:
            return jsonify({
                'status': 'error',
                'response': 'Please provide a valid query.'
            }), 400
        
        # Process query
        response = finance_chatbot.process_query(query)
        
        # Return response
        return jsonify({
            'status': 'success',
            'response': response
        })
    except Exception as e:
        logger.error(f"Chat route error: {e}")
        return jsonify({
            'status': 'error',
            'response': 'An error occurred while processing your request.'
        }), 500

# Advanced Features: Context Management
class ConversationContext:
    """
    Manage conversation context and maintain state
    """
    def __init__(self, max_context_length=5):
        self.context = []
        self.max_context_length = max_context_length
    
    def add_interaction(self, user_query, bot_response):
        """
        Add user-bot interaction to context
        """
        interaction = {
            'user_query': user_query,
            'bot_response': bot_response
        }
        
        # Maintain max context length
        if len(self.context) >= self.max_context_length:
            self.context.pop(0)
        
        self.context.append(interaction)
    
    def get_context(self):
        """
        Retrieve current conversation context
        """
        return self.context
    
    def clear_context(self):
        """
        Clear conversation context
        """
        self.context = []

# Add context management to routes
conversation_context = ConversationContext()

@app.route('/context', methods=['GET'])
def get_context():
    """
    Retrieve conversation context
    """
    context = conversation_context.get_context()
    return jsonify({
        'status': 'success',
        'context': context
    })

@app.route('/clear-context', methods=['POST'])
def clear_context():
    """
    Clear conversation context
    """
    conversation_context.clear_context()
    return jsonify({
        'status': 'success',
        'message': 'Context cleared successfully.'
    })

# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    """
    Handle 404 errors
    """
    return jsonify({
        'status': 'error',
        'message': 'Resource not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """
    Handle 500 errors
    """
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Ensure proper logging and error handling during startup
    try:
        # Configure additional logging if needed
        file_handler = logging.FileHandler('finance_chatbot.log')
        file_handler.setLevel(logging.WARNING)
        app.logger.addHandler(file_handler)
        
        # Run the application
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.critical(f"Application startup failed: {e}")
        raise
