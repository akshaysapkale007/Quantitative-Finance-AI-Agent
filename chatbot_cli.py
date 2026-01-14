import os
from dotenv import load_dotenv
import logging
from typing import Optional, List
from datetime import datetime, timezone, timedelta
import json

# LangChain and LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# OpenBB imports
from openbb import obb

# Logging setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# ===== TOOL DEFINITIONS =====
# Using @tool decorator for LangGraph compatibility

@tool
def get_stock_quote(ticker: str) -> str:
    """Get real-time stock price, volume, market cap, and daily changes for a ticker symbol.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
    """
    try:
        ticker = ticker.strip().upper()
        quote_data = obb.equity.price.quote(symbol=ticker, provider="yfinance")
        
        if quote_data and quote_data.results:
            data = quote_data.results[0]
            
            # Handle different field names between providers
            price = getattr(data, 'last_price', None) or getattr(data, 'price', None) or 0
            change = getattr(data, 'change', None) or 0
            change_pct = getattr(data, 'change_percent', None) or getattr(data, 'percent_change', None) or 0
            day_high = getattr(data, 'high', None) or getattr(data, 'day_high', None) or 0
            day_low = getattr(data, 'low', None) or getattr(data, 'day_low', None) or 0
            vol = getattr(data, 'volume', None) or 0
            mkt_cap = getattr(data, 'market_cap', None)
            
            direction = "📈 UP" if change and change > 0 else "📉 DOWN" if change and change < 0 else "➡️ FLAT"
            
            return json.dumps({
                "ticker": ticker,
                "price": round(float(price), 2) if price else None,
                "change": round(float(change), 2) if change else None,
                "change_percent": round(float(change_pct), 2) if change_pct else None,
                "direction": direction,
                "day_high": round(float(day_high), 2) if day_high else None,
                "day_low": round(float(day_low), 2) if day_low else None,
                "volume": vol,
                "market_cap": mkt_cap,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        else:
            return json.dumps({"error": f"No quote data available for {ticker}"})
            
    except Exception as e:
        logger.error(f"Stock quote error for {ticker}: {e}")
        return json.dumps({"error": f"Failed to retrieve quote: {str(e)}"})


@tool
def get_company_profile(ticker: str) -> str:
    """Get company information: name, sector, industry, description, CEO, employees.
    
    Args:
        ticker: Stock ticker symbol for the company
    """
    try:
        ticker = ticker.strip().upper()
        profile_data = obb.equity.profile(symbol=ticker, provider="yfinance")
        
        if profile_data and profile_data.results:
            data = profile_data.results[0]
            return json.dumps({
                "ticker": ticker,
                "name": data.name,
                "sector": data.sector,
                "industry": data.industry,
                "country": data.country,
                "website": data.website,
                "description": data.description[:500] if data.description else "N/A",
                "ceo": getattr(data, 'ceo', 'N/A'),
                "employees": getattr(data, 'full_time_employees', 'N/A')
            })
        else:
            return json.dumps({"error": f"No profile data for {ticker}"})
            
    except Exception as e:
        logger.error(f"Company profile error: {e}")
        return json.dumps({"error": str(e)})


@tool
def get_financial_metrics(ticker: str) -> str:
    """Get financial ratios: P/E, EPS, revenue, ROE, profit margins.
    
    Args:
        ticker: Stock ticker symbol
    """
    try:
        ticker = ticker.strip().upper()
        metrics_data = obb.equity.fundamental.metrics(symbol=ticker, provider="yfinance")
        
        if metrics_data and metrics_data.results:
            data = metrics_data.results[0]
            return json.dumps({
                "ticker": ticker,
                "market_cap": getattr(data, 'market_cap', 0),
                "pe_ratio": getattr(data, 'pe_ratio', None),
                "forward_pe": getattr(data, 'forward_pe', None),
                "peg_ratio": getattr(data, 'peg_ratio', None),
                "price_to_book": getattr(data, 'price_to_book', None),
                "revenue": getattr(data, 'revenue', 0),
                "revenue_growth": getattr(data, 'revenue_growth', None),
                "net_income": getattr(data, 'net_income', 0),
                "profit_margin": getattr(data, 'profit_margin', None),
                "eps": getattr(data, 'eps', None),
                "return_on_equity": getattr(data, 'return_on_equity', None),
                "debt_to_equity": getattr(data, 'debt_to_equity', None)
            })
        else:
            return json.dumps({"error": f"No metrics data for {ticker}"})
            
    except Exception as e:
        logger.error(f"Financial metrics error: {e}")
        return json.dumps({"error": str(e)})


@tool
def get_market_news(query: Optional[str] = None) -> str:
    """Get recent news about a company or general market news.
    
    Args:
        query: Optional ticker symbol or topic. Leave empty for general market news.
    """
    try:
        if query and query.strip():
            if len(query.strip()) <= 5:
                news_data = obb.news.company(
                    symbol=query.strip().upper(),
                    provider="yfinance",
                    limit=5
                )
            else:
                news_data = obb.news.world(provider="yfinance", limit=5)
        else:
            news_data = obb.news.world(provider="yfinance", limit=5)
        
        if news_data and news_data.results:
            articles = []
            for article in news_data.results[:5]:
                articles.append({
                    "title": article.title,
                    "date": str(article.date),
                    "url": article.url,
                    "text": getattr(article, 'text', '')[:200]
                })
            return json.dumps({"news": articles})
        else:
            return json.dumps({"error": "No news available"})
            
    except Exception as e:
        logger.error(f"Market news error: {e}")
        return json.dumps({"error": str(e)})


@tool
def compare_stocks(tickers: List[str]) -> str:
    """Compare 2-4 stocks across key metrics like price, market cap, P/E ratio.
    
    Args:
        tickers: List of 2-4 stock ticker symbols to compare (e.g., ['AAPL', 'MSFT'])
    """
    try:
        comparison_data = []
        
        for ticker in tickers[:4]:
            ticker = ticker.strip().upper()
            
            try:
                quote_data = obb.equity.price.quote(symbol=ticker, provider="yfinance")
                metrics_data = obb.equity.fundamental.metrics(symbol=ticker, provider="yfinance")
                
                if quote_data and quote_data.results:
                    quote = quote_data.results[0]
                    
                    # Handle different field names
                    price = getattr(quote, 'last_price', None) or getattr(quote, 'price', None)
                    change_pct = getattr(quote, 'change_percent', None) or getattr(quote, 'percent_change', None)
                    mkt_cap = getattr(quote, 'market_cap', None)
                    
                    comp_entry = {
                        "ticker": ticker,
                        "price": round(float(price), 2) if price else None,
                        "change_percent": round(float(change_pct), 2) if change_pct else None,
                        "market_cap": mkt_cap,
                    }
                    
                    # Try to get metrics if available
                    if metrics_data and metrics_data.results:
                        metrics = metrics_data.results[0]
                        comp_entry.update({
                            "pe_ratio": getattr(metrics, 'pe_ratio', None),
                            "eps": getattr(metrics, 'eps', None),
                        })
                    
                    comparison_data.append(comp_entry)
            except Exception as ticker_err:
                logger.warning(f"Failed to get data for {ticker}: {ticker_err}")
                continue
        
        return json.dumps({"comparison": comparison_data})
        
    except Exception as e:
        logger.error(f"Stock comparison error: {e}")
        return json.dumps({"error": str(e)})


@tool
def get_historical_data(ticker: str, period: str = "1y") -> str:
    """Get historical price data and performance over time.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y')
    """
    try:
        ticker = ticker.strip().upper()
        
        period_map = {
            "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
            "6mo": 180, "1y": 365, "2y": 730, "5y": 1825
        }
        
        days = period_map.get(period, 365)
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        hist_data = obb.equity.price.historical(
            symbol=ticker,
            provider="yfinance",
            start_date=start_date
        )
        
        if hist_data and hist_data.results:
            results = hist_data.results
            start_price = results[0].close
            end_price = results[-1].close
            high_price = max([r.high for r in results])
            low_price = min([r.low for r in results])
            
            total_return = ((end_price - start_price) / start_price) * 100
            
            return json.dumps({
                "ticker": ticker,
                "period": period,
                "start_price": round(start_price, 2),
                "end_price": round(end_price, 2),
                "high_price": round(high_price, 2),
                "low_price": round(low_price, 2),
                "total_return_percent": round(total_return, 2),
                "data_points": len(results)
            })
        else:
            return json.dumps({"error": f"No historical data for {ticker}"})
            
    except Exception as e:
        logger.error(f"Historical data error: {e}")
        return json.dumps({"error": str(e)})


@tool
def analyze_market_sector(sector: Optional[str] = None) -> str:
    """Analyze market indices (S&P 500, Dow, NASDAQ) and optionally a specific sector.
    
    Args:
        sector: Optional specific sector to analyze (e.g., 'technology', 'healthcare')
    """
    try:
        indices = ["^GSPC", "^DJI", "^IXIC"]  # S&P 500, Dow, NASDAQ
        market_data = []
        
        for index in indices:
            quote_data = obb.equity.price.quote(symbol=index, provider="yfinance")
            if quote_data and quote_data.results:
                data = quote_data.results[0]
                price = getattr(data, 'last_price', None) or getattr(data, 'price', None)
                change_pct = getattr(data, 'change_percent', None) or getattr(data, 'percent_change', None)
                market_data.append({
                    "index": index,
                    "price": round(float(price), 2) if price else None,
                    "change_percent": round(float(change_pct), 2) if change_pct else None
                })
        
        result = {
            "market_indices": market_data,
            "analysis_time": datetime.now(timezone.utc).isoformat()
        }
        
        if sector:
            result["sector"] = sector
            result["note"] = f"Sector analysis for {sector} - use sector ETFs for detailed analysis."
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Market analysis error: {e}")
        return json.dumps({"error": str(e)})


# ===== CHATBOT CLASS =====

class FinanceChatbot:
    """CLI-based Finance AI Chatbot using Gemini 3 and LangGraph"""
    
    def __init__(self):
        # Fetch API key
        google_api_key = os.getenv('GOOGLE_API_KEY')
        openbb_pat = os.getenv('OPENBB_PAT')

        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set in the .env file.")
        
        # Initialize Gemini 3 Flash Preview
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=google_api_key,
            temperature=0.3,
        )
        
        # OpenBB Configuration
        if not openbb_pat:
            logger.warning("OPENBB_PAT not set. Some features may be limited.")
        else:
            logger.info("OpenBB PAT detected via environment")
        
        # Define tools
        self.tools = [
            get_stock_quote,
            get_company_profile,
            get_financial_metrics,
            get_market_news,
            compare_stocks,
            get_historical_data,
            analyze_market_sector,
        ]
        
        # System prompt
        self.system_prompt = """You are an expert AI financial analyst with access to real-time market data.

YOUR CAPABILITIES:
- Real-time stock quotes, company profiles, and financial metrics via OpenBB
- Historical price data and performance analysis
- Stock comparisons and market sector analysis
- Market news retrieval

TOOL USAGE:
- For stock prices: use get_stock_quote
- For company info: use get_company_profile
- For fundamentals (P/E, EPS): use get_financial_metrics
- For news: use get_market_news
- For comparisons: use compare_stocks
- For trends: use get_historical_data
- For market overview: use analyze_market_sector

RESPONSE GUIDELINES:
- Provide specific numbers, not vague statements
- Explain what the data means in practical terms
- Compare to benchmarks when relevant
- Highlight risks and limitations
- Be concise but thorough"""

        # Create LangGraph ReAct agent
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_prompt,
        )
        
        # Conversation state (messages)
        self.messages = []
    
    def chat(self, query: str) -> str:
        """Process a query and return the response"""
        try:
            # Add user message to history
            self.messages.append({"role": "user", "content": query})
            
            # Invoke the agent
            result = self.agent.invoke({"messages": self.messages})
            
            # Extract the last AI message
            last_message = result["messages"][-1]
            
            # Handle different content formats
            if hasattr(last_message, 'content'):
                content = last_message.content
                # Handle content as list of content blocks
                if isinstance(content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and 'text' in block:
                            text_parts.append(block['text'])
                        elif isinstance(block, str):
                            text_parts.append(block)
                    response = '\n'.join(text_parts) if text_parts else str(content)
                else:
                    response = str(content)
            else:
                response = str(last_message)
            
            # Add assistant response to history
            self.messages.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.messages = []


def main():
    """CLI interface for the chatbot"""
    print("\n" + "="*60)
    print("  💰 Finance AI Chatbot")
    print("  Powered by Gemini 3 Flash + LangGraph + OpenBB")
    print("="*60)
    print("\nCommands:")
    print("  'quit' or 'exit' - End the session")
    print("  'clear' - Clear conversation history\n")
    
    try:
        chatbot = FinanceChatbot()
        print("✅ Chatbot initialized successfully!\n")
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        return
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == 'clear':
                chatbot.clear_history()
                print("Conversation history cleared.\n")
                continue
            
            print("\n🤔 Thinking...\n")
            response = chatbot.chat(user_input)
            print(f"🤖 Assistant: {response}\n")
            print("-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
