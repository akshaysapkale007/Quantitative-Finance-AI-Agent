"""
Finance AI Chatbot - Flask Backend
Serves the chat UI and handles API requests using LangGraph + Gemini 3
"""
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# ===== TOOL DEFINITIONS =====

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
                "eps": getattr(data, 'eps', None),
                "revenue": getattr(data, 'revenue', 0),
                "profit_margin": getattr(data, 'profit_margin', None),
            })
        else:
            return json.dumps({"error": f"No metrics data for {ticker}"})
            
    except Exception as e:
        logger.error(f"Financial metrics error: {e}")
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
                
                if quote_data and quote_data.results:
                    quote = quote_data.results[0]
                    price = getattr(quote, 'last_price', None) or getattr(quote, 'price', None)
                    mkt_cap = getattr(quote, 'market_cap', None)
                    
                    comparison_data.append({
                        "ticker": ticker,
                        "price": round(float(price), 2) if price else None,
                        "market_cap": mkt_cap,
                    })
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
        period: Time period (e.g., '1mo', '3mo', '6mo', '1y')
    """
    try:
        ticker = ticker.strip().upper()
        
        period_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}
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
            total_return = ((end_price - start_price) / start_price) * 100
            
            return json.dumps({
                "ticker": ticker,
                "period": period,
                "start_price": round(start_price, 2),
                "end_price": round(end_price, 2),
                "total_return_percent": round(total_return, 2),
            })
        else:
            return json.dumps({"error": f"No historical data for {ticker}"})
            
    except Exception as e:
        logger.error(f"Historical data error: {e}")
        return json.dumps({"error": str(e)})


@tool
def analyze_market_sector(sector: Optional[str] = None) -> str:
    """Analyze market indices (S&P 500, Dow, NASDAQ).
    
    Args:
        sector: Optional specific sector to analyze
    """
    try:
        indices = ["^GSPC", "^DJI", "^IXIC"]
        market_data = []
        
        for index in indices:
            quote_data = obb.equity.price.quote(symbol=index, provider="yfinance")
            if quote_data and quote_data.results:
                data = quote_data.results[0]
                price = getattr(data, 'last_price', None) or getattr(data, 'price', None)
                change_pct = getattr(data, 'change_percent', None)
                market_data.append({
                    "index": index,
                    "price": round(float(price), 2) if price else None,
                    "change_percent": round(float(change_pct), 2) if change_pct else None
                })
        
        return json.dumps({"market_indices": market_data})
        
    except Exception as e:
        logger.error(f"Market analysis error: {e}")
        return json.dumps({"error": str(e)})


@tool
def get_top_gainers() -> str:
    """Get today's top gaining stocks in the market.
    
    Returns the stocks with the highest percentage gains today.
    """
    try:
        gainers_data = obb.equity.discovery.gainers(provider="yfinance")
        
        if gainers_data and gainers_data.results:
            top_gainers = []
            for stock in gainers_data.results[:10]:  # Top 10
                top_gainers.append({
                    "symbol": getattr(stock, 'symbol', 'N/A'),
                    "name": getattr(stock, 'name', 'N/A'),
                    "price": round(float(getattr(stock, 'price', 0)), 2),
                    "change_percent": round(float(getattr(stock, 'percent_change', 0) or getattr(stock, 'change_percent', 0)), 2)
                })
            return json.dumps({"top_gainers": top_gainers})
        else:
            return json.dumps({"error": "No gainers data available"})
            
    except Exception as e:
        logger.error(f"Top gainers error: {e}")
        return json.dumps({"error": str(e)})


@tool
def get_top_losers() -> str:
    """Get today's top losing stocks in the market.
    
    Returns the stocks with the largest percentage losses today.
    """
    try:
        losers_data = obb.equity.discovery.losers(provider="yfinance")
        
        if losers_data and losers_data.results:
            top_losers = []
            for stock in losers_data.results[:10]:  # Top 10
                top_losers.append({
                    "symbol": getattr(stock, 'symbol', 'N/A'),
                    "name": getattr(stock, 'name', 'N/A'),
                    "price": round(float(getattr(stock, 'price', 0)), 2),
                    "change_percent": round(float(getattr(stock, 'percent_change', 0) or getattr(stock, 'change_percent', 0)), 2)
                })
            return json.dumps({"top_losers": top_losers})
        else:
            return json.dumps({"error": "No losers data available"})
            
    except Exception as e:
        logger.error(f"Top losers error: {e}")
        return json.dumps({"error": str(e)})


@tool
def analyze_portfolio(holdings: str) -> str:
    """Analyze a stock portfolio with real-time data. Returns total value, P&L, sector breakdown, and metrics.
    
    This is a UNIQUE feature that ChatGPT and Gemini web apps cannot do!
    
    Args:
        holdings: Portfolio holdings as comma-separated "TICKER:SHARES" pairs.
                  Example: "AAPL:50, MSFT:30, TSLA:20, GOOGL:10"
    """
    try:
        # Parse holdings
        portfolio = []
        pairs = holdings.replace(" ", "").split(",")
        
        for pair in pairs:
            if ":" in pair:
                parts = pair.split(":")
                ticker = parts[0].strip().upper()
                try:
                    shares = float(parts[1].strip())
                    portfolio.append({"ticker": ticker, "shares": shares})
                except ValueError:
                    continue
        
        if not portfolio:
            return json.dumps({"error": "Could not parse portfolio. Use format: AAPL:50, MSFT:30"})
        
        # Analyze each holding
        total_value = 0
        total_daily_change = 0
        holdings_data = []
        sector_allocation = {}
        
        for holding in portfolio:
            ticker = holding["ticker"]
            shares = holding["shares"]
            
            try:
                # Get quote data
                quote_data = obb.equity.price.quote(symbol=ticker, provider="yfinance")
                
                if quote_data and quote_data.results:
                    quote = quote_data.results[0]
                    price = float(getattr(quote, 'last_price', None) or getattr(quote, 'price', None) or 0)
                    change = float(getattr(quote, 'change', None) or 0)
                    change_pct = float(getattr(quote, 'change_percent', None) or 0)
                    
                    # If change is 0 (market closed), calculate from historical data
                    if change == 0 and price > 0:
                        try:
                            # Get last 5 days of data to ensure we have 2 trading days
                            hist = obb.equity.price.historical(
                                symbol=ticker,
                                provider="yfinance",
                                start_date=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                            )
                            if hist and hist.results and len(hist.results) >= 2:
                                today_close = hist.results[-1].close
                                prev_close = hist.results[-2].close
                                change = today_close - prev_close
                                change_pct = (change / prev_close) * 100 if prev_close > 0 else 0
                        except Exception as hist_err:
                            logger.warning(f"Could not get historical data for {ticker}: {hist_err}")
                    
                    position_value = price * shares
                    daily_pnl = change * shares
                    
                    total_value += position_value
                    total_daily_change += daily_pnl
                    
                    # Get sector info
                    try:
                        profile = obb.equity.profile(symbol=ticker, provider="yfinance")
                        if profile and profile.results:
                            sector = profile.results[0].sector or "Unknown"
                        else:
                            sector = "Unknown"
                    except:
                        sector = "Unknown"
                    
                    # Track sector allocation
                    if sector in sector_allocation:
                        sector_allocation[sector] += position_value
                    else:
                        sector_allocation[sector] = position_value
                    
                    holdings_data.append({
                        "ticker": ticker,
                        "shares": shares,
                        "price": round(price, 2),
                        "value": round(position_value, 2),
                        "daily_change_pct": round(change_pct, 2),
                        "daily_pnl": round(daily_pnl, 2),
                        "sector": sector
                    })
                    
            except Exception as e:
                logger.warning(f"Error fetching {ticker}: {e}")
                holdings_data.append({
                    "ticker": ticker,
                    "shares": shares,
                    "error": str(e)
                })
        
        # Calculate sector percentages
        sector_breakdown = []
        for sector, value in sorted(sector_allocation.items(), key=lambda x: x[1], reverse=True):
            pct = (value / total_value * 100) if total_value > 0 else 0
            sector_breakdown.append({
                "sector": sector,
                "value": round(value, 2),
                "percentage": round(pct, 1)
            })
        
        # Calculate portfolio metrics
        daily_return_pct = (total_daily_change / (total_value - total_daily_change) * 100) if (total_value - total_daily_change) > 0 else 0
        
        result = {
            "portfolio_summary": {
                "total_value": round(total_value, 2),
                "daily_pnl": round(total_daily_change, 2),
                "daily_return_pct": round(daily_return_pct, 2),
                "num_holdings": len(holdings_data),
                "direction": "📈 UP" if total_daily_change > 0 else "📉 DOWN" if total_daily_change < 0 else "➡️ FLAT"
            },
            "holdings": holdings_data,
            "sector_breakdown": sector_breakdown,
            "analysis_note": "This real-time portfolio analysis is a unique feature that ChatGPT and Gemini cannot provide!"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Portfolio analysis error: {e}")
        return json.dumps({"error": str(e)})


# ===== FLASK APP =====

app = Flask(__name__)
CORS(app)  # Enable CORS for API requests

# Initialize chatbot on startup
agent = None
messages = []


def init_agent():
    """Initialize the LangGraph agent"""
    global agent
    
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=google_api_key,
        temperature=0.3,
    )
    
    tools = [
        get_stock_quote,
        get_company_profile,
        get_financial_metrics,
        compare_stocks,
        get_historical_data,
        analyze_market_sector,
        get_top_gainers,
        get_top_losers,
        analyze_portfolio,
    ]
    
    system_prompt = """You are an expert AI financial analyst with access to real-time market data.

IMPORTANT: US stock markets (NYSE, NASDAQ) are open Mon-Fri 9:30 AM - 4:00 PM Eastern Time.
If markets are closed, say "As of market close" or "In the last trading session" instead of "during the current session".

TOOL USAGE:
- For stock prices: use get_stock_quote
- For company info: use get_company_profile
- For fundamentals: use get_financial_metrics
- For comparisons: use compare_stocks
- For trends: use get_historical_data
- For market overview: use analyze_market_sector
- For top gainers: use get_top_gainers
- For top losers: use get_top_losers
- For PORTFOLIO ANALYSIS: use analyze_portfolio (UNIQUE FEATURE!)
  Example: "Analyze my portfolio: AAPL:50, MSFT:30, TSLA:20"

Be concise and provide specific numbers. Use appropriate phrasing based on market hours."""

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )
    
    logger.info("Agent initialized successfully")


def parse_response(result):
    """Parse agent response"""
    last_message = result["messages"][-1]
    
    if hasattr(last_message, 'content'):
        content = last_message.content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and 'text' in block:
                    text_parts.append(block['text'])
                elif isinstance(block, str):
                    text_parts.append(block)
            return '\n'.join(text_parts) if text_parts else str(content)
        else:
            return str(content)
    return str(last_message)


@app.route('/')
def index():
    """Serve the chat UI"""
    return render_template('chat.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat API requests"""
    global messages
    
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'response': 'Please provide a message.'}), 400
        
        # Add user message to history
        messages.append({"role": "user", "content": user_message})
        
        # Get agent response
        result = agent.invoke({"messages": messages})
        response = parse_response(result)
        
        # Add to history
        messages.append({"role": "assistant", "content": response})
        
        # Keep history manageable
        if len(messages) > 20:
            messages = messages[-20:]
        
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'response': f'Error: {str(e)}'}), 500


@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    global messages
    messages = []
    return jsonify({'status': 'cleared'})


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'agent_initialized': agent is not None
    })


if __name__ == '__main__':
    # Initialize agent
    init_agent()
    
    # Run Flask server
    logger.info("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)