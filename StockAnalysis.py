

pip install crewai crewai_tools python-dotenv

import os
from datetime import datetime
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# --- 1. Environment and Tool Setup ---
load_dotenv()  # Loads your .env file (for API keys)

# Safety: Check API keys exist
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
if not OPENAI_API_KEY or not SERPER_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY and SERPER_API_KEY in your .env file.")

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# --- 2. Main Crew Class ---
class StockAnalysisCrew:
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol.upper().strip()

    def create_agents(self):
        # 1. News Researcher Agent
        news_researcher = Agent(
            role='Stock News Researcher',
            goal=f'Gather and analyze the latest news, market sentiment, and recent developments about {self.stock_symbol}',
            backstory="""You are a top financial news researcher, adept at identifying credible news and analyzing its likely market impact.""",
            tools=[search_tool, scrape_tool],
            verbose=True,
            allow_delegation=False
            model="gpt-4.1-nano"
        )
        # 2. Fundamental Analyst
        fundamental_analyst = Agent(
            role='Fundamental Analysis Expert',
            goal=f'Perform comprehensive fundamental analysis of {self.stock_symbol}',
            backstory="""A fundamental analyst who dissects business models, leadership, growth, and competitive positioning.""",
            tools=[search_tool, scrape_tool],
            verbose=True,
            allow_delegation=False
        )
        # 3. Financial Analyst
        financial_analyst = Agent(
            role='Financial Data Analyst',
            goal=f'Analyze financial statements, ratios, and key financial metrics for {self.stock_symbol}',
            backstory="""An expert in financial ratios, statements, and trend analysis for investment decision-making.""",
            tools=[search_tool, scrape_tool],
            verbose=True,
            allow_delegation=False
        )
        # 4. Senior Stock Analyst
        stock_analyst = Agent(
            role='Senior Stock Analyst',
            goal=f'Synthesize all research and provide a final investment recommendation for {self.stock_symbol}',
            backstory="""A senior analyst with 15+ years of experience, providing actionable investment recommendations.""",
            verbose=True,
            allow_delegation=False
        )
        return news_researcher, fundamental_analyst, financial_analyst, stock_analyst

    def create_tasks(self, agents):
        news_researcher, fundamental_analyst, financial_analyst, stock_analyst = agents
        # 1. News Research Task
        news_research_task = Task(
            description=(
                f"Research and analyze the latest news about {self.stock_symbol}. Focus on:\n"
                "• News from the past 30 days\n• Earnings/Guidance\n• Management/Strategy changes\n"
                "• Industry trends\n• Legal/Regulatory news\n• Market sentiment\n"
                "Provide a summary with likely impact on the stock price."
            ),
            agent=news_researcher,
            expected_output="Comprehensive news analysis with key findings and stock price implications."
        )
        # 2. Fundamental Analysis Task
        fundamental_analysis_task = Task(
            description=(
                f"Perform in-depth fundamental analysis of {self.stock_symbol}:\n"
                "• Business model\n• Competitive position\n• Management quality\n"
                "• Growth prospects\n• SWOT\n• Moats/competitive advantages\n"
                "Use news findings to inform your analysis."
            ),
            agent=fundamental_analyst,
            expected_output="Full fundamental analysis with assessment of business quality.",
            context=[news_research_task]
        )
        # 3. Financial Analysis Task
        financial_analysis_task = Task(
            description=(
                f"Analyze financials of {self.stock_symbol}:\n"
                "• Revenue/profit trends (last 5 years)\n• Key ratios (P/E, ROE, D/E, etc.)\n"
                "• Cash flows\n• Margins\n• Dividend history\n• Peer comparison\n"
                "Consider the previous analyses for context."
            ),
            agent=financial_analyst,
            expected_output="Detailed financial health report with key ratios and trends.",
            context=[news_research_task, fundamental_analysis_task]
        )
        # 4. Investment Recommendation Task
        investment_recommendation_task = Task(
            description=(
                f"Based on the news, fundamental, and financial analyses, provide a final investment call for {self.stock_symbol}:\n"
                "• Clear BUY/HOLD/SELL recommendation\n• Target price range\n• Risks/catalysts\n"
                "• Investment timeline\n• Sizing suggestion\n• Key metrics to monitor\n"
                "• Scenarios (bull/base/bear)"
            ),
            agent=stock_analyst,
            expected_output="Final investment report with clear recommendation and rationale.",
            context=[news_research_task, fundamental_analysis_task, financial_analysis_task]
        )
        return [news_research_task, fundamental_analysis_task, financial_analysis_task, investment_recommendation_task]

    def run_analysis(self, verbose_level=2, save_report=True, custom_save_path=None):
        agents = self.create_agents()
        tasks = self.create_tasks(agents)
        crew = Crew(
            agents=list(agents),
            tasks=tasks,
            process=Process.sequential,
            verbose=verbose_level
        )
        print(f"\n🚀 Starting comprehensive analysis for {self.stock_symbol}...\n" + "="*60)
        try:
            result = crew.kickoff()
        except Exception as e:
            print(f"❌ Error during Crew execution: {e}")
            return None
        print(f"\n✅ Analysis complete for {self.stock_symbol}\n" + "="*60)

        if save_report and result:
            filename = custom_save_path or f"{self.stock_symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Stock Analysis Report for {self.stock_symbol}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                f.write(str(result))
            print(f"💾 Report saved to: {filename}")
        return result

# --- 3. Command-line/Interactive Entry Point ---
if __name__ == "__main__":
    print("====== STOCK ANALYSIS AI CREW ======")
    stock_symbol = input("Enter stock symbol (e.g., AAPL, RELIANCE, TSLA): ").strip().upper()
    if not stock_symbol.isalnum():
        print("❌ Please enter a valid stock symbol (letters/numbers only).")
        exit(1)

    # Optional: Let user customize save path
    save_opt = input("Do you want to specify the report save path? (y/N): ").strip().lower()
    save_path = None
    if save_opt == "y":
        save_path = input("Enter the full file path for saving the report: ").strip()
    # Run analysis
    crew = StockAnalysisCrew(stock_symbol)
    result = crew.run_analysis(verbose_level=2, save_report=True, custom_save_path=save_path)
    if result:
        print("\n📊 FINAL RECOMMENDATION:\n" + "="*40)
        print(result)
    else:
        print("No result produced.")

# --- 4. Single-task shortcut for advanced users ---
def run_single_analysis(stock_symbol, task_type="full"):
    """
    Run a single step of the analysis pipeline for the given stock symbol.
    task_type: "news", "fundamental", "financial", "recommendation", or "full"
    """
    crew = StockAnalysisCrew(stock_symbol)
    agents = crew.create_agents()
    tasks = crew.create_tasks(agents)
    if task_type == "news":
        single_crew = Crew(agents=[agents[0]], tasks=[tasks[0]], verbose=2)
    elif task_type == "fundamental":
        single_crew = Crew(agents=agents[:2], tasks=tasks[:2], verbose=2)
    elif task_type == "financial":
        single_crew = Crew(agents=agents[:3], tasks=tasks[:3], verbose=2)
    else:
        single_crew = Crew(agents=list(agents), tasks=tasks, process=Process.sequential, verbose=2)
    return single_crew.kickoff()

