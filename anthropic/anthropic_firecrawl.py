import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import asyncio
from firecrawl import AsyncFirecrawlApp


OPENAI_API_KEY = "secret-api-key"
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)

class ModelPricing(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    cost_per_1M_input_token: float = Field(..., description="Cost per 1M tokens where prompts <= 200k tokens in USD")
    cost_per_1M_output_token: float = Field(..., description="Cost per 1M tokens where prompts <= 200k tokens in USD")
    prompt_caching: float = Field(..., description="Cost of Prompt Caching in USD")

class PricingList(BaseModel):
    pricinglist: List[ModelPricing]


# Firecrawl markdown scraping logic
async def scrape_anthropic_pricing_markdown():
    app = AsyncFirecrawlApp(api_key='secret-api-key')
    response = await app.scrape_url(
        url='https://www.anthropic.com/pricing#api',
        formats=['markdown'],
        only_main_content=True,
        parse_pdf=True
    )
    markdown_content = response.get('markdown') if isinstance(response, dict) else str(response)
    with open('firecrawl.md', 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    print('Markdown content saved to firecrawl.md')


def extract_pricing_from_text(text: str, model_names: List[str], provider: str) -> pd.DataFrame:
    try:
        model_list_str = "\n- " + "\n- ".join(model_names)
        prompt = PromptTemplate(
            template=f"""
You are an extraction engine.

From the following documentation content, extract pricing details ONLY for the following {provider} models:
{model_list_str}

For each model, extract:
- model_name
- cost_per_1M_input_token (USD)
- cost_per_1M_output_token (USD)
- prompt_caching (write)

Documentation:
\"\"\"{{doc}}\"\"\"
""",
            input_variables=["doc"],
        )
        chain = prompt | llm.with_structured_output(PricingList)
        output = chain.invoke({"doc": text})
        df = pd.DataFrame([m.model_dump() for m in output.pricinglist])
        df["provider"] = provider
        print('DataFrame extracted from text:')
        print(df.head())
        print('End of DataFrame\n')
        return df
    except Exception as e:
        print(f"Error extracting pricing from text: {e}")
        return pd.DataFrame()

def get_ground_truth_from_mongodb():
    try:
        mongo_uri = "monogodb_connection_string"
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        
        client.admin.command("ping")
        print("MongoDB connected successfully.")
        
        collection = client["db_name"]["collection_name"]
        
        pipeline = [
            {"$match": {
                "tool_provider": "anthropic",
                "tool_type": 'llm',
                
                "is_active": True,
                "cost_metrics": {"$exists": True}
            }},
            {"$sort": {"start_date": -1}},  
            {"$group": {
                "_id": "$model_name",
                "input_token_cost": {"$first": "$cost_metrics.input_token_cost"},
                "output_token_cost": {"$first": "$cost_metrics.output_token_cost"},
                "cached_input_token_cost": {"$first": "$cost_metrics.cached_input_token_cost"}
            }}
        ]
        
        results = list(collection.aggregate(pipeline))
        return {
            r["_id"]: {
                "input_token_cost": r["input_token_cost"], 
                "output_token_cost": r["output_token_cost"],
                "cached_input_token_cost": r.get("cached_input_token_cost", 0)
            } for r in results
        }
        
    except ConnectionFailure as e:
        print(f"Could not connect to MongoDB: {e}")
        return 0
    except Exception as e:
        print(f"Error getting ground truth from MongoDB: {e}")
        return 0

def needs_check(row, ground_truth):
    print('\n\n ground truth \n\n', ground_truth, '\n\n')
    normalized_name = row["model_name"].lower().replace(" ", "-")
    gt = ground_truth.get(normalized_name)
    print(f"\n\nChecking model: {row['model_name']}, Ground Truth: {gt}\n\n")
    mismatches = []
    if not gt:
        return ["no_ground_truth"]
    input_token_cost = gt.get("input_token_cost") or 0.0
    if "embedding" in normalized_name or (gt.get("output_token_cost") is None and gt.get("cached_input_token_cost") is None):
        if abs(row["cost_per_1M_input_token"] - input_token_cost) >= 0.01:
            mismatches.append("input_token_cost")
        return mismatches
    output_token_cost = gt.get("output_token_cost") or 0.0
    cached_input_token_cost = gt.get("cached_input_token_cost") or 0.0
    if abs(row["cost_per_1M_input_token"] - input_token_cost) >= 0.01:
        mismatches.append("input_token_cost")
    if abs(row["cost_per_1M_output_token"] - output_token_cost) >= 0.01:
        mismatches.append("output_token_cost")
    if abs(row["prompt_caching"] - cached_input_token_cost) >= 0.01:
        mismatches.append("prompt_caching")
    return mismatches


def main():
    print("Starting pricing extraction...")
    asyncio.run(scrape_anthropic_pricing_markdown())

    ground_truth = get_ground_truth_from_mongodb()
    print(f"Ground truth data: {ground_truth}")

    anthropic_models = ["Claude Sonnet 4", "Claude 3-5 Haiku", "Claude 3-7 Sonnet"]

    provider = "Anthropic"

    with open("firecrawl.md", "r", encoding="utf-8") as f:
        page_content = f.read()

    print(f"Extracting pricing for models: {anthropic_models}")
    anthropic_df = extract_pricing_from_text(page_content, anthropic_models, provider)

    if anthropic_df.empty:
        print("No pricing data extracted. Please check the markdown text and extraction logic.")
        return

    anthropic_df["need_to_check"] = anthropic_df.apply(lambda row: needs_check(row, ground_truth), axis=1)

    print("\nFinal Results:")
    print(anthropic_df.to_markdown(index=False))

    needs_check_count = anthropic_df["need_to_check"].sum()
    print(f"\nSummary: {needs_check_count} out of {len(anthropic_df)} models need manual verification.")

    anthropic_df.to_csv("anthropic_pricing_results.csv", index=False)
    print("Results saved to anthropic_pricing_results.csv")

if __name__ == "__main__":
    main()
