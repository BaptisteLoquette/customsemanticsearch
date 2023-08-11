"""
Generate dataset of (query, document) pairs with OpenAI's gpt-3.5-turbo
"""
import os
import argparse
from langchain.llms import OpenAI
from src.utils.synthetic_data_utils import write_to_csv_query_dset_openai, get_prompt_and_instructions

parser  =   argparse.ArgumentParser(
    prog="generate_query_dataset_openai",
    description="Generate a queries dataset on any csv dataset with OpenAI's gpt-3.5-turbo"
)
parser.add_argument('openai_api_key', help="Your OpenAI API key", type=str)
parser.add_argument('csv_path', help="Input path of the CSV dataset, must contain a columns 'text' where ", type=str)
parser.add_argument('path_out', help="Output path to save the CSV generated dataset", type=str)
parser.add_argument('--n_queries', help="Output path to save the CSV generated dataset", type=int, default=3)

if "__main__" == __name__:
    args            =   parser.parse_args()
    openai_api_key  =   args.openai_api_key
    n_queries       =   args.n_queries

    os.environ['OPENAI_API_KEY']    =   openai_api_key
    llm =   OpenAI(temperature=0.2)

    prompt_template, format_instructions    =   get_prompt_and_instructions()   # Get PromptTemplate and Format Instructions for OpenAI API call
    # Generate and write to csv the generated dataset
    write_to_csv_query_dset_openai("huff_post_news_articles.csv", "test_openai.csv", llm, prompt_template, format_instructions, n_queries=n_queries)