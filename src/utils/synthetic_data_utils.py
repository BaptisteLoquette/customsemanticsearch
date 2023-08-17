"""
Utility functions for synthetic data.

Eg. Query Generation
"""
from typing import List, Union
import re
import csv
import json
import time
import pandas as pd
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from openai.error import InvalidRequestError, Timeout

from src.utils.infer_utils import generate_single_query

def write_to_csv_query_dset_t5(csv_path:str, path_out:str, model:T5ForConditionalGeneration, tokenizer:T5Tokenizer, device, positive_and_negative=False, n_queries=3) -> None:
    """Writes the (generated queries, document) pairs into a csv
    Inputs :
        - csv_path : the path of the csv dataset you want to annotate (must contain a column "text")
        - path_out : the path to save the CSV file
        - model : the pretrained T5 model for query generation
        - tokenizer : the model's tokenizer
        - device
        - positive_and_negative : True if the user wants to generative positive queries as well as hard negative queries for the retriever
    """

    df  =   pd.read_csv(csv_path, sep="\t")   # Load dataset

    # Init CSV file
    f       =   open(path_out, 'w')
    writer  =   csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow(["query"] + ["document"] + ["label"])

    for sample in tqdm(df['text']):
        doc             =   re.sub("\n", " ", sample)   # Format text
        greedy_pos_q    =   generate_single_query(doc, model, tokenizer, device, positive=True, greedy=True)
        pos_gen_queries =   generate_single_query(doc, model, tokenizer, device, positive=True, n_queries=n_queries)
        ## Generating using a mixture of nucleus sampling and greedy decoding
        writer.writerow([greedy_pos_q[0]] + [doc] + [1])
        for q in pos_gen_queries:
            writer.writerow([q] + [doc] + [1])  # 1 to indicate that the generated query is positive

        if positive_and_negative:
            greedy_neg_q    =   generate_single_query(doc, model, tokenizer, device, positive=False, greedy=True)
            neg_gen_queries =   generate_single_query(doc, model, tokenizer, device, positive=False, n_queries=n_queries)
            writer.writerow([greedy_neg_q[0]] + [doc] + [0])
            for q in neg_gen_queries:
                writer.writerow([q] + [doc] + [0])  # 0 to indicate that the generated query is negative

def write_to_csv_query_dset_openai(csv_path:str, path_out:str, llm:OpenAI, prompt_template:PromptTemplate, format_instructions, n_queries=3) -> None:
    """Writes the (generated queries, document) pairs into a csv
    Inputs :
        - csv_path : the path of the csv dataset you want to annotate (must contain a column "text")
        - path_out : the path to save the CSV file
        - llm : Model to generate the queries
        - prompt_template : Prompt Template given to the model
        - format_instructions : Instructions for format of the output of the llm
        - n_queries : number of queries to generate
    """
    df  =   pd.read_csv(csv_path, sep="\t")   # Load dataset

    # Init CSV file
    f       =   open(path_out, 'w')
    writer  =   csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow(["query"] + ["document"] + ["label"])

    for sample in tqdm(df['text']):
        doc         =   re.sub("\n", " ", sample)   # Format text
        gen_prompt  =   prompt_template.format_prompt(content=doc, format_instructions=format_instructions, n_queries=n_queries)    # Format prompt

        try:
            queries     =   generate_query(llm, gen_prompt)

            for q in queries:
                writer.writerow([q] + [doc] + [1])  # 1 to indicate that the generated query is positive
        except InvalidRequestError:
            continue
        except Timeout:
            print("OpenAI TimeOut")
            time.sleep(1)


def get_prompt_and_instructions() -> Union[PromptTemplate, str]:
    """Gets the Prompt Template and the Formatting Instructions for query generation using GPT"""
    response_schemas = [
    ResponseSchema(name="query", description="The generated queries based on the input text snippet"),
]

    output_parser   =   StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions =   output_parser.get_format_instructions()

    prompt  =   """Given a text input, generate {n_queries} probable queries a user might enter to retrieve it in a search engine.
    {format_instructions}

    {content}

    QUERIES :
    """

    prompt_template =   PromptTemplate.from_template(template=prompt,)

    return prompt_template, format_instructions

def generate_query(llm:OpenAI, gen_prompt:PromptTemplate) -> List[str]:
    """
    Generate queries for a single doc
    Inputs :
        - llm : OpenAI's model
        - gen_prompt : The formatted prompt (with the doc and the format instructions)
    Outputs :
        - gen_queries : The list of the n_queries generated queries
    """

    gen_queries =   []
    res         =   llm(gen_prompt.to_string())

    json_string =   re.search(r'```json\n(.*?)```', res, re.DOTALL)
    json_string =   re.search(r'```json\n(.*?)```', res, re.DOTALL).group(1)
    json_data   =   json.loads(f'[{json_string}]')
    for gen in json_data:
        gen_queries.append(gen['query'])
    
    return gen_queries