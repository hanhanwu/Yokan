import os
import yaml
import asyncio
import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

with open('eval_prompts.yaml', 'r') as f:
    prompt_versions = yaml.safe_load(f)


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _invoke_with_retry(chain, inputs):
    return await chain.ainvoke(inputs)


# ------------------------------------------ RETRIEVAL RELEVANCY ------------------------------------------ #
class RetrievalRelevancy(BaseModel):
    score: int = Field(description="""Score with:
                - Only generate the score as 0, 1, 2 or 3
                - Scoring as 0: the RETRIEVED CHUNK is completely irrelevant to the USER QUERY
                - Scoring as 1: loosely related topic but no useful information to answer the query
                - Scoring as 2: relevant and partially useful for answering the query
                - Scoring as 3: highly relevant and sufficient to fully or mostly answer the query
            """)
    reasoning: str = Field(description="Brief reasoning for the given score.")


async def evaluate_retrieval_relevancy_async(llm, user_query, retrieved_chunk, rr_prompt_template):
    output_parser = PydanticOutputParser(pydantic_object=RetrievalRelevancy)
    prompt = PromptTemplate(
        template=rr_prompt_template,
        input_variables=["user_query", "retrieved_chunk"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    chain = prompt | llm | output_parser
    result = await _invoke_with_retry(chain, {
        "user_query": user_query,
        "retrieved_chunk": retrieved_chunk,
    })
    return result


async def _process_retrieval_relevancy_record_async(llm, record, chunk_col, score_col, reasoning_col, rr_prompt_template):
    eval_result = await evaluate_retrieval_relevancy_async(
        llm,
        record['query'],
        record[chunk_col],
        rr_prompt_template,
    )
    record[score_col] = eval_result.score
    record[reasoning_col] = eval_result.reasoning
    return record


async def get_retrieval_relevancy_output_async(input_df, llm, chunk_col='retrieved_chunk',
                                           score_col='retrieval_relevancy_score', reasoning_col='rr_reasoning',
                                           rr_prompt_template=None, concurrency=2):
    """Evaluate retrieval relevancy for every row in *input_df*.

    Expected columns: ``query`` and the column named by *chunk_col* (default ``retrieved_chunk``).
    Returns a new DataFrame with two additional columns named by *score_col* and *reasoning_col*.
    """
    if rr_prompt_template is None:
        rr_prompt_template = prompt_versions['rr_prompt_template']

    sem = asyncio.Semaphore(concurrency)

    async def sem_task(record):
        async with sem:
            return await _process_retrieval_relevancy_record_async(
                llm, record, chunk_col, score_col, reasoning_col, rr_prompt_template=rr_prompt_template
            )

    input_records = input_df.to_dict(orient='records')
    tasks = [sem_task(record) for record in input_records]
    output_lst = await asyncio.gather(*tasks)
    return pd.DataFrame(output_lst)
# ------------------------------------------ RETRIEVAL RELEVANCY ------------------------------------------ #
