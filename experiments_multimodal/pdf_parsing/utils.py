import os
import yaml
import asyncio
import pandas as pd
from typing import Optional
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


# ------------------------------------------ ANSWER QUALITY ------------------------------------------ #
class AnswerQuality(BaseModel):
    score: int = Field(description="""Score with:
                - Only generate the score as -1, 0, 1, 2 or 3
                - Scoring as -1: the ANSWER is hallucination
                - Scoring as 0: the ANSWER does not address the USER QUERY at all
                - Scoring as 1: loosely related but fails to answer the query
                - Scoring as 2: partially addresses the USER QUERY
                - Scoring as 3: fully and correctly addresses the USER QUERY
            """)
    grounded_in_context: Optional[bool] = Field(description="""
                - null if retrieval_relevancy_score is 0 or 1 (context not reliable enough to judge grounding)
                - true if the answer is consistent with and supported by the retrieved chunk
                - false if the answer contradicts or goes beyond the retrieved chunk
            """)
    reasoning: str = Field(description="Brief reasoning covering whether the answer addresses the query and, if applicable, faithfulness to the retrieved chunk.")


async def evaluate_answer_quality_async(llm, user_query, answer, retrieved_chunk, retrieval_relevancy_score, aq_prompt_template):
    output_parser = PydanticOutputParser(pydantic_object=AnswerQuality)
    prompt = PromptTemplate(
        template=aq_prompt_template,
        input_variables=["user_query", "answer", "retrieved_chunk", "retrieval_relevancy_score"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    chain = prompt | llm | output_parser
    result = await _invoke_with_retry(chain, {
        "user_query": user_query,
        "answer": answer,
        "retrieved_chunk": retrieved_chunk,
        "retrieval_relevancy_score": retrieval_relevancy_score,
    })
    return result


async def _process_answer_quality_record_async(llm, record, answer_col, chunk_col, rr_score_col,
                                               score_col, grounded_col, reasoning_col, aq_prompt_template):
    eval_result = await evaluate_answer_quality_async(
        llm,
        record["query"],
        record[answer_col],
        record.get(chunk_col, ""),
        record.get(rr_score_col, 0),
        aq_prompt_template,
    )
    record[score_col] = eval_result.score
    record[grounded_col] = eval_result.grounded_in_context
    record[reasoning_col] = eval_result.reasoning
    return record


async def get_answer_quality_output_async(input_df, llm, answer_col="answer", chunk_col="retrieved_chunk",
                                          rr_score_col="retrieval_relevancy_score",
                                          score_col="answer_quality_score",
                                          grounded_col="aq_grounded_in_context",
                                          reasoning_col="aq_reasoning",
                                          aq_prompt_template=None, concurrency=2):
    """Evaluate answer quality for every row in *input_df*.

    Expected columns: ``query``, the column named by *answer_col* (default ``answer``),
    optionally *chunk_col* and *rr_score_col* for grounding checks when retrieval relevancy score is 2 or 3.
    Returns a new DataFrame with additional columns named by *score_col*, *grounded_col*, and *reasoning_col*.
    """
    if aq_prompt_template is None:
        aq_prompt_template = prompt_versions["aq_prompt_template"]

    sem = asyncio.Semaphore(concurrency)

    async def sem_task(record):
        async with sem:
            return await _process_answer_quality_record_async(
                llm, record, answer_col, chunk_col, rr_score_col,
                score_col, grounded_col, reasoning_col, aq_prompt_template=aq_prompt_template
            )

    input_records = input_df.to_dict(orient="records")
    tasks = [sem_task(record) for record in input_records]
    output_lst = await asyncio.gather(*tasks)
    return pd.DataFrame(output_lst)
# ------------------------------------------ ANSWER QUALITY ------------------------------------------ #
