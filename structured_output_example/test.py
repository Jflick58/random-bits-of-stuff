from typing import List, Optional, Dict, Union, Literal
from pydantic import BaseModel, Field, field_validator
import re
from langchain.output_parsers import PydanticOutputParser, RetryOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")


parser = PydanticOutputParser(pydantic_object=Action)


prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

llm = ChatOpenAI(temperature=0, model="gpt-4o")

completion_chain = prompt | llm

retry_parser = RetryOutputParser.from_llm(parser=parser, llm=llm)

main_chain = RunnableParallel(
    completion=completion_chain, prompt_value=prompt
) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(completion=x["completion"].content, prompt_value=x["prompt_value"]))


print(main_chain.invoke({"query": "who is leo di caprios gf?"}))
