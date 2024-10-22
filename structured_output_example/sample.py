from typing import List, Optional, Dict, Union, Literal
from pydantic import BaseModel, Field, field_validator
import re
from langchain.output_parsers import PydanticOutputParser, RetryOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel


class ValueFrequency(BaseModel):
    value: Optional[Union[str, int, float]]
    freq: int

class ColumnStats(BaseModel):
    type: str
    has_duplicates: bool
    null_percentage: float
    min: Optional[float] = None
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None
    max: Optional[float] = None
    avg: Optional[float] = None
    pattern: Optional[str] = None
    value_frequency: Optional[List[ValueFrequency]] = None

pattern_prompt = """

        Expected regex pattern for string columns or null for non-string:
        
        1. For name fields, use regex patterns that describe the structure, like '^[A-Z][a-z]+ [A-Z][a-z]+$' for 'FirstName LastName'.
        2. For formatted numbers (e.g., SSN, phone numbers), use patterns like '^\d{{3}}-\d{{2}}-\d{{4}}$' to represent the format, not specific numbers.
        3. For dates, use patterns like '^\\d{{4}}-\\d{{2}}-\\d{{2}}$' for YYYY-MM-DD format.
        4. Use character classes (e.g., \d for digits, [A-Za-z] for letters) and quantifiers (e.g., +, {{}}, *) to describe the structure.
        5. Avoid using specific examples in the pattern; focus on the general structure.
        6. If the column doesn't require a specific pattern, you may set it to null.

        Ensure the expected_pattern accurately represents the structure of the data, not just a sample value."""

class ColumnSemanticReview(BaseModel):
    expected_type: Literal['list', 'field'] = Field(description="The expected data type")
    allow_missing: bool = Field(description="Whether missing values are allowed - true/false")
    expected_range: Optional[List[float]] = Field(description="Expected range for numeric columns", default=None)
    expected_pattern: Optional[str] = Field(description=pattern_prompt, default=None)
    potential_errors: List[str] = Field(description="Descriptions of potential errors")
    disguised_missing_values: List[str] = Field(description="Potential disguised missing values")
    missing_records: Optional[str] = Field(description="Description of potential missing records", default=None)

    @field_validator('expected_pattern')
    def validate_regex(cls, v):
        if v is not None:
            try:
                re.compile(v)
            except re.error:
                raise ValueError('Invalid regex pattern')
        return v

class SemanticContext(BaseModel):
    table_summary: str
    column_summary: Dict[str, str]

class StatisticalProfile(BaseModel):
    columns: Dict[str, ColumnStats]

class SemanticProfiler:
    def __init__(self, llm):
        self.llm = llm

    def semantic_review(self, table_name: str, semantic_context: SemanticContext, statistical_profile: StatisticalProfile) -> Dict[str, ColumnSemanticReview]:
        parser = PydanticOutputParser(pydantic_object=ColumnSemanticReview)
        retry_parser = RetryOutputParser.from_llm(parser=parser, llm=self.llm)

        prompt = PromptTemplate(
            template="""Given the following information about the column '{col}':
            
            Semantic context: {semantic_context}
            Statistical profile: {stats}
            
            {format_instructions}
            Provide a semantic review for this column.""",
            input_variables=["col", "semantic_context", "stats"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # print("Example Prompt:")
        formatted_prompt = prompt.format(
            col="email",
            semantic_context=semantic_context.column_summary.get("email", ""),
            stats=statistical_profile.columns["email"].model_dump()
            ) 
        

        # print(formatted_prompt)

        chain = prompt | self.llm 
        retry_chain = RunnableParallel(
        completion=chain, prompt_value=prompt
        ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(completion=x["completion"].content, prompt_value=x["prompt_value"]))


        review = {}
        col = "email"

        result = chain.invoke({
        "col": col,
        "semantic_context":semantic_context.column_summary.get("email", ""),
        "stats":statistical_profile.columns["email"].model_dump()
        }) 
        review[col] = result


        return review

# Example usage
if __name__ == "__main__":
    # Set up the LLM (you'll need to set your OpenAI API key in your environment variables)
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    
    profiler = SemanticProfiler(llm)

    # Example data
    table_name = "employees"
    semantic_context = SemanticContext(
        table_summary="Employee information table",
        column_summary={
            "employee_id": "Unique identifier for each employee",
            "name": "Full name of the employee",
            "email": "Employee's work email address",
            "age": "Age of the employee in years",
            "department_id": "ID of the department the employee works in"
        }
    )
    statistical_profile = StatisticalProfile(
        columns={
            "employee_id": ColumnStats(type="BIGINT", has_duplicates=False, null_percentage=0.0, min=1, max=1000),
            "name": ColumnStats(type="TEXT", has_duplicates=False, null_percentage=0.0, pattern="John Doe"),
            "email": ColumnStats(type="TEXT", has_duplicates=False, null_percentage=0.0, pattern="john.doe@example.com"),
            "age": ColumnStats(type="BIGINT", has_duplicates=True, null_percentage=0.0, min=18, max=65, avg=35),
            "department_id": ColumnStats(type="BIGINT", has_duplicates=True, null_percentage=0.0, min=1, max=10)
        }
    )

    semantic_review = profiler.semantic_review(table_name, semantic_context, statistical_profile)

    print(semantic_review)


    # print("Semantic Review:")
    # for col, review in semantic_review.items():
    #     print(f"\n{col}:")
    #     print(review.model_dump_json(indent=2))

    
