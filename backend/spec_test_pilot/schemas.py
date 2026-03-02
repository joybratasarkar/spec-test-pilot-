"""
Pydantic schemas for SpecTestPilot strict JSON output contract.

All agent output MUST validate against these schemas.
"""

from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class EndpointInfo(BaseModel):
    """Detected endpoint from OpenAPI spec."""
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str = Field(..., description="API path, e.g., /users/{id}")
    operation_id: str = Field(default="unknown", description="Operation ID from spec")


class AuthInfo(BaseModel):
    """Authentication information from spec."""
    type: Literal["none", "apiKey", "bearer", "oauth2", "unknown"] = "unknown"
    details: str = "unknown"


class SpecSummary(BaseModel):
    """Summary of the parsed OpenAPI specification."""
    title: str = "unknown"
    version: str = "unknown"
    base_url: str = "unknown"
    auth: AuthInfo = Field(default_factory=AuthInfo)
    endpoints_detected: List[EndpointInfo] = Field(default_factory=list)


class MemoryExcerpt(BaseModel):
    """Excerpt from GAM memory system."""
    source: Literal["convention", "existing_tests", "runbook", "validator", "memo"]
    excerpt: str = Field(..., max_length=500, description="Max 2 lines of content")


class DeepResearch(BaseModel):
    """GAM-style deep research output."""
    plan: List[str] = Field(default_factory=list, description="Research plan steps")
    memory_excerpts: List[MemoryExcerpt] = Field(
        default_factory=list,
        max_length=5,
        description="Max 5 excerpts, each max 2 lines"
    )
    reflection: str = Field(default="", description="Reflection on research quality")


class RequestSpec(BaseModel):
    """Test case request specification."""
    headers: Dict[str, str] = Field(default_factory=dict)
    path_params: Dict[str, str] = Field(default_factory=dict)
    query_params: Dict[str, str] = Field(default_factory=dict)
    body: Dict[str, Any] = Field(default_factory=dict)


class Assertion(BaseModel):
    """Test assertion specification."""
    type: Literal["status_code", "schema", "field", "header", "response_time"]
    expected: Union[int, str, Dict[str, Any]] = Field(..., description="Expected value")
    path: Optional[str] = Field(default=None, description="JSONPath for field assertions")
    rule: Optional[str] = Field(default=None, description="Assertion rule, e.g., exists, equals")


class DataVariant(BaseModel):
    """Data variant for parameterized testing."""
    description: str = "optional"
    overrides: Dict[str, Any] = Field(default_factory=dict)


class TestEndpoint(BaseModel):
    """Endpoint reference in a test case."""
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str


class TestCase(BaseModel):
    """Individual test case specification."""
    test_id: str = Field(..., pattern=r"^T\d{3,}$", description="Test ID, e.g., T001")
    name: str = Field(..., description="<METHOD> <PATH> <case>")
    endpoint: TestEndpoint
    objective: str = Field(..., description="What this test validates")
    preconditions: List[str] = Field(default_factory=list)
    request: RequestSpec = Field(default_factory=RequestSpec)
    assertions: List[Assertion] = Field(default_factory=list, min_length=1)
    data_variants: List[DataVariant] = Field(default_factory=list)
    notes: str = ""

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        """Ensure name follows <METHOD> <PATH> <case> format."""
        parts = v.split()
        if len(parts) < 3:
            # Allow but don't enforce strict format
            pass
        return v


class CoverageChecklist(BaseModel):
    """Coverage checklist for test suite completeness."""
    happy_paths: Literal["true", "false", "unknown"] = "unknown"
    validation_negative: Literal["true", "false", "unknown"] = "unknown"
    auth_negative: Literal["true", "false", "unknown"] = "unknown"
    error_contract: Literal["true", "false", "unknown"] = "unknown"
    idempotency: Literal["true", "false", "unknown"] = "unknown"
    pagination_filtering: Literal["true", "false", "unknown"] = "unknown"
    rate_limit: Literal["true", "false", "unknown"] = "unknown"


class SpecTestPilotOutput(BaseModel):
    """
    Complete output schema for SpecTestPilot agent.
    
    This is the strict contract that all agent output MUST match.
    """
    spec_summary: SpecSummary = Field(default_factory=SpecSummary)
    deep_research: DeepResearch = Field(default_factory=DeepResearch)
    test_suite: List[TestCase] = Field(default_factory=list)
    coverage_checklist: CoverageChecklist = Field(default_factory=CoverageChecklist)
    missing_info: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_no_invented_endpoints(self) -> "SpecTestPilotOutput":
        """Ensure all test endpoints exist in detected endpoints."""
        detected = {
            (e.method, e.path) for e in self.spec_summary.endpoints_detected
        }
        for test in self.test_suite:
            test_endpoint = (test.endpoint.method, test.endpoint.path)
            if test_endpoint not in detected:
                raise ValueError(
                    f"Test {test.test_id} references endpoint {test_endpoint} "
                    f"not in detected endpoints: {detected}"
                )
        return self

    @classmethod
    def create_empty_spec_output(cls) -> "SpecTestPilotOutput":
        """
        Create output for missing/empty/unparseable spec.
        
        Per contract:
        - endpoints_detected = []
        - test_suite = []
        - coverage_checklist fields = "unknown"
        - missing_info MUST include required items
        """
        return cls(
            spec_summary=SpecSummary(
                title="unknown",
                version="unknown",
                base_url="unknown",
                auth=AuthInfo(type="unknown", details="unknown"),
                endpoints_detected=[]
            ),
            deep_research=DeepResearch(
                plan=["Unable to plan: spec missing or unparseable"],
                memory_excerpts=[],
                reflection="No spec available for research"
            ),
            test_suite=[],
            coverage_checklist=CoverageChecklist(
                happy_paths="unknown",
                validation_negative="unknown",
                auth_negative="unknown",
                error_contract="unknown",
                idempotency="unknown",
                pagination_filtering="unknown",
                rate_limit="unknown"
            ),
            missing_info=[
                "API spec content (OpenAPI/Swagger YAML/JSON)",
                "auth method details (if any)",
                "environment/base URL + required headers (if any)"
            ]
        )


# Type aliases for convenience
EndpointTuple = tuple[str, str]  # (method, path)


def validate_output(output_dict: Dict[str, Any]) -> SpecTestPilotOutput:
    """
    Validate output dictionary against schema.
    
    Args:
        output_dict: Raw output dictionary
        
    Returns:
        Validated SpecTestPilotOutput
        
    Raises:
        ValidationError: If output doesn't match schema
    """
    return SpecTestPilotOutput.model_validate(output_dict)


def output_to_json(output: SpecTestPilotOutput) -> str:
    """
    Convert output to JSON string.
    
    Args:
        output: Validated output
        
    Returns:
        JSON string
    """
    return output.model_dump_json(indent=2)
