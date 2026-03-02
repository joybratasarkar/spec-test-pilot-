"""
LangGraph state machine for SpecTestPilot agent.

Implements the following nodes:
- parse_spec
- detect_endpoints
- deep_research_plan
- deep_research_search
- deep_research_integrate
- deep_research_reflect (loop controller)
- generate_tests
- finalize_and_validate
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal
from dataclasses import dataclass, field
import json
import random

from langgraph.graph import StateGraph, END
from pydantic import ValidationError

from spec_test_pilot.schemas import (
    SpecTestPilotOutput,
    SpecSummary,
    DeepResearch,
    TestCase,
    CoverageChecklist,
    EndpointInfo,
    AuthInfo,
    MemoryExcerpt,
    TestEndpoint,
    RequestSpec,
    Assertion,
)
from spec_test_pilot.openapi_parse import (
    parse_openapi_spec,
    ParsedSpec,
    ParsedEndpoint,
    get_path_parameters,
)
from spec_test_pilot.memory.gam import GAMMemorySystem, ResearchResult


# State definition for LangGraph
class AgentState(TypedDict):
    """State model for the LangGraph agent."""
    # Input
    spec_text: str
    
    # Parsing
    parsed_spec: Optional[Dict[str, Any]]
    endpoints: List[Dict[str, Any]]
    parse_errors: List[str]
    
    # Research
    research_plan: List[str]
    retrieved_pages: List[Dict[str, Any]]
    memory_excerpts: List[Dict[str, str]]
    reflection_count: int
    reflection_text: str
    should_continue_research: bool
    
    # Generation
    draft_output: Optional[Dict[str, Any]]
    validated_output: Optional[Dict[str, Any]]
    missing_info: List[str]
    
    # Reward (for RL)
    reward: float
    intermediate_rewards: Dict[str, float]
    
    # Metadata
    run_id: str
    verbose: bool


def create_initial_state(spec_text: str, run_id: str = "", verbose: bool = False) -> AgentState:
    """Create initial state for the agent."""
    return AgentState(
        spec_text=spec_text,
        parsed_spec=None,
        endpoints=[],
        parse_errors=[],
        research_plan=[],
        retrieved_pages=[],
        memory_excerpts=[],
        reflection_count=0,
        reflection_text="",
        should_continue_research=True,
        draft_output=None,
        validated_output=None,
        missing_info=[],
        reward=0.0,
        intermediate_rewards={},
        run_id=run_id or f"run_{random.randint(10000, 99999)}",
        verbose=verbose
    )


# Node implementations

def parse_spec_node(state: AgentState) -> AgentState:
    """Parse the OpenAPI specification."""
    spec_text = state["spec_text"]
    
    if not spec_text or not spec_text.strip():
        return {
            **state,
            "parsed_spec": None,
            "parse_errors": ["Empty or missing spec content"],
            "intermediate_rewards": {
                **state.get("intermediate_rewards", {}),
                "parse_spec": 0.0
            }
        }
    
    parsed = parse_openapi_spec(spec_text)
    
    parsed_dict = {
        "title": parsed.title,
        "version": parsed.version,
        "base_url": parsed.base_url,
        "description": parsed.description,
        "auth": {
            "type": parsed.auth.type,
            "details": parsed.auth.details
        },
        "schemas": parsed.schemas,
        "is_valid": parsed.is_valid
    }
    
    # Calculate intermediate reward for parsing
    parse_reward = 0.5 if parsed.is_valid else 0.1
    
    return {
        **state,
        "parsed_spec": parsed_dict,
        "parse_errors": parsed.parse_errors,
        "intermediate_rewards": {
            **state.get("intermediate_rewards", {}),
            "parse_spec": parse_reward
        }
    }


def detect_endpoints_node(state: AgentState) -> AgentState:
    """Detect and extract endpoints from parsed spec."""
    spec_text = state["spec_text"]
    
    if not spec_text:
        return {
            **state,
            "endpoints": [],
            "intermediate_rewards": {
                **state.get("intermediate_rewards", {}),
                "detect_endpoints": 0.0
            }
        }
    
    parsed = parse_openapi_spec(spec_text)
    
    endpoints = []
    for ep in parsed.endpoints:
        endpoint_dict = {
            "method": ep.method,
            "path": ep.path,
            "operation_id": ep.operation_id,
            "summary": ep.summary,
            "description": ep.description,
            "tags": ep.tags,
            "parameters": [
                {
                    "name": p.name,
                    "location": p.location,
                    "required": p.required,
                    "schema_type": p.schema_type,
                    "description": p.description,
                    "example": p.example
                }
                for p in ep.parameters
            ],
            "request_body": {
                "content_type": ep.request_body.content_type,
                "required": ep.request_body.required,
                "schema": ep.request_body.schema
            } if ep.request_body else None,
            "responses": [
                {
                    "status_code": r.status_code,
                    "description": r.description,
                    "schema": r.schema
                }
                for r in ep.responses
            ],
            "security": ep.security
        }
        endpoints.append(endpoint_dict)
    
    # Intermediate reward based on endpoints found
    endpoint_reward = min(1.0, len(endpoints) * 0.1) if endpoints else 0.0
    
    return {
        **state,
        "endpoints": endpoints,
        "intermediate_rewards": {
            **state.get("intermediate_rewards", {}),
            "detect_endpoints": endpoint_reward
        }
    }


def deep_research_plan_node(state: AgentState) -> AgentState:
    """Create research plan based on spec context."""
    parsed_spec = state.get("parsed_spec", {})
    endpoints = state.get("endpoints", [])
    
    # Build context for research
    context = {
        "spec_title": parsed_spec.get("title", "unknown") if parsed_spec else "unknown",
        "auth_type": parsed_spec.get("auth", {}).get("type", "unknown") if parsed_spec else "unknown",
        "endpoints": endpoints
    }
    
    # Create GAM memory system and get plan
    memory = GAMMemorySystem(use_vector_search=False)  # Disable vector for speed
    plan = memory.researcher.plan(context)
    
    return {
        **state,
        "research_plan": plan,
        "intermediate_rewards": {
            **state.get("intermediate_rewards", {}),
            "research_plan": 0.3 if plan else 0.0
        }
    }


def deep_research_search_node(state: AgentState) -> AgentState:
    """Execute search based on research plan."""
    plan = state.get("research_plan", [])
    
    memory = GAMMemorySystem(use_vector_search=False)
    results = memory.researcher.search(plan)
    
    retrieved = [
        {
            "id": page.id,
            "title": page.title,
            "tags": page.tags,
            "content": page.content,
            "source": page.source,
            "score": score
        }
        for page, score in results
    ]
    
    return {
        **state,
        "retrieved_pages": retrieved,
        "intermediate_rewards": {
            **state.get("intermediate_rewards", {}),
            "research_search": min(1.0, len(retrieved) * 0.1)
        }
    }


def deep_research_integrate_node(state: AgentState) -> AgentState:
    """Integrate search results into memory excerpts."""
    retrieved = state.get("retrieved_pages", [])
    
    excerpts = []
    for page in retrieved[:5]:  # Max 5 excerpts
        content = page.get("content", "")
        if len(content) > 200:
            content = content[:200] + "..."
        
        excerpts.append({
            "source": page.get("source", "memo"),
            "excerpt": content
        })
    
    return {
        **state,
        "memory_excerpts": excerpts,
        "intermediate_rewards": {
            **state.get("intermediate_rewards", {}),
            "research_integrate": min(1.0, len(excerpts) * 0.2)
        }
    }


def deep_research_reflect_node(state: AgentState) -> AgentState:
    """Reflect on research and decide if more iterations needed."""
    reflection_count = state.get("reflection_count", 0) + 1
    excerpts = state.get("memory_excerpts", [])
    parsed_spec = state.get("parsed_spec", {})
    
    # Build context
    context = {
        "spec_title": parsed_spec.get("title", "unknown") if parsed_spec else "unknown",
        "auth_type": parsed_spec.get("auth", {}).get("type", "unknown") if parsed_spec else "unknown",
        "endpoints": state.get("endpoints", [])
    }
    
    memory = GAMMemorySystem(use_vector_search=False)
    reflection, should_continue = memory.researcher.reflect(
        context, excerpts, reflection_count
    )
    
    # Max 2 iterations
    should_continue = should_continue and reflection_count < 2
    
    return {
        **state,
        "reflection_count": reflection_count,
        "reflection_text": reflection,
        "should_continue_research": should_continue,
        "intermediate_rewards": {
            **state.get("intermediate_rewards", {}),
            "research_reflect": 0.5 if not should_continue else 0.2
        }
    }


def generate_tests_node(state: AgentState) -> AgentState:
    """Generate test cases based on endpoints and research."""
    endpoints = state.get("endpoints", [])
    parsed_spec = state.get("parsed_spec", {})
    memory_excerpts = state.get("memory_excerpts", [])
    
    if not endpoints:
        # No endpoints - create empty output
        return {
            **state,
            "draft_output": SpecTestPilotOutput.create_empty_spec_output().model_dump(),
            "intermediate_rewards": {
                **state.get("intermediate_rewards", {}),
                "generate_tests": 0.0
            }
        }
    
    # Build spec summary
    auth_info = parsed_spec.get("auth", {}) if parsed_spec else {}
    spec_summary = SpecSummary(
        title=parsed_spec.get("title", "unknown") if parsed_spec else "unknown",
        version=parsed_spec.get("version", "unknown") if parsed_spec else "unknown",
        base_url=parsed_spec.get("base_url", "unknown") if parsed_spec else "unknown",
        auth=AuthInfo(
            type=auth_info.get("type", "unknown"),
            details=auth_info.get("details", "unknown")
        ),
        endpoints_detected=[
            EndpointInfo(
                method=ep["method"],
                path=ep["path"],
                operation_id=ep.get("operation_id", "unknown")
            )
            for ep in endpoints
        ]
    )
    
    # Build deep research
    deep_research = DeepResearch(
        plan=state.get("research_plan", []),
        memory_excerpts=[
            MemoryExcerpt(
                source=exc.get("source", "memo"),
                excerpt=exc.get("excerpt", "")
            )
            for exc in memory_excerpts
        ],
        reflection=state.get("reflection_text", "")
    )
    
    # Generate test cases
    test_suite = []
    test_id_counter = 1
    
    for ep in endpoints:
        method = ep["method"]
        path = ep["path"]
        params = ep.get("parameters", [])
        request_body = ep.get("request_body")
        responses = ep.get("responses", [])
        
        # Get path parameters
        path_params = get_path_parameters(path)
        
        # 1. Happy path test
        happy_test = _create_happy_path_test(
            test_id_counter, method, path, params, path_params, 
            request_body, responses, auth_info
        )
        test_suite.append(happy_test)
        test_id_counter += 1
        
        # 2. Validation negative test (for POST/PUT/PATCH with body)
        if method in ["POST", "PUT", "PATCH"] and request_body:
            validation_test = _create_validation_negative_test(
                test_id_counter, method, path, path_params, 
                request_body, auth_info
            )
            test_suite.append(validation_test)
            test_id_counter += 1
        
        # 3. Auth negative test (if auth required)
        if auth_info.get("type") not in ["none", "unknown"]:
            auth_test = _create_auth_negative_test(
                test_id_counter, method, path, path_params
            )
            test_suite.append(auth_test)
            test_id_counter += 1
        
        # 4. Not found test (for endpoints with path params)
        if path_params:
            not_found_test = _create_not_found_test(
                test_id_counter, method, path, path_params, auth_info
            )
            test_suite.append(not_found_test)
            test_id_counter += 1
    
    # Build coverage checklist
    has_auth = auth_info.get("type") not in ["none", "unknown"]
    coverage = CoverageChecklist(
        happy_paths="true",
        validation_negative="true" if any(
            ep["method"] in ["POST", "PUT", "PATCH"] and ep.get("request_body")
            for ep in endpoints
        ) else "false",
        auth_negative="true" if has_auth else "false",
        error_contract="true",
        idempotency="unknown",  # Would need more analysis
        pagination_filtering="unknown",
        rate_limit="unknown"
    )
    
    # Build output
    output = SpecTestPilotOutput(
        spec_summary=spec_summary,
        deep_research=deep_research,
        test_suite=test_suite,
        coverage_checklist=coverage,
        missing_info=[]
    )
    
    # Calculate reward
    tests_per_endpoint = len(test_suite) / len(endpoints) if endpoints else 0
    generate_reward = min(1.0, tests_per_endpoint * 0.3)
    
    return {
        **state,
        "draft_output": output.model_dump(),
        "intermediate_rewards": {
            **state.get("intermediate_rewards", {}),
            "generate_tests": generate_reward
        }
    }


def finalize_and_validate_node(state: AgentState) -> AgentState:
    """Validate output against Pydantic schema and finalize."""
    draft = state.get("draft_output")
    
    if not draft:
        # Create empty spec output
        output = SpecTestPilotOutput.create_empty_spec_output()
        return {
            **state,
            "validated_output": output.model_dump(),
            "missing_info": output.missing_info,
            "reward": 0.0,
            "intermediate_rewards": {
                **state.get("intermediate_rewards", {}),
                "finalize": 0.0
            }
        }
    
    try:
        # Validate with Pydantic
        validated = SpecTestPilotOutput.model_validate(draft)
        
        # Calculate final reward
        intermediate = state.get("intermediate_rewards", {})
        total_reward = sum(intermediate.values()) / max(len(intermediate), 1)
        
        return {
            **state,
            "validated_output": validated.model_dump(),
            "missing_info": validated.missing_info,
            "reward": total_reward,
            "intermediate_rewards": {
                **intermediate,
                "finalize": 1.0
            }
        }
    except ValidationError as e:
        # Validation failed - return empty output
        output = SpecTestPilotOutput.create_empty_spec_output()
        output.missing_info.append(f"Validation error: {str(e)[:100]}")
        
        return {
            **state,
            "validated_output": output.model_dump(),
            "missing_info": output.missing_info,
            "reward": 0.0,
            "intermediate_rewards": {
                **state.get("intermediate_rewards", {}),
                "finalize": 0.0
            }
        }


# Helper functions for test generation

def _create_happy_path_test(
    test_id: int,
    method: str,
    path: str,
    params: List[Dict],
    path_params: List[str],
    request_body: Optional[Dict],
    responses: List[Dict],
    auth_info: Dict
) -> TestCase:
    """Create a happy path test case."""
    # Build request
    headers = {"Content-Type": "application/json"}
    if auth_info.get("type") == "bearer":
        headers["Authorization"] = "Bearer <token>"
    elif auth_info.get("type") == "apiKey":
        headers["X-API-Key"] = "<api_key>"
    
    path_param_values = {p: f"<{p}>" for p in path_params}
    
    query_params = {}
    for p in params:
        if p.get("location") == "query":
            query_params[p["name"]] = f"<{p['name']}>"
    
    body = {}
    if request_body and request_body.get("schema"):
        schema = request_body["schema"]
        if "properties" in schema:
            for prop_name in list(schema["properties"].keys())[:3]:
                body[prop_name] = f"<{prop_name}>"
    
    # Determine expected status
    expected_status = 200
    for r in responses:
        if 200 <= r.get("status_code", 0) < 300:
            expected_status = r["status_code"]
            break
    
    return TestCase(
        test_id=f"T{test_id:03d}",
        name=f"{method} {path} happy path",
        endpoint=TestEndpoint(method=method, path=path),
        objective=f"Verify {method} {path} returns success with valid inputs",
        preconditions=["Valid authentication" if auth_info.get("type") not in ["none", "unknown"] else "None"],
        request=RequestSpec(
            headers=headers,
            path_params=path_param_values,
            query_params=query_params,
            body=body
        ),
        assertions=[
            Assertion(type="status_code", expected=expected_status),
            Assertion(type="schema", expected="response_schema")
        ],
        data_variants=[],
        notes=""
    )


def _create_validation_negative_test(
    test_id: int,
    method: str,
    path: str,
    path_params: List[str],
    request_body: Dict,
    auth_info: Dict
) -> TestCase:
    """Create a validation negative test case."""
    headers = {"Content-Type": "application/json"}
    if auth_info.get("type") == "bearer":
        headers["Authorization"] = "Bearer <token>"
    elif auth_info.get("type") == "apiKey":
        headers["X-API-Key"] = "<api_key>"
    
    path_param_values = {p: f"<{p}>" for p in path_params}
    
    return TestCase(
        test_id=f"T{test_id:03d}",
        name=f"{method} {path} missing required field",
        endpoint=TestEndpoint(method=method, path=path),
        objective=f"Verify {method} {path} returns 400 when required field is missing",
        preconditions=["Valid authentication" if auth_info.get("type") not in ["none", "unknown"] else "None"],
        request=RequestSpec(
            headers=headers,
            path_params=path_param_values,
            query_params={},
            body={}  # Empty body to trigger validation error
        ),
        assertions=[
            Assertion(type="status_code", expected=400),
            Assertion(type="field", path="$.error", rule="exists", expected="error message")
        ],
        data_variants=[],
        notes="Tests missing required field validation"
    )


def _create_auth_negative_test(
    test_id: int,
    method: str,
    path: str,
    path_params: List[str]
) -> TestCase:
    """Create an auth negative test case."""
    path_param_values = {p: f"<{p}>" for p in path_params}
    
    return TestCase(
        test_id=f"T{test_id:03d}",
        name=f"{method} {path} missing auth",
        endpoint=TestEndpoint(method=method, path=path),
        objective=f"Verify {method} {path} returns 401 without authentication",
        preconditions=["No authentication provided"],
        request=RequestSpec(
            headers={"Content-Type": "application/json"},
            path_params=path_param_values,
            query_params={},
            body={}
        ),
        assertions=[
            Assertion(type="status_code", expected=401)
        ],
        data_variants=[],
        notes="Tests authentication requirement"
    )


def _create_not_found_test(
    test_id: int,
    method: str,
    path: str,
    path_params: List[str],
    auth_info: Dict
) -> TestCase:
    """Create a not found test case."""
    headers = {"Content-Type": "application/json"}
    if auth_info.get("type") == "bearer":
        headers["Authorization"] = "Bearer <token>"
    elif auth_info.get("type") == "apiKey":
        headers["X-API-Key"] = "<api_key>"
    
    # Use invalid IDs
    path_param_values = {p: "nonexistent_id_12345" for p in path_params}
    
    return TestCase(
        test_id=f"T{test_id:03d}",
        name=f"{method} {path} not found",
        endpoint=TestEndpoint(method=method, path=path),
        objective=f"Verify {method} {path} returns 404 for non-existent resource",
        preconditions=["Valid authentication" if auth_info.get("type") not in ["none", "unknown"] else "None"],
        request=RequestSpec(
            headers=headers,
            path_params=path_param_values,
            query_params={},
            body={}
        ),
        assertions=[
            Assertion(type="status_code", expected=404)
        ],
        data_variants=[],
        notes="Tests not found handling"
    )


# Router function for research loop
def should_continue_research(state: AgentState) -> Literal["deep_research_plan", "generate_tests"]:
    """Decide whether to continue research or proceed to test generation."""
    if state.get("should_continue_research", False) and state.get("reflection_count", 0) < 2:
        return "deep_research_plan"
    return "generate_tests"


def build_graph() -> StateGraph:
    """Build the LangGraph state machine."""
    # Create graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("parse_spec", parse_spec_node)
    graph.add_node("detect_endpoints", detect_endpoints_node)
    graph.add_node("deep_research_plan", deep_research_plan_node)
    graph.add_node("deep_research_search", deep_research_search_node)
    graph.add_node("deep_research_integrate", deep_research_integrate_node)
    graph.add_node("deep_research_reflect", deep_research_reflect_node)
    graph.add_node("generate_tests", generate_tests_node)
    graph.add_node("finalize_and_validate", finalize_and_validate_node)
    
    # Add edges
    graph.set_entry_point("parse_spec")
    graph.add_edge("parse_spec", "detect_endpoints")
    graph.add_edge("detect_endpoints", "deep_research_plan")
    graph.add_edge("deep_research_plan", "deep_research_search")
    graph.add_edge("deep_research_search", "deep_research_integrate")
    graph.add_edge("deep_research_integrate", "deep_research_reflect")
    
    # Conditional edge for research loop
    graph.add_conditional_edges(
        "deep_research_reflect",
        should_continue_research,
        {
            "deep_research_plan": "deep_research_plan",
            "generate_tests": "generate_tests"
        }
    )
    
    graph.add_edge("generate_tests", "finalize_and_validate")
    graph.add_edge("finalize_and_validate", END)
    
    return graph


def compile_graph():
    """Compile the graph for execution."""
    graph = build_graph()
    return graph.compile()


def run_agent(spec_text: str, run_id: str = "", verbose: bool = False) -> Dict[str, Any]:
    """
    Run the SpecTestPilot agent on an OpenAPI spec.
    
    Args:
        spec_text: OpenAPI spec content (YAML or JSON)
        run_id: Optional run identifier
        verbose: Whether to print verbose output
        
    Returns:
        Dict containing validated_output, reward, and intermediate_rewards
    """
    # Create initial state
    initial_state = create_initial_state(spec_text, run_id, verbose)
    
    # Compile and run graph
    app = compile_graph()
    final_state = app.invoke(initial_state)
    
    return {
        "output": final_state.get("validated_output"),
        "reward": final_state.get("reward", 0.0),
        "intermediate_rewards": final_state.get("intermediate_rewards", {}),
        "run_id": final_state.get("run_id", "")
    }


# Backward-compatible aliases for older integrations.
parse_spec = parse_spec_node
detect_endpoints = detect_endpoints_node
deep_research_plan = deep_research_plan_node
deep_research_search = deep_research_search_node
deep_research_integrate = deep_research_integrate_node
deep_research_reflect = deep_research_reflect_node
generate_tests = generate_tests_node
finalize_and_validate_json = finalize_and_validate_node
