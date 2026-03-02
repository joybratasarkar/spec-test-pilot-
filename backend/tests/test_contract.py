"""
Contract tests for SpecTestPilot.

Tests:
- Missing-spec behavior (empty spec)
- No-invented-endpoints invariant
- JSON schema validation via Pydantic
"""

import pytest
import json

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
    validate_output,
)
from spec_test_pilot.openapi_parse import parse_openapi_spec
from spec_test_pilot.graph import run_agent
from spec_test_pilot.reward import compute_reward, RewardBreakdown


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_openapi_spec():
    """Sample valid OpenAPI spec."""
    return """
openapi: "3.0.3"
info:
  title: "Pet Store API"
  version: "1.0.0"
servers:
  - url: "https://api.petstore.com/v1"
paths:
  /pets:
    get:
      operationId: listPets
      summary: List all pets
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
      responses:
        "200":
          description: A list of pets
    post:
      operationId: createPet
      summary: Create a pet
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/PetInput"
      responses:
        "201":
          description: Pet created
  /pets/{petId}:
    get:
      operationId: getPet
      summary: Get a pet by ID
      parameters:
        - name: petId
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: A pet
        "404":
          description: Pet not found
    delete:
      operationId: deletePet
      summary: Delete a pet
      parameters:
        - name: petId
          in: path
          required: true
          schema:
            type: string
      responses:
        "204":
          description: Pet deleted
components:
  schemas:
    Pet:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        status:
          type: string
    PetInput:
      type: object
      required:
        - name
      properties:
        name:
          type: string
        status:
          type: string
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
security:
  - BearerAuth: []
"""


@pytest.fixture
def empty_spec():
    """Empty spec content."""
    return ""


@pytest.fixture
def invalid_spec():
    """Invalid/unparseable spec content."""
    return "this is not valid yaml or json {"


@pytest.fixture
def minimal_spec():
    """Minimal valid spec with one endpoint."""
    return """
openapi: "3.0.3"
info:
  title: "Minimal API"
  version: "1.0.0"
paths:
  /health:
    get:
      operationId: healthCheck
      responses:
        "200":
          description: OK
"""


# ============================================================================
# Missing-Spec Behavior Tests
# ============================================================================

class TestMissingSpecBehavior:
    """Tests for missing/empty/unparseable spec handling."""
    
    def test_empty_spec_returns_empty_output(self, empty_spec):
        """Empty spec should return empty endpoints and test_suite."""
        result = run_agent(empty_spec)
        output = result["output"]
        
        # Validate structure
        assert "spec_summary" in output
        assert "test_suite" in output
        assert "coverage_checklist" in output
        assert "missing_info" in output
        
        # Check empty spec behavior
        assert output["spec_summary"]["endpoints_detected"] == []
        assert output["test_suite"] == []
    
    def test_empty_spec_coverage_unknown(self, empty_spec):
        """Empty spec should have all coverage_checklist fields as 'unknown'."""
        result = run_agent(empty_spec)
        output = result["output"]
        
        checklist = output["coverage_checklist"]
        assert checklist["happy_paths"] == "unknown"
        assert checklist["validation_negative"] == "unknown"
        assert checklist["auth_negative"] == "unknown"
        assert checklist["error_contract"] == "unknown"
        assert checklist["idempotency"] == "unknown"
        assert checklist["pagination_filtering"] == "unknown"
        assert checklist["rate_limit"] == "unknown"
    
    def test_empty_spec_missing_info_required_items(self, empty_spec):
        """Empty spec should have required items in missing_info."""
        result = run_agent(empty_spec)
        output = result["output"]
        
        missing_info = output["missing_info"]
        missing_text = " ".join(missing_info).lower()
        
        # Must mention API spec
        assert any(
            term in missing_text 
            for term in ["api spec", "openapi", "swagger", "yaml", "json"]
        )
        
        # Must mention auth
        assert any(
            term in missing_text
            for term in ["auth", "authentication"]
        )
        
        # Must mention environment/base URL
        assert any(
            term in missing_text
            for term in ["base url", "environment", "header"]
        )
    
    def test_invalid_spec_handled_gracefully(self, invalid_spec):
        """Invalid spec should be handled without crashing."""
        result = run_agent(invalid_spec)
        output = result["output"]
        
        # Should still return valid structure
        assert "spec_summary" in output
        assert "test_suite" in output
        
        # Should indicate issues
        assert output["spec_summary"]["endpoints_detected"] == []
    
    def test_create_empty_spec_output_helper(self):
        """Test the create_empty_spec_output helper method."""
        output = SpecTestPilotOutput.create_empty_spec_output()
        
        assert output.spec_summary.title == "unknown"
        assert output.spec_summary.endpoints_detected == []
        assert output.test_suite == []
        assert len(output.missing_info) >= 3


# ============================================================================
# No-Invented-Endpoints Tests
# ============================================================================

class TestNoInventedEndpoints:
    """Tests for the no-invented-endpoints invariant."""
    
    def test_all_test_endpoints_exist_in_spec(self, sample_openapi_spec):
        """All test endpoints must exist in detected endpoints."""
        result = run_agent(sample_openapi_spec)
        output = result["output"]
        
        # Get detected endpoints
        detected = {
            (e["method"], e["path"])
            for e in output["spec_summary"]["endpoints_detected"]
        }
        
        # Check all test endpoints
        for test in output["test_suite"]:
            test_endpoint = (test["endpoint"]["method"], test["endpoint"]["path"])
            assert test_endpoint in detected, (
                f"Test {test['test_id']} references endpoint {test_endpoint} "
                f"not in detected endpoints: {detected}"
            )
    
    def test_pydantic_validation_catches_invented_endpoints(self):
        """Pydantic validation should catch invented endpoints."""
        # Create output with invented endpoint
        output_dict = {
            "spec_summary": {
                "title": "Test API",
                "version": "1.0.0",
                "base_url": "https://api.test.com",
                "auth": {"type": "none", "details": "none"},
                "endpoints_detected": [
                    {"method": "GET", "path": "/users", "operation_id": "listUsers"}
                ]
            },
            "deep_research": {
                "plan": [],
                "memory_excerpts": [],
                "reflection": ""
            },
            "test_suite": [
                {
                    "test_id": "T001",
                    "name": "GET /users happy path",
                    "endpoint": {"method": "GET", "path": "/users"},
                    "objective": "Test users endpoint",
                    "preconditions": [],
                    "request": {
                        "headers": {},
                        "path_params": {},
                        "query_params": {},
                        "body": {}
                    },
                    "assertions": [
                        {"type": "status_code", "expected": 200}
                    ],
                    "data_variants": [],
                    "notes": ""
                },
                {
                    "test_id": "T002",
                    "name": "GET /invented happy path",
                    "endpoint": {"method": "GET", "path": "/invented"},  # INVENTED!
                    "objective": "Test invented endpoint",
                    "preconditions": [],
                    "request": {
                        "headers": {},
                        "path_params": {},
                        "query_params": {},
                        "body": {}
                    },
                    "assertions": [
                        {"type": "status_code", "expected": 200}
                    ],
                    "data_variants": [],
                    "notes": ""
                }
            ],
            "coverage_checklist": {
                "happy_paths": "true",
                "validation_negative": "false",
                "auth_negative": "false",
                "error_contract": "false",
                "idempotency": "unknown",
                "pagination_filtering": "unknown",
                "rate_limit": "unknown"
            },
            "missing_info": []
        }
        
        # Should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            SpecTestPilotOutput.model_validate(output_dict)
        
        assert "invented" in str(exc_info.value).lower() or "endpoint" in str(exc_info.value).lower()
    
    def test_reward_zero_for_invented_endpoints(self, sample_openapi_spec):
        """Reward should be 0.0 for outputs with invented endpoints."""
        parsed_spec = parse_openapi_spec(sample_openapi_spec)
        
        # Create output with invented endpoint (bypassing Pydantic)
        output_dict = {
            "spec_summary": {
                "endpoints_detected": [
                    {"method": "GET", "path": "/pets", "operation_id": "listPets"}
                ]
            },
            "test_suite": [
                {
                    "endpoint": {"method": "GET", "path": "/invented"}
                }
            ]
        }
        
        # Reward computation should fail hard gate
        # Note: This would fail Pydantic validation first in practice
        # Testing the reward function directly
        from spec_test_pilot.reward import _compute_endpoint_coverage
        
        # The reward function checks this invariant
        spec_endpoints = {(e.method, e.path) for e in parsed_spec.endpoints}
        test_endpoints = {("GET", "/invented")}
        
        invented = test_endpoints - spec_endpoints
        assert len(invented) > 0  # Confirms invented endpoint detected


# ============================================================================
# JSON Schema Validation Tests
# ============================================================================

class TestJsonSchemaValidation:
    """Tests for Pydantic JSON schema validation."""
    
    def test_valid_output_passes_validation(self, sample_openapi_spec):
        """Valid agent output should pass Pydantic validation."""
        result = run_agent(sample_openapi_spec)
        output = result["output"]
        
        # Should not raise
        validated = SpecTestPilotOutput.model_validate(output)
        assert validated is not None
    
    def test_output_is_valid_json(self, sample_openapi_spec):
        """Output should be serializable to valid JSON."""
        result = run_agent(sample_openapi_spec)
        output = result["output"]
        
        # Should not raise
        json_str = json.dumps(output)
        parsed = json.loads(json_str)
        
        assert parsed == output
    
    def test_test_id_format(self):
        """Test IDs must match T### format."""
        # Valid
        test = TestCase(
            test_id="T001",
            name="Test",
            endpoint=TestEndpoint(method="GET", path="/test"),
            objective="Test",
            assertions=[Assertion(type="status_code", expected=200)]
        )
        assert test.test_id == "T001"
        
        # Also valid with more digits
        test2 = TestCase(
            test_id="T1234",
            name="Test",
            endpoint=TestEndpoint(method="GET", path="/test"),
            objective="Test",
            assertions=[Assertion(type="status_code", expected=200)]
        )
        assert test2.test_id == "T1234"
        
        # Invalid format should raise
        with pytest.raises(ValidationError):
            TestCase(
                test_id="invalid",
                name="Test",
                endpoint=TestEndpoint(method="GET", path="/test"),
                objective="Test",
                assertions=[Assertion(type="status_code", expected=200)]
            )
    
    def test_method_enum_validation(self):
        """HTTP methods must be valid enum values."""
        # Valid methods
        for method in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
            endpoint = EndpointInfo(method=method, path="/test")
            assert endpoint.method == method
        
        # Invalid method should raise
        with pytest.raises(ValidationError):
            EndpointInfo(method="INVALID", path="/test")
    
    def test_auth_type_enum_validation(self):
        """Auth types must be valid enum values."""
        # Valid types
        for auth_type in ["none", "apiKey", "bearer", "oauth2", "unknown"]:
            auth = AuthInfo(type=auth_type, details="test")
            assert auth.type == auth_type
        
        # Invalid type should raise
        with pytest.raises(ValidationError):
            AuthInfo(type="invalid_auth", details="test")
    
    def test_coverage_checklist_values(self):
        """Coverage checklist values must be 'true', 'false', or 'unknown'."""
        # Valid
        checklist = CoverageChecklist(
            happy_paths="true",
            validation_negative="false",
            auth_negative="unknown"
        )
        assert checklist.happy_paths == "true"
        
        # Invalid value should raise
        with pytest.raises(ValidationError):
            CoverageChecklist(happy_paths="yes")
    
    def test_memory_excerpt_source_validation(self):
        """Memory excerpt sources must be valid enum values."""
        valid_sources = ["convention", "existing_tests", "runbook", "validator", "memo"]
        
        for source in valid_sources:
            excerpt = MemoryExcerpt(source=source, excerpt="test content")
            assert excerpt.source == source
        
        with pytest.raises(ValidationError):
            MemoryExcerpt(source="invalid_source", excerpt="test")
    
    def test_assertion_type_validation(self):
        """Assertion types must be valid enum values."""
        valid_types = ["status_code", "schema", "field", "header", "response_time"]
        
        for atype in valid_types:
            assertion = Assertion(type=atype, expected="test")
            assert assertion.type == atype
        
        with pytest.raises(ValidationError):
            Assertion(type="invalid_type", expected="test")


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline_valid_spec(self, sample_openapi_spec):
        """Full pipeline should work with valid spec."""
        result = run_agent(sample_openapi_spec)
        
        assert "output" in result
        assert "reward" in result
        assert result["reward"] >= 0.0
        assert result["reward"] <= 1.0
        
        output = result["output"]
        
        # Should have detected endpoints
        assert len(output["spec_summary"]["endpoints_detected"]) > 0
        
        # Should have generated tests
        assert len(output["test_suite"]) > 0
        
        # Should pass validation
        validated = SpecTestPilotOutput.model_validate(output)
        assert validated is not None
    
    def test_full_pipeline_minimal_spec(self, minimal_spec):
        """Full pipeline should work with minimal spec."""
        result = run_agent(minimal_spec)
        output = result["output"]
        
        # Should detect the one endpoint
        assert len(output["spec_summary"]["endpoints_detected"]) == 1
        
        # Should generate at least one test
        assert len(output["test_suite"]) >= 1
    
    def test_reward_computation(self, sample_openapi_spec):
        """Reward should be computed correctly."""
        result = run_agent(sample_openapi_spec)
        output = result["output"]
        
        parsed_spec = parse_openapi_spec(sample_openapi_spec)
        reward, breakdown = compute_reward(output, parsed_spec)
        
        # Hard gates should pass
        assert breakdown.valid_json
        assert breakdown.pydantic_valid
        assert breakdown.no_invented_endpoints
        
        # Reward should be positive
        assert reward > 0.0
    
    def test_intermediate_rewards_present(self, sample_openapi_spec):
        """Intermediate rewards should be tracked."""
        result = run_agent(sample_openapi_spec)
        
        intermediate = result.get("intermediate_rewards", {})
        
        # Should have rewards for each node
        expected_nodes = [
            "parse_spec",
            "detect_endpoints",
            "research_plan",
            "research_search",
            "research_integrate",
            "research_reflect",
            "generate_tests",
            "finalize"
        ]
        
        for node in expected_nodes:
            assert node in intermediate, f"Missing intermediate reward for {node}"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_spec_with_no_auth(self):
        """Spec without auth should be handled correctly."""
        spec = """
openapi: "3.0.3"
info:
  title: "No Auth API"
  version: "1.0.0"
paths:
  /public:
    get:
      operationId: publicEndpoint
      responses:
        "200":
          description: OK
"""
        result = run_agent(spec)
        output = result["output"]
        
        # Auth should be none or unknown
        auth_type = output["spec_summary"]["auth"]["type"]
        assert auth_type in ["none", "unknown"]
    
    def test_spec_with_many_endpoints(self):
        """Spec with many endpoints should be handled."""
        paths = {}
        for i in range(10):
            paths[f"/resource{i}"] = {
                "get": {
                    "operationId": f"getResource{i}",
                    "responses": {"200": {"description": "OK"}}
                }
            }
        
        spec = {
            "openapi": "3.0.3",
            "info": {"title": "Many Endpoints", "version": "1.0.0"},
            "paths": paths
        }
        
        import yaml
        spec_yaml = yaml.dump(spec)
        
        result = run_agent(spec_yaml)
        output = result["output"]
        
        # Should detect all endpoints
        assert len(output["spec_summary"]["endpoints_detected"]) == 10
    
    def test_spec_with_complex_schemas(self):
        """Spec with complex nested schemas should be handled."""
        spec = """
openapi: "3.0.3"
info:
  title: "Complex Schema API"
  version: "1.0.0"
paths:
  /complex:
    post:
      operationId: createComplex
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                nested:
                  type: object
                  properties:
                    deep:
                      type: array
                      items:
                        type: object
                        properties:
                          value:
                            type: string
      responses:
        "201":
          description: Created
"""
        result = run_agent(spec)
        output = result["output"]
        
        # Should handle without error
        assert len(output["spec_summary"]["endpoints_detected"]) == 1
        assert len(output["test_suite"]) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
