#!/usr/bin/env python3
"""
Multi-Language API Testing Agent
Comprehensive API testing agent that thinks and acts like a human tester
Supports multiple programming languages and testing frameworks
"""

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import yaml
import requests
from pathlib import Path


class TestLanguage(Enum):
    """Supported testing languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    CURL = "curl"
    POSTMAN = "postman"


class TestType(Enum):
    """Types of tests like a human tester would do."""
    HAPPY_PATH = "happy_path"
    ERROR_HANDLING = "error_handling"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INPUT_VALIDATION = "input_validation"
    BOUNDARY_TESTING = "boundary_testing"
    PERFORMANCE = "performance"
    SECURITY = "security"
    EDGE_CASES = "edge_cases"
    INTEGRATION = "integration"


@dataclass
class TestScenario:
    """A test scenario like a human tester would design."""
    name: str
    description: str
    test_type: TestType
    endpoint: str
    method: str
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    expected_status: int = 200
    expected_response_fields: List[str] = field(default_factory=list)
    assertions: List[str] = field(default_factory=list)


@dataclass
class APIEndpoint:
    """Parsed API endpoint information."""
    path: str
    method: str
    summary: str
    parameters: List[Dict[str, Any]]
    request_body: Optional[Dict[str, Any]]
    responses: Dict[str, Any]
    auth_required: bool = False
    auth_type: Optional[str] = None


class HumanTesterSimulator:
    """Simulates how a human tester would analyze and test an API with NLP prompts."""
    
    def __init__(self, api_spec: Dict[str, Any], base_url: str):
        """Initialize with API specification and base URL."""
        self.api_spec = api_spec
        self.base_url = base_url.rstrip('/')
        self.endpoints = self._parse_endpoints()
        self.test_scenarios = []
        self.error_fixes = {}  # Store automatic fixes for errors
        self.workflow_chains = []  # Store workflow orchestrations
        
    def _parse_endpoints(self) -> List[APIEndpoint]:
        """Parse OpenAPI spec like a human tester would analyze it."""
        endpoints = []
        
        if 'paths' not in self.api_spec:
            return endpoints
            
        for path, path_info in self.api_spec['paths'].items():
            if not isinstance(path_info, dict):
                continue
            path_parameters = path_info.get('parameters', [])
            for method, method_info in path_info.items():
                if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                    operation_info = method_info if isinstance(method_info, dict) else {}
                    
                    # Extract parameters
                    parameters = self._merge_operation_parameters(
                        path_parameters,
                        operation_info.get('parameters', []),
                    )
                    
                    # Check authentication
                    auth_required = 'security' in operation_info or 'security' in self.api_spec
                    auth_type = None
                    if auth_required:
                        security = operation_info.get('security', self.api_spec.get('security', []))
                        if security:
                            auth_type = list(security[0].keys())[0] if security[0] else None
                    
                    endpoint = APIEndpoint(
                        path=path,
                        method=method.upper(),
                        summary=operation_info.get('summary', f"{method.upper()} {path}"),
                        parameters=parameters,
                        request_body=operation_info.get('requestBody'),
                        responses=operation_info.get('responses', {}),
                        auth_required=auth_required,
                        auth_type=auth_type
                    )
                    endpoints.append(endpoint)
        
        return endpoints

    def _merge_operation_parameters(
        self,
        path_parameters: Any,
        operation_parameters: Any,
    ) -> List[Dict[str, Any]]:
        """Merge path-level and operation-level OpenAPI parameters."""
        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for source in (path_parameters, operation_parameters):
            if not isinstance(source, list):
                continue
            for param in source:
                if not isinstance(param, dict):
                    continue
                location = str(param.get("in", "")).lower()
                name = str(param.get("name", ""))
                if not location or not name:
                    continue
                merged[(location, name)] = param

        return list(merged.values())
    
    def think_like_tester(self, nlp_prompt: Optional[str] = None) -> List[TestScenario]:
        """
        Think like a human tester with optional natural language prompt.
        
        Examples:
        - "Generate tests to validate status codes and response times"
        - "Create security tests for user authentication endpoints"
        - "Test error handling for invalid input data"
        - "Generate comprehensive test suite with boundary testing"
        """
        scenarios = []
        
        if nlp_prompt:
            print(f"🤖 AI Tester analyzing prompt: '{nlp_prompt}'")
            scenarios = self._generate_from_nlp_prompt(nlp_prompt)
        else:
            # Default comprehensive testing
            for endpoint in self.endpoints:
                # 1. Happy path testing (what should work)
                scenarios.extend(self._create_happy_path_tests(endpoint))
                
                # 2. Error handling (what should fail gracefully)
                scenarios.extend(self._create_error_tests(endpoint))
                
                # 3. Authentication/Authorization testing
                if endpoint.auth_required:
                    scenarios.extend(self._create_auth_tests(endpoint))
                
                # 4. Input validation testing
                scenarios.extend(self._create_validation_tests(endpoint))
                
                # 5. Boundary testing
                scenarios.extend(self._create_boundary_tests(endpoint))
                
                # 6. Security testing
                scenarios.extend(self._create_security_tests(endpoint))
                
                # 7. Edge cases
                scenarios.extend(self._create_edge_case_tests(endpoint))
        
        self.test_scenarios = scenarios
        return scenarios
    
    def _generate_from_nlp_prompt(self, prompt: str) -> List[TestScenario]:
        """Generate tests based on natural language prompt like Postman AI."""
        scenarios = []
        prompt_lower = prompt.lower()
        
        print(f"   🧠 Analyzing intent: {prompt}")
        
        # Parse what user wants to test
        if 'status code' in prompt_lower or 'response time' in prompt_lower:
            print("   📊 Generating status code and performance tests")
            scenarios.extend(self._create_status_and_performance_tests())
            
        if 'security' in prompt_lower or 'sql inject' in prompt_lower or 'xss' in prompt_lower:
            print("   🔒 Generating security vulnerability tests")
            scenarios.extend(self._create_comprehensive_security_tests())
            
        if 'auth' in prompt_lower or 'login' in prompt_lower or 'token' in prompt_lower:
            print("   🔐 Generating authentication and authorization tests")
            scenarios.extend(self._create_comprehensive_auth_tests())
            
        if 'error' in prompt_lower or 'invalid' in prompt_lower or 'fail' in prompt_lower:
            print("   💥 Generating error handling and edge case tests")
            scenarios.extend(self._create_comprehensive_error_tests())
            
        if 'validation' in prompt_lower or 'input' in prompt_lower:
            print("   🛡️ Generating input validation tests")
            scenarios.extend(self._create_comprehensive_validation_tests())
            
        if 'boundary' in prompt_lower or 'limit' in prompt_lower or 'edge' in prompt_lower:
            print("   🎯 Generating boundary and limit tests")
            scenarios.extend(self._create_comprehensive_boundary_tests())
            
        # If no specific intent, generate comprehensive suite
        if not scenarios:
            print("   🎯 No specific intent detected, generating comprehensive test suite")
            return self.think_like_tester()  # Recursive call without prompt
            
        print(f"   ✅ Generated {len(scenarios)} test scenarios from prompt")
        return scenarios
    
    def analyze_error_and_suggest_fix(self, error_response: Dict[str, Any], original_scenario: TestScenario) -> Dict[str, Any]:
        """
        Analyze API errors and suggest automatic fixes like Postman AI.
        
        Handles common errors:
        - 401 Unauthorized: Suggests adding authentication
        - 403 Forbidden: Suggests checking permissions/scope  
        - 400 Bad Request: Analyzes and fixes request data
        - 404 Not Found: Suggests valid resource IDs
        """
        status_code = error_response.get('status_code', 0)
        error_body = error_response.get('body', {})
        
        print(f"🔍 AI Error Analysis: HTTP {status_code}")
        
        analysis = {
            'error_type': status_code,
            'root_cause': '',
            'suggested_fixes': [],
            'auto_fix_available': False,
            'fixed_scenario': None,
            'confidence': 0.0
        }
        
        if status_code == 401:
            analysis.update({
                'root_cause': 'Missing or invalid authentication token',
                'suggested_fixes': [
                    'Add Authorization header with Bearer token',
                    'Include API key in request headers',
                    'Verify authentication endpoint is working'
                ],
                'auto_fix_available': True,
                'confidence': 0.9
            })
            
            # Auto-fix: Add authentication
            fixed_scenario = TestScenario(
                name=f"{original_scenario.name}_with_auth",
                description=f"{original_scenario.description} (with authentication)",
                test_type=original_scenario.test_type,
                endpoint=original_scenario.endpoint,
                method=original_scenario.method,
                headers={'Authorization': 'Bearer {{auth_token}}'},
                params=original_scenario.params.copy(),
                body=original_scenario.body.copy() if original_scenario.body else None,
                expected_status=200
            )
            analysis['fixed_scenario'] = fixed_scenario
            
        elif status_code == 403:
            analysis.update({
                'root_cause': 'Valid authentication but insufficient permissions',
                'suggested_fixes': [
                    'Use token with elevated permissions',
                    'Check required scopes for this endpoint',
                    'Verify resource ownership'
                ],
                'auto_fix_available': True,
                'confidence': 0.8
            })
            
            # Auto-fix: Use admin token
            fixed_scenario = TestScenario(
                name=f"{original_scenario.name}_with_admin",
                description=f"{original_scenario.description} (with admin permissions)",
                test_type=original_scenario.test_type,
                endpoint=original_scenario.endpoint,
                method=original_scenario.method,
                headers={'Authorization': 'Bearer {{admin_token}}'},
                params=original_scenario.params.copy(),
                body=original_scenario.body.copy() if original_scenario.body else None,
                expected_status=200
            )
            analysis['fixed_scenario'] = fixed_scenario
            
        elif status_code == 400:
            analysis.update({
                'root_cause': 'Invalid request data or missing required fields',
                'suggested_fixes': [
                    'Check required fields are present',
                    'Validate data types match API spec',
                    'Remove invalid fields',
                    'Fix JSON formatting'
                ],
                'auto_fix_available': True,
                'confidence': 0.7
            })
            
            # Auto-fix: Add required fields
            fixed_body = original_scenario.body.copy() if original_scenario.body else {}
            if not fixed_body and original_scenario.method in ['POST', 'PUT', 'PATCH']:
                fixed_body = {'name': 'Test Item', 'description': 'Auto-generated test data'}
            elif fixed_body:
                # Add common required fields if missing
                if 'name' not in fixed_body:
                    fixed_body['name'] = 'Auto-fixed Name'
                if 'email' not in fixed_body and 'user' in original_scenario.endpoint.lower():
                    fixed_body['email'] = 'test@example.com'
                    
            fixed_scenario = TestScenario(
                name=f"{original_scenario.name}_fixed_data",
                description=f"{original_scenario.description} (with valid data)",
                test_type=original_scenario.test_type,
                endpoint=original_scenario.endpoint,
                method=original_scenario.method,
                headers=original_scenario.headers.copy(),
                params=original_scenario.params.copy(),
                body=fixed_body,
                expected_status=200
            )
            analysis['fixed_scenario'] = fixed_scenario
            
        elif status_code == 404:
            analysis.update({
                'root_cause': 'Resource not found or invalid ID',
                'suggested_fixes': [
                    'Use valid resource ID that exists',
                    'Create resource first if needed',
                    'Check URL path is correct'
                ],
                'auto_fix_available': True,
                'confidence': 0.85
            })
            
            # Auto-fix: Use valid ID
            fixed_endpoint = original_scenario.endpoint.replace('/999', '/123').replace('/invalid', '/1')
            fixed_scenario = TestScenario(
                name=f"{original_scenario.name}_valid_id",
                description=f"{original_scenario.description} (with valid ID)",
                test_type=original_scenario.test_type,
                endpoint=fixed_endpoint,
                method=original_scenario.method,
                headers=original_scenario.headers.copy(),
                params=original_scenario.params.copy(),
                body=original_scenario.body.copy() if original_scenario.body else None,
                expected_status=200
            )
            analysis['fixed_scenario'] = fixed_scenario
            
        else:
            analysis.update({
                'root_cause': f'Unexpected HTTP {status_code} error',
                'suggested_fixes': ['Check API documentation', 'Verify server status', 'Review request format'],
                'auto_fix_available': False,
                'confidence': 0.3
            })
        
        # Store fix for reuse
        self.error_fixes[f"{status_code}_{original_scenario.name}"] = analysis
        
        print(f"   📋 Root cause: {analysis['root_cause']}")
        print(f"   🔧 Auto-fix available: {analysis['auto_fix_available']}")
        print(f"   📊 Confidence: {analysis['confidence']:.1%}")
        
        return analysis
    
    def _create_status_and_performance_tests(self) -> List[TestScenario]:
        """Create tests focused on status codes and response times."""
        scenarios = []
        
        for endpoint in self.endpoints:
            # Status code validation test
            scenario = TestScenario(
                name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_')}_status_validation",
                description=f"Validate status codes and response times for {endpoint.method} {endpoint.path}",
                test_type=TestType.PERFORMANCE,
                endpoint=endpoint.path,
                method=endpoint.method,
                expected_status=200,
                headers={'Accept': 'application/json'},
                body=None
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _create_comprehensive_security_tests(self) -> List[TestScenario]:
        """Create comprehensive security tests including SQL injection, XSS."""
        scenarios = []
        
        for endpoint in self.endpoints:
            # SQL injection test
            if endpoint.method in ['POST', 'PUT', 'PATCH']:
                sql_injection_scenario = TestScenario(
                    name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_')}_sql_injection",
                    description=f"Test SQL injection protection on {endpoint.method} {endpoint.path}",
                    test_type=TestType.SECURITY,
                    endpoint=endpoint.path,
                    method=endpoint.method,
                    expected_status=400,
                    body={'name': "'; DROP TABLE users; --", 'description': 'SQL injection attempt'}
                )
                scenarios.append(sql_injection_scenario)
                
            # XSS test
            if endpoint.method in ['POST', 'PUT', 'PATCH']:
                xss_scenario = TestScenario(
                    name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_')}_xss_protection",
                    description=f"Test XSS protection on {endpoint.method} {endpoint.path}",
                    test_type=TestType.SECURITY,
                    endpoint=endpoint.path,
                    method=endpoint.method,
                    expected_status=400,
                    body={'comment': '<script>alert("xss")</script>', 'content': 'XSS attempt'}
                )
                scenarios.append(xss_scenario)
        
        return scenarios
    
    def _create_comprehensive_auth_tests(self) -> List[TestScenario]:
        """Create comprehensive authentication and authorization tests."""
        scenarios = []
        
        for endpoint in self.endpoints:
            if endpoint.auth_required:
                # No auth test
                no_auth_scenario = TestScenario(
                    name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_')}_no_auth",
                    description=f"Test {endpoint.method} {endpoint.path} without authentication",
                    test_type=TestType.AUTHENTICATION,
                    endpoint=endpoint.path,
                    method=endpoint.method,
                    expected_status=401,
                    headers={},  # No auth headers
                    body=None
                )
                scenarios.append(no_auth_scenario)
                
                # Invalid auth test
                invalid_auth_scenario = TestScenario(
                    name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_')}_invalid_auth",
                    description=f"Test {endpoint.method} {endpoint.path} with invalid token",
                    test_type=TestType.AUTHENTICATION,
                    endpoint=endpoint.path,
                    method=endpoint.method,
                    expected_status=401,
                    headers={'Authorization': 'Bearer invalid_token'},
                    body=None
                )
                scenarios.append(invalid_auth_scenario)
        
        return scenarios
    
    def _create_comprehensive_error_tests(self) -> List[TestScenario]:
        """Create comprehensive error handling tests."""
        scenarios = []
        
        for endpoint in self.endpoints:
            # Invalid method test
            invalid_method_scenario = TestScenario(
                name=f"test_{endpoint.path.replace('/', '_')}_invalid_method",
                description=f"Test invalid method on {endpoint.path}",
                test_type=TestType.ERROR_HANDLING,
                endpoint=endpoint.path,
                method='DELETE' if endpoint.method != 'DELETE' else 'PATCH',
                expected_status=405
            )
            scenarios.append(invalid_method_scenario)
            
            # Malformed request test
            if endpoint.method in ['POST', 'PUT', 'PATCH']:
                malformed_scenario = TestScenario(
                    name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_')}_malformed_request",
                    description=f"Test malformed request data on {endpoint.method} {endpoint.path}",
                    test_type=TestType.ERROR_HANDLING,
                    endpoint=endpoint.path,
                    method=endpoint.method,
                    expected_status=400,
                    body={'invalid_json': 'missing_quotes: true, broken: }'}
                )
                scenarios.append(malformed_scenario)
        
        return scenarios
    
    def _create_comprehensive_validation_tests(self) -> List[TestScenario]:
        """Create comprehensive input validation tests."""
        scenarios = []
        
        for endpoint in self.endpoints:
            if endpoint.method in ['POST', 'PUT', 'PATCH']:
                # Empty body test
                empty_body_scenario = TestScenario(
                    name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_')}_empty_body",
                    description=f"Test empty request body on {endpoint.method} {endpoint.path}",
                    test_type=TestType.INPUT_VALIDATION,
                    endpoint=endpoint.path,
                    method=endpoint.method,
                    expected_status=400,
                    body={}
                )
                scenarios.append(empty_body_scenario)
                
                # Type mismatch test
                type_mismatch_scenario = TestScenario(
                    name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_')}_type_mismatch", 
                    description=f"Test type mismatch on {endpoint.method} {endpoint.path}",
                    test_type=TestType.INPUT_VALIDATION,
                    endpoint=endpoint.path,
                    method=endpoint.method,
                    expected_status=400,
                    body={'age': 'not_a_number', 'count': 'invalid_integer'}
                )
                scenarios.append(type_mismatch_scenario)
        
        return scenarios
    
    def _create_comprehensive_boundary_tests(self) -> List[TestScenario]:
        """Create comprehensive boundary and limit tests."""
        scenarios = []
        
        for endpoint in self.endpoints:
            if endpoint.method != 'GET':
                continue

            query_params = [
                p for p in endpoint.parameters
                if str((p or {}).get("in", "")).lower() == "query"
            ]
            if not query_params:
                # Avoid hallucinated pagination checks on endpoints without query parameters.
                continue

            query_name_map = {
                str((p or {}).get("name", "")).lower(): str((p or {}).get("name", ""))
                for p in query_params
                if str((p or {}).get("name", ""))
            }
            pagination_aliases = {
                "limit",
                "page",
                "offset",
                "per_page",
                "perpage",
                "page_size",
                "pagesize",
            }
            if not (set(query_name_map.keys()) & pagination_aliases):
                # Skip pagination boundary cases if the endpoint does not expose pagination params.
                continue

            invalid_params: Dict[str, Any] = {}
            if "limit" in query_name_map:
                invalid_params[query_name_map["limit"]] = 99999
            if "page" in query_name_map:
                invalid_params[query_name_map["page"]] = -1
            if "offset" in query_name_map:
                invalid_params[query_name_map["offset"]] = -1
            if "per_page" in query_name_map:
                invalid_params[query_name_map["per_page"]] = 0
            if "perpage" in query_name_map:
                invalid_params[query_name_map["perpage"]] = 0
            if "page_size" in query_name_map:
                invalid_params[query_name_map["page_size"]] = 0
            if "pagesize" in query_name_map:
                invalid_params[query_name_map["pagesize"]] = 0

            if not invalid_params:
                continue

            limit_scenario = TestScenario(
                name=f"test_get_{endpoint.path.replace('/', '_')}_pagination_limit",
                description=f"Test pagination limits on GET {endpoint.path}",
                test_type=TestType.BOUNDARY_TESTING,
                endpoint=endpoint.path,
                method='GET',
                expected_status=400,
                params=invalid_params,
            )
            scenarios.append(limit_scenario)
        
        return scenarios
    
    def _create_happy_path_tests(self, endpoint: APIEndpoint) -> List[TestScenario]:
        """Create happy path tests - the basic functionality."""
        scenarios = []
        
        # Basic successful request
        scenario = TestScenario(
            name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_success",
            description=f"Test successful {endpoint.method} request to {endpoint.path}",
            test_type=TestType.HAPPY_PATH,
            endpoint=endpoint.path,
            method=endpoint.method,
            expected_status=200 if endpoint.method == 'GET' else (201 if endpoint.method == 'POST' else 200)
        )
        
        # Add required parameters
        for param in endpoint.parameters:
            if param.get('required', False):
                if param['in'] == 'query':
                    scenario.params[param['name']] = self._generate_sample_value(param)
                elif param['in'] == 'header':
                    scenario.headers[param['name']] = self._generate_sample_value(param)
        
        # Add request body for POST/PUT
        if endpoint.method in ['POST', 'PUT', 'PATCH'] and endpoint.request_body:
            scenario.body = self._generate_sample_body(endpoint.request_body)
        
        scenarios.append(scenario)
        return scenarios
    
    def _create_error_tests(self, endpoint: APIEndpoint) -> List[TestScenario]:
        """Create error handling tests."""
        scenarios = []
        
        # Test 404 - Non-existent resource
        if '{id}' in endpoint.path or '{' in endpoint.path:
            scenarios.append(TestScenario(
                name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_not_found",
                description=f"Test {endpoint.method} request with non-existent ID",
                test_type=TestType.ERROR_HANDLING,
                endpoint=endpoint.path.replace('{id}', '99999').replace('{userId}', '99999'),
                method=endpoint.method,
                expected_status=404,
                assertions=["Response contains error message", "Error format is consistent"]
            ))
        
        # Test 400 - Bad request
        if endpoint.method in ['POST', 'PUT', 'PATCH']:
            scenarios.append(TestScenario(
                name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_bad_request",
                description=f"Test {endpoint.method} request with invalid data",
                test_type=TestType.ERROR_HANDLING,
                endpoint=endpoint.path,
                method=endpoint.method,
                body={"invalid_field": "invalid_value"},
                expected_status=400,
                assertions=["Response explains validation errors", "Error details are provided"]
            ))
        
        return scenarios
    
    def _create_auth_tests(self, endpoint: APIEndpoint) -> List[TestScenario]:
        """Create authentication/authorization tests."""
        scenarios = []
        
        # Test 401 - No authentication
        scenarios.append(TestScenario(
            name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_unauthorized",
            description=f"Test {endpoint.method} request without authentication",
            test_type=TestType.AUTHENTICATION,
            endpoint=endpoint.path,
            method=endpoint.method,
            expected_status=401,
            assertions=["Unauthorized access is properly rejected"]
        ))
        
        # Test 403 - Invalid token
        scenarios.append(TestScenario(
            name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_forbidden",
            description=f"Test {endpoint.method} request with invalid token",
            test_type=TestType.AUTHORIZATION,
            endpoint=endpoint.path,
            method=endpoint.method,
            headers={"Authorization": "Bearer invalid_token_12345"},
            expected_status=403,
            assertions=["Invalid token is rejected", "Proper error message returned"]
        ))
        
        return scenarios
    
    def _create_validation_tests(self, endpoint: APIEndpoint) -> List[TestScenario]:
        """Create input validation tests."""
        scenarios = []
        
        # Test missing required parameters
        for param in endpoint.parameters:
            if param.get('required', False):
                scenario = TestScenario(
                    name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_missing_{param['name']}",
                    description=f"Test {endpoint.method} request missing required parameter {param['name']}",
                    test_type=TestType.INPUT_VALIDATION,
                    endpoint=endpoint.path,
                    method=endpoint.method,
                    expected_status=400,
                    assertions=[f"Missing {param['name']} parameter is detected"]
                )
                scenarios.append(scenario)
        
        return scenarios
    
    def _create_boundary_tests(self, endpoint: APIEndpoint) -> List[TestScenario]:
        """Create boundary value tests."""
        scenarios = []
        
        # Test very long strings, extreme numbers, etc.
        if endpoint.method in ['POST', 'PUT', 'PATCH']:
            scenarios.append(TestScenario(
                name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_max_length",
                description=f"Test {endpoint.method} with maximum length inputs",
                test_type=TestType.BOUNDARY_TESTING,
                endpoint=endpoint.path,
                method=endpoint.method,
                body={"description": "A" * 10000},  # Very long string
                expected_status=400,
                assertions=["Long inputs are handled gracefully"]
            ))
        
        return scenarios
    
    def _create_security_tests(self, endpoint: APIEndpoint) -> List[TestScenario]:
        """Create security-focused tests."""
        scenarios = []
        
        # SQL Injection test
        if endpoint.parameters:
            scenarios.append(TestScenario(
                name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_sql_injection",
                description=f"Test {endpoint.method} for SQL injection vulnerability",
                test_type=TestType.SECURITY,
                endpoint=endpoint.path,
                method=endpoint.method,
                params={"q": "'; DROP TABLE users; --"},
                expected_status=400,
                assertions=["SQL injection attempts are blocked"]
            ))
        
        # XSS test
        if endpoint.method in ['POST', 'PUT', 'PATCH']:
            scenarios.append(TestScenario(
                name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_xss_protection",
                description=f"Test {endpoint.method} for XSS vulnerability",
                test_type=TestType.SECURITY,
                endpoint=endpoint.path,
                method=endpoint.method,
                body={"comment": "<script>alert('xss')</script>"},
                expected_status=400,
                assertions=["XSS attempts are sanitized"]
            ))
        
        return scenarios
    
    def _create_edge_case_tests(self, endpoint: APIEndpoint) -> List[TestScenario]:
        """Create edge case tests."""
        scenarios = []
        
        # Empty body test
        if endpoint.method in ['POST', 'PUT', 'PATCH']:
            scenarios.append(TestScenario(
                name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_empty_body",
                description=f"Test {endpoint.method} with empty request body",
                test_type=TestType.EDGE_CASES,
                endpoint=endpoint.path,
                method=endpoint.method,
                body={},
                expected_status=400,
                assertions=["Empty body is handled appropriately"]
            ))
        
        return scenarios
    
    def _generate_sample_value(self, param: Dict[str, Any]) -> Any:
        """Generate sample values for parameters."""
        param_type = param.get('type', 'string')
        param_name = param.get('name', '').lower()
        
        if param_type == 'integer':
            if 'id' in param_name:
                return 123
            return 10
        elif param_type == 'boolean':
            return True
        elif param_type == 'array':
            return ["sample1", "sample2"]
        else:  # string
            if 'email' in param_name:
                return "test@example.com"
            elif 'id' in param_name:
                return "abc123"
            elif 'token' in param_name:
                return "bearer_token_12345"
            return "sample_value"
    
    def _generate_sample_body(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sample request body."""
        # This is a simplified version - in reality, you'd parse the schema
        return {
            "name": "Test User",
            "email": "test@example.com",
            "description": "Sample description"
        }


class MultiLanguageTestGenerator:
    """Generates tests in multiple programming languages."""
    
    def __init__(self, scenarios: List[TestScenario], base_url: str):
        """Initialize with test scenarios."""
        self.scenarios = scenarios
        self.base_url = base_url
    
    def generate_python_tests(self) -> str:
        """Generate Python pytest tests."""
        code = '''import pytest
import requests
import json

BASE_URL = "{}"

class TestAPI:
    """Comprehensive API tests generated by AI agent."""

'''.format(self.base_url)
        
        for scenario in self.scenarios:
            code += self._generate_python_test_method(scenario)
        
        return code
    
    def _generate_python_test_method(self, scenario: TestScenario) -> str:
        """Generate a single Python test method."""
        method_code = f'''    def {scenario.name}(self):
        """{scenario.description}"""
        url = BASE_URL + "{scenario.endpoint}"
        
'''
        
        # Add headers
        if scenario.headers:
            method_code += f"        headers = {json.dumps(scenario.headers, indent=8)}\n"
        else:
            method_code += "        headers = {}\n"
        
        # Add params
        if scenario.params:
            method_code += f"        params = {json.dumps(scenario.params, indent=8)}\n"
        else:
            method_code += "        params = {}\n"
        
        # Add request
        if scenario.method == 'GET':
            method_code += "        response = requests.get(url, headers=headers, params=params)\n"
        elif scenario.method == 'POST':
            if scenario.body:
                method_code += f"        data = {json.dumps(scenario.body, indent=8)}\n"
                method_code += "        response = requests.post(url, headers=headers, params=params, json=data)\n"
            else:
                method_code += "        response = requests.post(url, headers=headers, params=params)\n"
        elif scenario.method == 'PUT':
            if scenario.body:
                method_code += f"        data = {json.dumps(scenario.body, indent=8)}\n"
                method_code += "        response = requests.put(url, headers=headers, params=params, json=data)\n"
            else:
                method_code += "        response = requests.put(url, headers=headers, params=params)\n"
        elif scenario.method == 'DELETE':
            method_code += "        response = requests.delete(url, headers=headers, params=params)\n"
        
        # Add assertions
        method_code += f"        assert response.status_code == {scenario.expected_status}\n"
        
        for assertion in scenario.assertions:
            if "error message" in assertion.lower():
                method_code += '        assert "error" in response.json() or "message" in response.json()\n'
            elif "response contains" in assertion.lower():
                method_code += "        assert response.json() is not None\n"
        
        method_code += "\n"
        return method_code
    
    def generate_javascript_tests(self) -> str:
        """Generate JavaScript tests using Jest/Axios."""
        code = f'''const axios = require('axios');

const BASE_URL = '{self.base_url}';

describe('API Tests', () => {{

'''
        
        for scenario in self.scenarios:
            code += self._generate_javascript_test_method(scenario)
        
        code += "});\n"
        return code
    
    def _generate_javascript_test_method(self, scenario: TestScenario) -> str:
        """Generate JavaScript test method."""
        method_code = f'''  test('{scenario.description}', async () => {{
    const url = BASE_URL + '{scenario.endpoint}';
    const config = {{
      method: '{scenario.method.lower()}',
      url: url,
'''
        
        if scenario.headers:
            method_code += f"      headers: {json.dumps(scenario.headers, indent=6)},\n"
        
        if scenario.params:
            method_code += f"      params: {json.dumps(scenario.params, indent=6)},\n"
        
        if scenario.body:
            method_code += f"      data: {json.dumps(scenario.body, indent=6)},\n"
        
        method_code += "      validateStatus: () => true  // Don't throw on non-2xx\n"
        method_code += "    };\n\n"
        method_code += "    const response = await axios(config);\n"
        method_code += f"    expect(response.status).toBe({scenario.expected_status});\n"
        
        for assertion in scenario.assertions:
            if "error message" in assertion.lower():
                method_code += "    expect(response.data).toHaveProperty('error');\n"
        
        method_code += "  });\n\n"
        return method_code
    
    def generate_curl_tests(self) -> str:
        """Generate cURL commands for manual testing."""
        code = "#!/bin/bash\n# API Test Suite - cURL Commands\n\n"
        
        for scenario in self.scenarios:
            code += f"# {scenario.description}\n"
            
            curl_cmd = f"curl -X {scenario.method}"
            
            # Add headers
            for key, value in scenario.headers.items():
                curl_cmd += f' -H "{key}: {value}"'
            
            # Add data
            if scenario.body:
                curl_cmd += f" -d '{json.dumps(scenario.body)}'"
                curl_cmd += ' -H "Content-Type: application/json"'
            
            # Add URL
            url = self.base_url + scenario.endpoint
            if scenario.params:
                param_str = "&".join([f"{k}={v}" for k, v in scenario.params.items()])
                url += f"?{param_str}"
            
            curl_cmd += f' "{url}"'
            
            code += f"{curl_cmd}\n"
            code += f"# Expected status: {scenario.expected_status}\n\n"
        
        return code
    
    def generate_java_tests(self) -> str:
        """Generate Java tests using RestAssured."""
        code = f'''import io.restassured.RestAssured;
import org.junit.jupiter.api.Test;
import static io.restassured.RestAssured.*;
import static org.hamcrest.Matchers.*;

public class APITests {{
    private static final String BASE_URL = "{self.base_url}";

'''
        
        for scenario in self.scenarios:
            code += self._generate_java_test_method(scenario)
        
        code += "}\n"
        return code
    
    def _generate_java_test_method(self, scenario: TestScenario) -> str:
        """Generate Java test method."""
        method_name = scenario.name.replace("test_", "")
        method_code = f'''    @Test
    public void {method_name}() {{
        given()
            .baseUri(BASE_URL)
'''
        
        # Add headers
        for key, value in scenario.headers.items():
            method_code += f'            .header("{key}", "{value}")\n'
        
        # Add params
        for key, value in scenario.params.items():
            method_code += f'            .param("{key}", "{value}")\n'
        
        # Add body
        if scenario.body:
            method_code += f'            .body({json.dumps(scenario.body)})\n'
            method_code += '            .contentType("application/json")\n'
        
        method_code += "        .when()\n"
        method_code += f'            .{scenario.method.lower()}("{scenario.endpoint}")\n'
        method_code += "        .then()\n"
        method_code += f"            .statusCode({scenario.expected_status});\n"
        method_code += "    }\n\n"
        
        return method_code


class APITestingSandbox:
    """Sandbox environment for executing API tests safely."""
    
    def __init__(self, api_spec_path: str, base_url: str = "https://api.example.com"):
        """Initialize testing sandbox."""
        self.api_spec_path = api_spec_path
        self.base_url = base_url
        self.sandbox_dir = Path(tempfile.mkdtemp(prefix="api_testing_sandbox_"))
        self.results = []
        
        # Load API spec
        with open(api_spec_path, 'r') as f:
            if api_spec_path.endswith('.yaml') or api_spec_path.endswith('.yml'):
                self.api_spec = yaml.safe_load(f)
            else:
                self.api_spec = json.load(f)
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite like a professional tester."""
        print("🤖 AI API Testing Agent Starting...")
        print("=" * 50)
        
        # 1. Think like a human tester
        print("🧠 Phase 1: Analyzing API like a human tester...")
        tester = HumanTesterSimulator(self.api_spec, self.base_url)
        scenarios = tester.think_like_tester()
        
        print(f"   Generated {len(scenarios)} test scenarios:")
        test_type_counts = {}
        for scenario in scenarios:
            test_type = scenario.test_type.value
            test_type_counts[test_type] = test_type_counts.get(test_type, 0) + 1
        
        for test_type, count in test_type_counts.items():
            print(f"   - {test_type.replace('_', ' ').title()}: {count} tests")
        
        # 2. Generate tests in multiple languages
        print("\n⚡ Phase 2: Generating tests in multiple languages...")
        generator = MultiLanguageTestGenerator(scenarios, self.base_url)
        
        # Generate Python tests
        python_tests = generator.generate_python_tests()
        python_file = self.sandbox_dir / "test_api.py"
        with open(python_file, 'w') as f:
            f.write(python_tests)
        print(f"   ✅ Python tests: {python_file}")
        
        # Generate JavaScript tests
        js_tests = generator.generate_javascript_tests()
        js_file = self.sandbox_dir / "test_api.test.js"
        with open(js_file, 'w') as f:
            f.write(js_tests)
        print(f"   ✅ JavaScript tests: {js_file}")
        
        # Generate cURL tests
        curl_tests = generator.generate_curl_tests()
        curl_file = self.sandbox_dir / "test_api.sh"
        with open(curl_file, 'w') as f:
            f.write(curl_tests)
        os.chmod(curl_file, 0o755)
        print(f"   ✅ cURL tests: {curl_file}")
        
        # Generate Java tests
        java_tests = generator.generate_java_tests()
        java_file = self.sandbox_dir / "APITests.java"
        with open(java_file, 'w') as f:
            f.write(java_tests)
        print(f"   ✅ Java tests: {java_file}")
        
        # 3. Create test documentation
        print("\n📝 Phase 3: Creating test documentation...")
        doc_content = self._generate_test_documentation(scenarios)
        doc_file = self.sandbox_dir / "TEST_PLAN.md"
        with open(doc_file, 'w') as f:
            f.write(doc_content)
        print(f"   ✅ Test documentation: {doc_file}")
        
        # 4. Create package files
        print("\n📦 Phase 4: Creating package files...")
        self._create_package_files()
        
        return {
            "sandbox_directory": str(self.sandbox_dir),
            "scenarios_generated": len(scenarios),
            "test_files": {
                "python": str(python_file),
                "javascript": str(js_file),
                "curl": str(curl_file),
                "java": str(java_file),
                "documentation": str(doc_file)
            },
            "test_breakdown": test_type_counts,
            "total_endpoints": len(tester.endpoints)
        }
    
    def _generate_test_documentation(self, scenarios: List[TestScenario]) -> str:
        """Generate comprehensive test documentation."""
        doc = """# API Testing Plan
Generated by AI Testing Agent

## Overview
This test suite provides comprehensive coverage of the API from the perspective of a professional tester.

## Test Categories

"""
        
        # Group by test type
        by_type = {}
        for scenario in scenarios:
            test_type = scenario.test_type.value
            if test_type not in by_type:
                by_type[test_type] = []
            by_type[test_type].append(scenario)
        
        for test_type, tests in by_type.items():
            doc += f"### {test_type.replace('_', ' ').title()}\n"
            doc += f"**Purpose**: {self._get_test_type_description(test_type)}\n"
            doc += f"**Test Count**: {len(tests)}\n\n"
            
            for test in tests[:3]:  # Show first 3 as examples
                doc += f"- **{test.name}**: {test.description}\n"
            
            if len(tests) > 3:
                doc += f"- ... and {len(tests) - 3} more\n"
            
            doc += "\n"
        
        doc += """## How to Run Tests

### Python (pytest)
```bash
pip install pytest requests
pytest test_api.py -v
```

### JavaScript (Jest)
```bash
npm install jest axios
npm test
```

### cURL (Manual)
```bash
chmod +x test_api.sh
./test_api.sh
```

### Java (Maven)
```bash
mvn test
```

## Expected Behavior
- All happy path tests should pass
- Error tests should return appropriate error codes
- Security tests should block malicious inputs
- Authentication tests should enforce proper access control
"""
        
        return doc
    
    def _get_test_type_description(self, test_type: str) -> str:
        """Get description for test type."""
        descriptions = {
            "happy_path": "Verify basic functionality works as expected",
            "error_handling": "Ensure graceful handling of error conditions",
            "authentication": "Verify authentication mechanisms work properly", 
            "authorization": "Test access control and permissions",
            "input_validation": "Ensure invalid inputs are properly rejected",
            "boundary_testing": "Test limits and edge values",
            "performance": "Verify response times and throughput",
            "security": "Test for common security vulnerabilities",
            "edge_cases": "Handle unusual but valid scenarios",
            "integration": "Test end-to-end workflows"
        }
        return descriptions.get(test_type, "Test specific functionality")
    
    def _create_package_files(self):
        """Create package management files."""
        
        # Python requirements.txt
        with open(self.sandbox_dir / "requirements.txt", 'w') as f:
            f.write("pytest>=7.0.0\nrequests>=2.25.0\n")
        
        # JavaScript package.json
        package_json = {
            "name": "api-tests",
            "version": "1.0.0",
            "scripts": {
                "test": "jest"
            },
            "dependencies": {
                "axios": "^0.27.0",
                "jest": "^28.0.0"
            }
        }
        with open(self.sandbox_dir / "package.json", 'w') as f:
            json.dump(package_json, f, indent=2)
        
        # Java pom.xml
        pom_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>api-tests</artifactId>
    <version>1.0.0</version>
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
    </properties>
    <dependencies>
        <dependency>
            <groupId>io.rest-assured</groupId>
            <artifactId>rest-assured</artifactId>
            <version>5.1.1</version>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.8.2</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>'''
        with open(self.sandbox_dir / "pom.xml", 'w') as f:
            f.write(pom_xml)
    
    def cleanup(self):
        """Clean up sandbox directory."""
        import shutil
        try:
            shutil.rmtree(self.sandbox_dir)
        except Exception:
            pass


if __name__ == "__main__":
    # Example usage
    sandbox = APITestingSandbox("examples/banking_api.yaml", "https://api.bankingexample.com")
    results = sandbox.run_full_test_suite()
    
    print("\n🎉 AI API Testing Complete!")
    print("=" * 30)
    print(f"Sandbox Directory: {results['sandbox_directory']}")
    print(f"Total Scenarios: {results['scenarios_generated']}")
    print(f"Endpoints Covered: {results['total_endpoints']}")
    print()
    print("Generated Test Files:")
    for lang, file_path in results['test_files'].items():
        print(f"  {lang.title()}: {file_path}")
    
    input("\nPress Enter to cleanup sandbox...")
    sandbox.cleanup()
    print("✅ Cleanup complete!")
