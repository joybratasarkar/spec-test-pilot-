"""
OpenAPI specification parser.

Extracts endpoints, parameters, schemas, and authentication from OpenAPI/Swagger specs.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import yaml


@dataclass
class ParsedParameter:
    """Parsed parameter from OpenAPI spec."""
    name: str
    location: str  # path, query, header, cookie
    required: bool = False
    schema_type: str = "string"
    description: str = ""
    example: Optional[Any] = None


@dataclass
class ParsedRequestBody:
    """Parsed request body from OpenAPI spec."""
    content_type: str = "application/json"
    required: bool = False
    schema: Dict[str, Any] = field(default_factory=dict)
    example: Optional[Any] = None


@dataclass
class ParsedResponse:
    """Parsed response from OpenAPI spec."""
    status_code: int
    description: str = ""
    content_type: str = "application/json"
    schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedEndpoint:
    """Parsed endpoint from OpenAPI spec."""
    method: str
    path: str
    operation_id: str = "unknown"
    summary: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parameters: List[ParsedParameter] = field(default_factory=list)
    request_body: Optional[ParsedRequestBody] = None
    responses: List[ParsedResponse] = field(default_factory=list)
    security: List[Dict[str, List[str]]] = field(default_factory=list)


@dataclass
class ParsedAuth:
    """Parsed authentication from OpenAPI spec."""
    type: str = "unknown"  # none, apiKey, bearer, oauth2, unknown
    details: str = "unknown"
    schemes: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ParsedSpec:
    """Complete parsed OpenAPI specification."""
    title: str = "unknown"
    version: str = "unknown"
    base_url: str = "unknown"
    description: str = ""
    auth: ParsedAuth = field(default_factory=ParsedAuth)
    endpoints: List[ParsedEndpoint] = field(default_factory=list)
    schemas: Dict[str, Any] = field(default_factory=dict)
    raw_spec: Dict[str, Any] = field(default_factory=dict)
    parse_errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if spec was parsed successfully."""
        return len(self.endpoints) > 0 or self.title != "unknown"


def parse_openapi_spec(spec_content: str) -> ParsedSpec:
    """
    Parse OpenAPI/Swagger specification from YAML or JSON string.
    
    Args:
        spec_content: Raw spec content (YAML or JSON)
        
    Returns:
        ParsedSpec with extracted information
    """
    if not spec_content or not spec_content.strip():
        return ParsedSpec(
            parse_errors=["Empty or missing spec content"]
        )
    
    # Try to parse as YAML first (also handles JSON)
    try:
        spec_dict = yaml.safe_load(spec_content)
    except yaml.YAMLError as e:
        # Try JSON explicitly
        try:
            spec_dict = json.loads(spec_content)
        except json.JSONDecodeError as je:
            return ParsedSpec(
                parse_errors=[f"Failed to parse spec: YAML error: {e}, JSON error: {je}"]
            )
    
    if not isinstance(spec_dict, dict):
        return ParsedSpec(
            parse_errors=["Spec content is not a valid object/dictionary"]
        )
    
    return _parse_spec_dict(spec_dict)


def _parse_spec_dict(spec: Dict[str, Any]) -> ParsedSpec:
    """Parse spec dictionary into ParsedSpec."""
    errors: List[str] = []
    
    # Detect OpenAPI version
    openapi_version = spec.get("openapi", spec.get("swagger", ""))
    is_openapi3 = str(openapi_version).startswith("3")
    
    # Parse info
    info = spec.get("info", {})
    title = info.get("title", "unknown")
    version = info.get("version", "unknown")
    description = info.get("description", "")
    
    # Parse base URL
    base_url = _extract_base_url(spec, is_openapi3)
    
    # Parse authentication
    auth = _parse_auth(spec, is_openapi3)
    
    # Parse schemas/definitions
    if is_openapi3:
        schemas = spec.get("components", {}).get("schemas", {})
    else:
        schemas = spec.get("definitions", {})
    
    # Parse endpoints
    endpoints = _parse_endpoints(spec, is_openapi3, errors)
    
    return ParsedSpec(
        title=title,
        version=version,
        base_url=base_url,
        description=description,
        auth=auth,
        endpoints=endpoints,
        schemas=schemas,
        raw_spec=spec,
        parse_errors=errors
    )


def _extract_base_url(spec: Dict[str, Any], is_openapi3: bool) -> str:
    """Extract base URL from spec."""
    if is_openapi3:
        servers = spec.get("servers", [])
        if servers and isinstance(servers, list):
            return servers[0].get("url", "unknown")
    else:
        # Swagger 2.0
        host = spec.get("host", "")
        base_path = spec.get("basePath", "")
        schemes = spec.get("schemes", ["https"])
        scheme = schemes[0] if schemes else "https"
        if host:
            return f"{scheme}://{host}{base_path}"
    return "unknown"


def _parse_auth(spec: Dict[str, Any], is_openapi3: bool) -> ParsedAuth:
    """Parse authentication schemes from spec."""
    if is_openapi3:
        security_schemes = spec.get("components", {}).get("securitySchemes", {})
    else:
        security_schemes = spec.get("securityDefinitions", {})
    
    if not security_schemes:
        # Check if there's global security requirement
        global_security = spec.get("security", [])
        if not global_security:
            return ParsedAuth(type="none", details="No authentication required")
    
    # Determine primary auth type
    auth_type = "unknown"
    details_parts = []
    
    for name, scheme in security_schemes.items():
        scheme_type = scheme.get("type", "").lower()
        
        if scheme_type == "apikey":
            auth_type = "apiKey"
            location = scheme.get("in", "header")
            key_name = scheme.get("name", "api_key")
            details_parts.append(f"API Key '{key_name}' in {location}")
            
        elif scheme_type == "http":
            http_scheme = scheme.get("scheme", "").lower()
            if http_scheme == "bearer":
                auth_type = "bearer"
                details_parts.append("Bearer token authentication")
            elif http_scheme == "basic":
                auth_type = "apiKey"  # Treat basic as apiKey variant
                details_parts.append("HTTP Basic authentication")
                
        elif scheme_type == "oauth2":
            auth_type = "oauth2"
            flows = scheme.get("flows", scheme.get("flow", {}))
            if isinstance(flows, dict):
                flow_types = list(flows.keys())
                details_parts.append(f"OAuth2 with flows: {', '.join(flow_types)}")
            else:
                details_parts.append(f"OAuth2 flow: {flows}")
                
        elif scheme_type == "openidconnect":
            auth_type = "oauth2"
            details_parts.append("OpenID Connect")
    
    details = "; ".join(details_parts) if details_parts else "unknown"
    
    return ParsedAuth(
        type=auth_type,
        details=details,
        schemes=security_schemes
    )


def _parse_endpoints(
    spec: Dict[str, Any],
    is_openapi3: bool,
    errors: List[str]
) -> List[ParsedEndpoint]:
    """Parse all endpoints from spec."""
    endpoints = []
    paths = spec.get("paths", {})
    
    if not paths:
        errors.append("No paths found in spec")
        return endpoints
    
    http_methods = {"get", "post", "put", "patch", "delete", "options", "head"}
    
    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
            
        # Get path-level parameters
        path_params = path_item.get("parameters", [])
        
        for method, operation in path_item.items():
            if method.lower() not in http_methods:
                continue
                
            if not isinstance(operation, dict):
                continue
            
            try:
                endpoint = _parse_operation(
                    path, method.upper(), operation, path_params, is_openapi3
                )
                endpoints.append(endpoint)
            except Exception as e:
                errors.append(f"Error parsing {method.upper()} {path}: {e}")
    
    return endpoints


def _parse_operation(
    path: str,
    method: str,
    operation: Dict[str, Any],
    path_params: List[Dict[str, Any]],
    is_openapi3: bool
) -> ParsedEndpoint:
    """Parse a single operation into ParsedEndpoint."""
    # Combine path-level and operation-level parameters
    all_params = path_params + operation.get("parameters", [])
    
    parameters = []
    for param in all_params:
        if not isinstance(param, dict):
            continue
        
        # Handle $ref
        if "$ref" in param:
            # Skip refs for now, would need resolution
            continue
            
        param_schema = param.get("schema", {})
        parameters.append(ParsedParameter(
            name=param.get("name", "unknown"),
            location=param.get("in", "query"),
            required=param.get("required", False),
            schema_type=param_schema.get("type", param.get("type", "string")),
            description=param.get("description", ""),
            example=param_schema.get("example", param.get("example"))
        ))
    
    # Parse request body (OpenAPI 3.x)
    request_body = None
    if is_openapi3 and "requestBody" in operation:
        rb = operation["requestBody"]
        content = rb.get("content", {})
        for content_type, media_type in content.items():
            request_body = ParsedRequestBody(
                content_type=content_type,
                required=rb.get("required", False),
                schema=media_type.get("schema", {}),
                example=media_type.get("example")
            )
            break  # Take first content type
    elif not is_openapi3:
        # Swagger 2.0: body parameter
        for param in all_params:
            if param.get("in") == "body":
                request_body = ParsedRequestBody(
                    content_type="application/json",
                    required=param.get("required", False),
                    schema=param.get("schema", {}),
                    example=param.get("example")
                )
                break
    
    # Parse responses
    responses = []
    for status_code, response in operation.get("responses", {}).items():
        if not isinstance(response, dict):
            continue
            
        try:
            code = int(status_code)
        except ValueError:
            code = 0  # default, e.g., for "default" response
            
        if is_openapi3:
            content = response.get("content", {})
            schema = {}
            content_type = "application/json"
            for ct, media_type in content.items():
                content_type = ct
                schema = media_type.get("schema", {})
                break
        else:
            schema = response.get("schema", {})
            content_type = "application/json"
            
        responses.append(ParsedResponse(
            status_code=code,
            description=response.get("description", ""),
            content_type=content_type,
            schema=schema
        ))
    
    return ParsedEndpoint(
        method=method,
        path=path,
        operation_id=operation.get("operationId", "unknown"),
        summary=operation.get("summary", ""),
        description=operation.get("description", ""),
        tags=operation.get("tags", []),
        parameters=parameters,
        request_body=request_body,
        responses=responses,
        security=operation.get("security", [])
    )


def get_path_parameters(path: str) -> List[str]:
    """Extract path parameter names from path template."""
    return re.findall(r"\{(\w+)\}", path)


def endpoint_to_tuple(endpoint: ParsedEndpoint) -> Tuple[str, str]:
    """Convert endpoint to (method, path) tuple for comparison."""
    return (endpoint.method, endpoint.path)


def spec_to_endpoint_set(spec: ParsedSpec) -> set:
    """Get set of (method, path) tuples from parsed spec."""
    return {endpoint_to_tuple(e) for e in spec.endpoints}
