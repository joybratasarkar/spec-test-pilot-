#!/usr/bin/env python3
"""
Dynamic Mock API Server - Automatically implements ANY OpenAPI spec.

This server dynamically creates endpoints based on the OpenAPI specification:
1. Reads OpenAPI spec (YAML/JSON)
2. Creates FastAPI routes for ALL endpoints automatically
3. Generates realistic responses based on schemas
4. Handles all HTTP methods (GET, POST, PUT, DELETE, PATCH)
5. Validates requests according to spec
6. Returns proper error codes

Usage:
    python dynamic_mock_server.py --spec any_api.yaml --port 8000
    
Works with ANY valid OpenAPI 3.0+ specification!
"""

import json
import yaml
import argparse
import time
import uvicorn
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, get_args
from datetime import datetime
import logging
import re

# FastAPI
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, create_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DynamicMockServer")


class DynamicResponseGenerator:
    """Generate realistic responses based on OpenAPI schemas."""
    
    @staticmethod
    def generate_from_schema(schema: Dict, operation_id: str = "", path: str = "") -> Any:
        """Generate data from OpenAPI schema."""
        if not schema:
            return {"message": "success", "timestamp": datetime.now().isoformat()}
        
        schema_type = schema.get("type", "object")
        
        if schema_type == "object":
            return DynamicResponseGenerator._generate_object(schema, operation_id, path)
        elif schema_type == "array":
            items_schema = schema.get("items", {})
            item = DynamicResponseGenerator.generate_from_schema(items_schema, operation_id, path)
            return [item] * 2  # Return 2 items
        elif schema_type == "string":
            return DynamicResponseGenerator._generate_string_value(schema, path)
        elif schema_type == "integer":
            return schema.get("example", 123)
        elif schema_type == "number":
            return schema.get("example", 12.34)
        elif schema_type == "boolean":
            return schema.get("example", True)
        else:
            return {"data": "generated"}
    
    @staticmethod
    def _generate_object(schema: Dict, operation_id: str, path: str) -> Dict:
        """Generate object from schema."""
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        result = {}
        
        # Generate required fields
        for field_name in required:
            if field_name in properties:
                field_schema = properties[field_name]
                result[field_name] = DynamicResponseGenerator.generate_from_schema(
                    field_schema, operation_id, path
                )
        
        # Generate some optional fields
        for field_name, field_schema in list(properties.items())[:5]:  # Limit to 5 fields
            if field_name not in result:
                result[field_name] = DynamicResponseGenerator.generate_from_schema(
                    field_schema, operation_id, path
                )
        
        # Add common fields based on context
        if "list" in operation_id.lower() or "get" in operation_id.lower():
            if "id" not in result:
                result["id"] = DynamicResponseGenerator._generate_id(path)
        
        # Add timestamps
        if not any(field in result for field in ["created_at", "updated_at", "timestamp"]):
            result["created_at"] = datetime.now().isoformat()
        
        return result
    
    @staticmethod
    def _generate_string_value(schema: Dict, path: str) -> str:
        """Generate string value based on schema and context."""
        # Check for enum values
        enum_values = schema.get("enum", [])
        if enum_values:
            return enum_values[0]
        
        # Check for examples
        example = schema.get("example")
        if example:
            return str(example)
        
        # Generate based on field name context
        path_lower = path.lower()
        
        if "email" in path_lower:
            return "test@example.com"
        elif "name" in path_lower:
            return "Test Name"
        elif "status" in path_lower:
            return "active"
        elif "url" in path_lower:
            return "https://example.com"
        elif "id" in path_lower:
            return "test_id_123"
        else:
            return "sample_value"
    
    @staticmethod
    def _generate_id(path: str) -> Union[int, str]:
        """Generate ID based on path context."""
        path_lower = path.lower()
        
        if "user" in path_lower:
            return 456
        elif "pet" in path_lower:
            return 123  
        elif "order" in path_lower:
            return 789
        else:
            return 1001


class DynamicMockServer:
    """Completely dynamic mock server that works with ANY OpenAPI spec."""
    
    def __init__(self, spec_file: str, host: str = "localhost", port: int = 8000):
        """Initialize dynamic mock server."""
        self.host = host
        self.port = port
        self.spec_file = spec_file
        
        # Load and parse OpenAPI spec
        self.spec = self._load_spec(spec_file)
        self.api_info = self.spec.get("info", {})
        self.paths = self.spec.get("paths", {})
        self.components = self.spec.get("components", {})
        self.security = self.spec.get("security", [])
        
        # Create FastAPI app
        self.app = FastAPI(
            title=self.api_info.get("title", "Dynamic Mock API"),
            description=self.api_info.get("description", "Automatically generated from OpenAPI spec"),
            version=self.api_info.get("version", "1.0.0"),
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request logging
        self.request_log = []
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_dynamic_routes()
        
        logger.info(f"🚀 Dynamic Mock Server initialized")
        logger.info(f"📋 API: {self.api_info.get('title', 'Unknown')} v{self.api_info.get('version', '1.0.0')}")
        logger.info(f"📍 Loaded {len(self.paths)} paths with {self._count_operations()} operations")
    
    def _load_spec(self, spec_file: str) -> Dict:
        """Load OpenAPI specification from file."""
        path = Path(spec_file)
        
        if not path.exists():
            raise FileNotFoundError(f"Spec file not found: {spec_file}")
        
        content = path.read_text()
        
        try:
            if spec_file.endswith(('.yaml', '.yml')):
                return yaml.safe_load(content)
            else:
                return json.loads(content)
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid spec file format: {e}")
    
    def _count_operations(self) -> int:
        """Count total operations across all paths."""
        count = 0
        for path_config in self.paths.values():
            for method in ["get", "post", "put", "patch", "delete", "head", "options"]:
                if method in path_config:
                    count += 1
        return count
    
    def _setup_middleware(self):
        """Setup request logging middleware."""
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            
            # Log request
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "ip": request.client.host if request.client else "unknown"
            }
            
            self.request_log.append(log_entry)
            logger.info(f"📥 {request.method} {request.url.path}")
            
            # Process request
            response = await call_next(request)
            
            # Add timing header
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
    
    def _setup_dynamic_routes(self):
        """Dynamically create routes from OpenAPI spec."""
        # Add health check
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "api": self.api_info.get("title", "Unknown API"),
                "version": self.api_info.get("version", "1.0.0"),
                "endpoints": self._count_operations(),
                "requests_handled": len(self.request_log)
            }
        
        # Create routes for each path and method
        for path, path_config in self.paths.items():
            if not isinstance(path_config, dict):
                continue
            path_level_parameters = path_config.get("parameters", [])
            for method, operation in path_config.items():
                if method.upper() in ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]:
                    self._create_dynamic_route(
                        path,
                        method.upper(),
                        operation,
                        path_level_parameters=path_level_parameters,
                    )
        
        logger.info(f"📍 Created {self._count_operations()} dynamic routes")
    
    def _create_dynamic_route(
        self,
        path: str,
        method: str,
        operation: Dict,
        path_level_parameters: Optional[List[Dict]] = None,
    ):
        """Create a single dynamic route."""
        operation_id = operation.get("operationId", f"{method.lower()}_{path.replace('/', '_').replace('{', '').replace('}', '')}")
        
        async def dynamic_handler(request: Request):
            return await self._handle_dynamic_request(
                request,
                path,
                method,
                operation,
                path_level_parameters=path_level_parameters,
            )
        
        # Register the route with FastAPI
        if method == "GET":
            self.app.add_api_route(path, dynamic_handler, methods=["GET"])
        elif method == "POST":
            self.app.add_api_route(path, dynamic_handler, methods=["POST"], status_code=201)
        elif method == "PUT":
            self.app.add_api_route(path, dynamic_handler, methods=["PUT"])
        elif method == "PATCH":
            self.app.add_api_route(path, dynamic_handler, methods=["PATCH"])
        elif method == "DELETE":
            self.app.add_api_route(path, dynamic_handler, methods=["DELETE"], status_code=204)
        
        logger.info(f"   📌 {method} {path} ({operation_id})")
    
    async def _handle_dynamic_request(
        self,
        request: Request,
        path: str,
        method: str,
        operation: Dict,
        path_level_parameters: Optional[List[Dict]] = None,
    ) -> Any:
        """Handle any dynamic request based on OpenAPI operation."""
        try:
            # Check authentication if required
            await self._check_auth_if_required(request, operation)
            
            # Validate path parameters
            path_params = self._extract_path_params(request.url.path, path)
            await self._validate_query_params(
                request=request,
                operation=operation,
                path_level_parameters=path_level_parameters,
            )
            
            # Handle special test cases (404s, etc.)
            await self._handle_special_cases(request, path, method, path_params)
            
            # Validate request body for POST/PUT/PATCH
            if method in ["POST", "PUT", "PATCH"]:
                body = await self._validate_request_body(request, operation)
            else:
                body = None
            
            # Generate response
            return await self._generate_dynamic_response(request, path, method, operation, path_params, body)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Error in {method} {path}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    def _collect_operation_parameters(
        self,
        operation: Dict,
        path_level_parameters: Optional[List[Dict]],
    ) -> List[Dict]:
        merged: Dict[tuple, Dict] = {}

        if isinstance(path_level_parameters, list):
            for param in path_level_parameters:
                if not isinstance(param, dict):
                    continue
                location = str(param.get("in", "")).lower()
                name = str(param.get("name", ""))
                if location and name:
                    merged[(location, name)] = param

        operation_params = operation.get("parameters", [])
        if isinstance(operation_params, list):
            for param in operation_params:
                if not isinstance(param, dict):
                    continue
                location = str(param.get("in", "")).lower()
                name = str(param.get("name", ""))
                if location and name:
                    merged[(location, name)] = param

        return list(merged.values())

    async def _validate_query_params(
        self,
        request: Request,
        operation: Dict,
        path_level_parameters: Optional[List[Dict]] = None,
    ) -> None:
        parameters = self._collect_operation_parameters(operation, path_level_parameters)
        query_specs = [
            p for p in parameters if str((p or {}).get("in", "")).lower() == "query"
        ]
        query_spec_map = {
            str((p or {}).get("name", "")): p
            for p in query_specs
            if str((p or {}).get("name", ""))
        }
        incoming = dict(request.query_params)

        if incoming and not query_spec_map:
            unknown = ", ".join(sorted(incoming.keys())[:5])
            raise HTTPException(
                status_code=400,
                detail=f"Unexpected query parameter(s): {unknown}",
            )

        for name in incoming.keys():
            if name not in query_spec_map:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unexpected query parameter: {name}",
                )

        for name, param in query_spec_map.items():
            if bool(param.get("required", False)) and name not in incoming:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required query parameter: {name}",
                )
            if name in incoming:
                self._validate_query_param_value(name, incoming[name], param)

    def _validate_query_param_value(self, name: str, raw_value: str, param: Dict) -> None:
        schema = param.get("schema", {})
        if not isinstance(schema, dict):
            return

        enum_values = schema.get("enum", [])
        if isinstance(enum_values, list) and enum_values:
            allowed = {str(v) for v in enum_values}
            if str(raw_value) not in allowed:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for query parameter '{name}': {raw_value}",
                )

        value_type = str(schema.get("type", "string")).lower()
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")

        if value_type == "integer":
            try:
                parsed = int(str(raw_value))
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=f"Query parameter '{name}' must be an integer",
                )
            self._validate_numeric_boundaries(
                name,
                parsed,
                minimum,
                maximum,
                exclusive_minimum,
                exclusive_maximum,
            )
            return

        if value_type == "number":
            try:
                parsed = float(str(raw_value))
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=f"Query parameter '{name}' must be a number",
                )
            self._validate_numeric_boundaries(
                name,
                parsed,
                minimum,
                maximum,
                exclusive_minimum,
                exclusive_maximum,
            )
            return

        if value_type == "boolean":
            if str(raw_value).lower() not in {"true", "false", "1", "0"}:
                raise HTTPException(
                    status_code=400,
                    detail=f"Query parameter '{name}' must be a boolean",
                )
            return

        if value_type == "string":
            text = str(raw_value)
            min_length = schema.get("minLength")
            max_length = schema.get("maxLength")
            if min_length is not None and len(text) < int(min_length):
                raise HTTPException(
                    status_code=400,
                    detail=f"Query parameter '{name}' is shorter than minLength={min_length}",
                )
            if max_length is not None and len(text) > int(max_length):
                raise HTTPException(
                    status_code=400,
                    detail=f"Query parameter '{name}' is longer than maxLength={max_length}",
                )
            return

    def _validate_numeric_boundaries(
        self,
        name: str,
        value: Union[int, float],
        minimum: Any,
        maximum: Any,
        exclusive_minimum: Any,
        exclusive_maximum: Any,
    ) -> None:
        if minimum is not None and value < float(minimum):
            raise HTTPException(
                status_code=400,
                detail=f"Query parameter '{name}' must be >= {minimum}",
            )
        if maximum is not None and value > float(maximum):
            raise HTTPException(
                status_code=400,
                detail=f"Query parameter '{name}' must be <= {maximum}",
            )
        if isinstance(exclusive_minimum, (int, float)):
            if value <= float(exclusive_minimum):
                raise HTTPException(
                    status_code=400,
                    detail=f"Query parameter '{name}' must be > {exclusive_minimum}",
                )
        elif exclusive_minimum is True and minimum is not None and value <= float(minimum):
            raise HTTPException(
                status_code=400,
                detail=f"Query parameter '{name}' must be > {minimum}",
            )
        if isinstance(exclusive_maximum, (int, float)):
            if value >= float(exclusive_maximum):
                raise HTTPException(
                    status_code=400,
                    detail=f"Query parameter '{name}' must be < {exclusive_maximum}",
                )
        elif exclusive_maximum is True and maximum is not None and value >= float(maximum):
            raise HTTPException(
                status_code=400,
                detail=f"Query parameter '{name}' must be < {maximum}",
            )
    
    async def _check_auth_if_required(self, request: Request, operation: Dict):
        """Check authentication if required by operation."""
        # Check if this operation requires auth
        security = operation.get("security", self.security)
        
        if not security:
            return  # No auth required
        
        auth_header = request.headers.get("authorization", "")
        
        if not auth_header:
            raise HTTPException(status_code=401, detail="Authorization header required")
        
        if not auth_header.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization format. Expected: Bearer <token>")
        
        token = auth_header.split(" ", 1)[1] if " " in auth_header else ""
        
        if token in ["invalid", "expired", ""]:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    def _extract_path_params(self, request_path: str, spec_path: str) -> Dict[str, str]:
        """Extract path parameters from request URL."""
        # Convert spec path to regex
        path_regex = spec_path
        param_names = []
        
        # Find all parameters {param}
        for match in re.finditer(r'\{([^}]+)\}', spec_path):
            param_name = match.group(1)
            param_names.append(param_name)
            path_regex = path_regex.replace(f'{{{param_name}}}', r'([^/]+)')
        
        # Match against request path
        match = re.match(f'^{path_regex}$', request_path)
        if match and param_names:
            return dict(zip(param_names, match.groups()))
        
        return {}
    
    async def _handle_special_cases(self, request: Request, path: str, method: str, path_params: Dict):
        """Handle special test cases for specific parameter values."""
        # Simulate 404 for specific test values
        for param_name, param_value in path_params.items():
            if param_value in ["999", "nonexistent", "invalid", "notfound"]:
                resource_name = param_name.replace("Id", "").replace("_id", "")
                raise HTTPException(
                    status_code=404,
                    detail=f"{resource_name.title()} not found: {param_value}"
                )
        
        # Simulate conflict for POST with specific values
        if method == "POST":
            try:
                body = await request.json()
                if body and body.get("name") == "duplicate":
                    raise HTTPException(status_code=409, detail="Resource already exists")
            except:
                pass
    
    async def _validate_request_body(self, request: Request, operation: Dict) -> Optional[Dict]:
        """Validate request body against OpenAPI schema."""
        request_body_spec = operation.get("requestBody", {})
        if not request_body_spec:
            return None
        
        # Get JSON content type schema
        content = request_body_spec.get("content", {})
        json_content = content.get("application/json", {})
        schema = json_content.get("schema", {})
        
        if not schema:
            return None
        
        try:
            body = await request.json()
        except:
            raise HTTPException(status_code=400, detail="Invalid JSON in request body")
        
        # Basic validation - check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in body or body[field] is None or body[field] == "":
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        return body
    
    async def _generate_dynamic_response(
        self, 
        request: Request, 
        path: str, 
        method: str, 
        operation: Dict, 
        path_params: Dict,
        body: Optional[Dict]
    ) -> Any:
        """Generate response dynamically based on operation definition."""
        responses = operation.get("responses", {})
        
        # Determine success status code
        if method == "POST":
            success_code = "201"
        elif method == "DELETE":
            success_code = "204"  
        else:
            success_code = "200"
        
        # Get success response schema
        success_response = responses.get(success_code, responses.get("200", {}))
        content = success_response.get("content", {})
        json_content = content.get("application/json", {})
        schema = json_content.get("schema", {})
        
        # Generate response data
        if schema:
            response_data = DynamicResponseGenerator.generate_from_schema(
                schema, 
                operation.get("operationId", ""),
                path
            )
        else:
            # Fallback response
            response_data = self._generate_fallback_response(method, path, path_params, body)
        
        # Add path parameters to response
        for param_name, param_value in path_params.items():
            if param_name.endswith("Id") or param_name.endswith("_id"):
                if isinstance(response_data, dict):
                    id_field = param_name if param_name in response_data else "id"
                    response_data[id_field] = int(param_value) if param_value.isdigit() else param_value
        
        # Handle DELETE (204 No Content)
        if method == "DELETE":
            return {"message": f"Resource deleted successfully"}
        
        return response_data
    
    def _generate_fallback_response(self, method: str, path: str, path_params: Dict, body: Optional[Dict]) -> Dict:
        """Generate fallback response when no schema is available."""
        base_response = {
            "message": f"{method} {path} executed successfully",
            "timestamp": datetime.now().isoformat()
        }
        
        # Add path params
        if path_params:
            base_response.update(path_params)
        
        # For list endpoints, return array
        if method == "GET" and not path_params:
            resource_name = path.strip("/").split("/")[0]
            base_response = {
                resource_name: [
                    {"id": 1, "name": "Sample Item 1"},
                    {"id": 2, "name": "Sample Item 2"}
                ],
                "total": 2,
                "page": 1
            }
        
        # For POST/PUT, echo back the body with an ID
        if method in ["POST", "PUT"] and body:
            base_response = {
                "id": 1001,
                **body,
                "message": f"Resource {method.lower()}ed successfully"
            }
        
        return base_response
    
    def run(self, debug: bool = True):
        """Start the dynamic mock server."""
        print(f"\n{'='*70}")
        print(f"🚀 Dynamic Mock API Server")
        print(f"{'='*70}")
        print(f"📋 API: {self.api_info.get('title', 'Unknown API')} v{self.api_info.get('version', '1.0.0')}")
        print(f"📁 Spec: {self.spec_file}")
        print(f"🌐 URL: http://{self.host}:{self.port}")
        print(f"📖 Docs: http://{self.host}:{self.port}/docs")
        print(f"📊 Endpoints: {self._count_operations()} operations across {len(self.paths)} paths")
        
        # Show all endpoints
        print(f"\n📍 Available Endpoints:")
        for path, path_config in self.paths.items():
            methods = []
            for method in ["get", "post", "put", "patch", "delete"]:
                if method in path_config:
                    methods.append(method.upper())
            if methods:
                print(f"   {', '.join(methods)} {path}")
        
        print(f"\n🧪 Authentication: Bearer tokens (any token except 'invalid')")
        print(f"🔧 Special test values: '999'/'nonexistent' → 404, 'duplicate' → 409")
        print(f"{'='*70}\n")
        
        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info" if debug else "warning"
            )
        except KeyboardInterrupt:
            logger.info("🛑 Server stopped by user")


def main():
    """CLI interface for dynamic mock server."""
    parser = argparse.ArgumentParser(
        description="Dynamic Mock API Server - Works with ANY OpenAPI spec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Pet Store API
    python dynamic_mock_server.py --spec sample_api.yaml --port 8000
    
    # Custom API
    python dynamic_mock_server.py --spec customer_api.yaml --port 9000
    
    # Banking API  
    python dynamic_mock_server.py --spec banking_api.yaml --port 8080
        """
    )
    
    parser.add_argument("--spec", "-s", required=True,
                       help="OpenAPI specification file (YAML or JSON)")
    parser.add_argument("--host", default="localhost",
                       help="Server host (default: localhost)")
    parser.add_argument("--port", "-p", type=int, default=8000,
                       help="Server port (default: 8000)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    try:
        # Create and start dynamic server
        server = DynamicMockServer(args.spec, args.host, args.port)
        server.run(debug=args.debug)
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Make sure the OpenAPI spec file exists")
    except ValueError as e:
        print(f"❌ {e}")
        print("💡 Check that the spec file is valid YAML or JSON")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")


if __name__ == "__main__":
    main()
