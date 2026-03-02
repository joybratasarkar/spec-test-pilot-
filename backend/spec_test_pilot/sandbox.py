#!/usr/bin/env python3
"""
Sandbox Environment for Agent Lightning + GAM Training

Provides:
- Safe mock execution environment
- No external API calls
- Deterministic responses for testing
- Isolated file system operations
- Mock LLM responses for training
"""

import json
import random
import time
import tempfile
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class MockLLMProvider:
    """Mock LLM that provides deterministic responses for training."""
    
    def __init__(self, seed: int = 42):
        """Initialize mock LLM with seed for reproducibility."""
        self.seed = seed
        self.call_count = 0
        random.seed(seed)
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate mock response based on prompt patterns."""
        self.call_count += 1
        
        # Use prompt hash for deterministic responses
        prompt_hash = hash(prompt.lower()) % 1000
        rng = random.Random(prompt_hash + self.seed)
        
        # Analyze prompt to determine response type
        prompt_lower = prompt.lower()
        
        if "generate test" in prompt_lower or "create test" in prompt_lower:
            return self._generate_mock_tests(rng, context)
        elif "plan" in prompt_lower:
            return self._generate_mock_plan(rng)
        elif "parse" in prompt_lower or "analyze" in prompt_lower:
            return self._generate_mock_analysis(rng, context)
        elif "reflect" in prompt_lower or "evaluate" in prompt_lower:
            return self._generate_mock_reflection(rng)
        else:
            return self._generate_generic_response(rng)
    
    def _generate_mock_tests(self, rng: random.Random, context: Dict[str, Any] = None) -> str:
        """Generate mock test cases."""
        api_name = context.get("spec_title", "API") if context else "MockAPI"
        endpoint_count = rng.randint(3, 8)
        test_count = endpoint_count * rng.randint(2, 4)
        
        test_template = '''
import pytest
import requests

def test_{endpoint}_success():
    """Test {endpoint} endpoint success case."""
    response = requests.get("https://api.example.com/{endpoint}")
    assert response.status_code == 200
    assert "data" in response.json()

def test_{endpoint}_auth_required():
    """Test {endpoint} requires authentication."""
    response = requests.get("https://api.example.com/{endpoint}")
    assert response.status_code == 401

def test_{endpoint}_validation():
    """Test {endpoint} input validation."""
    response = requests.post("https://api.example.com/{endpoint}", json={{}})
    assert response.status_code == 400
'''
        
        endpoints = [f"endpoint_{i}" for i in range(1, endpoint_count + 1)]
        tests = []
        
        for endpoint in endpoints:
            tests.append(test_template.format(endpoint=endpoint))
        
        result = {
            "success": True,
            "generated_tests": "\n".join(tests),
            "test_count": test_count,
            "endpoint_count": endpoint_count,
            "api_name": api_name
        }
        
        return json.dumps(result)
    
    def _generate_mock_plan(self, rng: random.Random) -> str:
        """Generate mock planning response."""
        actions = [
            "Parse OpenAPI specification",
            "Identify authentication methods", 
            "Extract endpoint definitions",
            "Generate positive test cases",
            "Generate negative test cases",
            "Create validation tests"
        ]
        
        selected_actions = rng.sample(actions, rng.randint(3, 5))
        
        plan = {
            "actions": selected_actions,
            "estimated_tests": rng.randint(10, 25),
            "complexity": rng.choice(["low", "medium", "high"])
        }
        
        return json.dumps(plan)
    
    def _generate_mock_analysis(self, rng: random.Random, context: Dict[str, Any] = None) -> str:
        """Generate mock analysis response."""
        analysis = {
            "endpoints_found": rng.randint(5, 15),
            "auth_methods": rng.choice([["bearer"], ["api_key"], ["oauth2"], ["bearer", "api_key"]]),
            "complexity_score": rng.uniform(0.3, 0.9),
            "estimated_coverage": f"{rng.randint(80, 95)}%"
        }
        
        return json.dumps(analysis)
    
    def _generate_mock_reflection(self, rng: random.Random) -> str:
        """Generate mock reflection response."""
        reflection = {
            "quality_score": rng.uniform(0.6, 0.95),
            "improvements": rng.sample([
                "Add more edge case tests",
                "Improve error handling tests", 
                "Add performance tests",
                "Enhance security tests"
            ], rng.randint(1, 3)),
            "confidence": rng.uniform(0.7, 0.9)
        }
        
        return json.dumps(reflection)
    
    def _generate_generic_response(self, rng: random.Random) -> str:
        """Generate generic mock response."""
        response = {
            "status": "completed",
            "confidence": rng.uniform(0.5, 0.9),
            "metadata": {
                "processing_time": rng.uniform(0.1, 2.0),
                "tokens_used": rng.randint(100, 500)
            }
        }
        
        return json.dumps(response)


class SandboxFileSystem:
    """Isolated file system for sandbox operations."""
    
    def __init__(self):
        """Initialize sandbox with temporary directory."""
        self.sandbox_dir = Path(tempfile.mkdtemp(prefix="agent_lightning_sandbox_"))
        self.created_files = []
    
    def write_file(self, filename: str, content: str) -> str:
        """Write file in sandbox directory."""
        file_path = self.sandbox_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        self.created_files.append(str(file_path))
        return str(file_path)
    
    def read_file(self, filename: str) -> str:
        """Read file from sandbox directory."""
        file_path = self.sandbox_dir / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                return f.read()
        return ""
    
    def list_files(self) -> List[str]:
        """List all files in sandbox."""
        return [str(f.relative_to(self.sandbox_dir)) for f in self.sandbox_dir.rglob("*") if f.is_file()]
    
    def cleanup(self):
        """Clean up sandbox directory."""
        import shutil
        try:
            shutil.rmtree(self.sandbox_dir)
        except Exception:
            pass  # Best effort cleanup


class MockSpecTestPilotAgent:
    """Mock SpecTestPilot agent for safe sandbox execution."""
    
    def __init__(self, sandbox_fs: SandboxFileSystem, mock_llm: MockLLMProvider):
        """Initialize mock agent."""
        self.sandbox_fs = sandbox_fs
        self.mock_llm = mock_llm
        self.execution_count = 0
        
        # Import multi-language tester
        try:
            from .multi_language_tester import APITestingSandbox
            self.has_multi_language = True
        except ImportError:
            self.has_multi_language = False
    
    def run_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock agent execution that simulates SpecTestPilot."""
        self.execution_count += 1
        
        # Simulate processing time
        processing_time = random.uniform(0.1, 0.5)
        time.sleep(processing_time)
        
        # Extract input parameters (enhanced with Postman-like capabilities)
        openapi_spec = input_data.get("openapi_spec", "")
        spec_title = input_data.get("spec_title", "Mock API")
        output_format = input_data.get("output_format", "pytest")
        nlp_prompt = input_data.get("nlp_prompt")  # New: Natural language prompt
        enable_error_fixing = input_data.get("enable_error_fixing", False)  # New: Error fixing
        enable_workflow_chains = input_data.get("enable_workflow_chains", False)  # New: Workflows
        
        # Simulate different success rates for RL training
        success_probability = min(0.1 + (self.execution_count * 0.01), 0.8)
        success = random.random() < success_probability
        
        if success:
            # Try multi-language testing if available
            if self.has_multi_language and openapi_spec and openapi_spec != "":
                try:
                    # Create temporary spec file
                    spec_content = f"""
openapi: 3.0.0
info:
  title: {spec_title}
  version: 1.0.0
paths:
  /api/test:
    get:
      summary: Test endpoint
      responses:
        '200':
          description: Success
"""
                    spec_filename = f"{spec_title.lower().replace(' ', '_')}_spec.yaml"
                    spec_path = self.sandbox_fs.write_file(spec_filename, spec_content)
                    
                    # Use multi-language tester
                    from .multi_language_tester import APITestingSandbox
                    temp_base_url = "https://api.example.com"
                    
                    # Create mini tester (simplified for sandbox)
                    tester_sandbox = APITestingSandbox(spec_path, temp_base_url)
                    ml_results = tester_sandbox.run_full_test_suite()
                    
                    # Copy generated files to our sandbox
                    import shutil
                    from pathlib import Path
                    ml_sandbox_dir = Path(ml_results["sandbox_directory"])
                    
                    files_copied = []
                    for file_path in ml_sandbox_dir.glob("*"):
                        if file_path.is_file():
                            dest_name = f"ml_{file_path.name}"
                            content = file_path.read_text()
                            self.sandbox_fs.write_file(dest_name, content)
                            files_copied.append(dest_name)
                    
                    # Cleanup temp tester
                    tester_sandbox.cleanup()
                    
                    return {
                        "success": True,
                        "generated_tests": f"Multi-language test suite with {ml_results['scenarios_generated']} scenarios",
                        "test_count": ml_results['scenarios_generated'],
                        "endpoint_count": ml_results['total_endpoints'],
                        "processing_time": processing_time,
                        "output_files": self.sandbox_fs.list_files(),
                        "execution_id": self.execution_count,
                        "multi_language_files": files_copied,
                        "languages_supported": ["python", "javascript", "java", "curl"]
                    }
                    
                except Exception as e:
                    # Fall back to regular mock testing
                    pass
            
            # Regular mock test output
            context = {
                "spec_title": spec_title,
                "output_format": output_format
            }
            
            # Simulate LLM calls
            analysis_response = self.mock_llm.generate_response(
                f"Analyze OpenAPI spec for {spec_title}", context
            )
            analysis = json.loads(analysis_response)
            
            test_response = self.mock_llm.generate_response(
                f"Generate {output_format} tests for {spec_title}", context
            )
            test_result = json.loads(test_response)
            
            # Write output files to sandbox
            if test_result.get("generated_tests"):
                test_filename = f"{spec_title.lower().replace(' ', '_')}_tests.py"
                self.sandbox_fs.write_file(test_filename, test_result["generated_tests"])
            
            return {
                "success": True,
                "generated_tests": test_result.get("generated_tests", ""),
                "test_count": test_result.get("test_count", 0),
                "endpoint_count": analysis.get("endpoints_found", 0),
                "processing_time": processing_time,
                "output_files": self.sandbox_fs.list_files(),
                "execution_id": self.execution_count
            }
        else:
            # Simulate failure cases
            error_types = [
                "Invalid OpenAPI specification",
                "Network timeout during processing",
                "Unsupported authentication method",
                "Parsing error in endpoint definitions"
            ]
            
            error = random.choice(error_types)
            
            return {
                "success": False,
                "error": error,
                "processing_time": processing_time,
                "execution_id": self.execution_count,
                "partial_results": {
                    "endpoints_analyzed": random.randint(0, 3)
                }
            }


class AgentLightningSandbox:
    """Complete sandbox environment for Agent Lightning testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize sandbox environment."""
        self.seed = seed
        self.mock_llm = MockLLMProvider(seed)
        self.sandbox_fs = SandboxFileSystem()
        self.mock_agent = MockSpecTestPilotAgent(self.sandbox_fs, self.mock_llm)
        
        # Track sandbox state
        self.executions = []
        self.start_time = time.time()
    
    def execute_agent_task(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent task in sandbox environment."""
        execution_start = time.time()
        
        try:
            result = self.mock_agent.run_agent(input_data)
            
            execution_info = {
                "input": input_data,
                "result": result,
                "execution_time": time.time() - execution_start,
                "timestamp": time.time(),
                "sandbox_files": self.sandbox_fs.list_files()
            }
            
            self.executions.append(execution_info)
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Sandbox execution error: {str(e)}",
                "execution_id": len(self.executions) + 1
            }
            
            self.executions.append({
                "input": input_data,
                "result": error_result,
                "execution_time": time.time() - execution_start,
                "timestamp": time.time(),
                "error": str(e)
            })
            
            return error_result
    
    def get_sandbox_stats(self) -> Dict[str, Any]:
        """Get sandbox execution statistics."""
        if not self.executions:
            return {"total_executions": 0}
        
        successful_runs = [e for e in self.executions if e["result"].get("success", False)]
        failed_runs = [e for e in self.executions if not e["result"].get("success", False)]
        
        avg_execution_time = sum(e["execution_time"] for e in self.executions) / len(self.executions)
        
        return {
            "total_executions": len(self.executions),
            "successful_runs": len(successful_runs),
            "failed_runs": len(failed_runs),
            "success_rate": len(successful_runs) / len(self.executions) if self.executions else 0.0,
            "avg_execution_time": avg_execution_time,
            "total_sandbox_time": time.time() - self.start_time,
            "files_created": len(self.sandbox_fs.created_files),
            "sandbox_directory": str(self.sandbox_fs.sandbox_dir)
        }
    
    def cleanup(self):
        """Clean up sandbox environment."""
        self.sandbox_fs.cleanup()


# Integration function for Agent Lightning
def create_sandbox_agent_function(sandbox: AgentLightningSandbox):
    """Create agent function that works with Agent Lightning in sandbox."""
    
    async def sandbox_agent_function(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sandbox agent function for Agent Lightning."""
        return sandbox.execute_agent_task(input_data)
    
    return sandbox_agent_function


def create_sandbox_reward_function():
    """Create reward function optimized for sandbox training."""
    
    def sandbox_reward_function(task, result: Dict[str, Any], execution_time: float) -> float:
        """Reward function for sandbox environment."""
        
        if not result.get("success", False):
            # Small reward for attempting, more for partial results
            base_reward = 0.1
            if result.get("partial_results"):
                base_reward += 0.2
            return base_reward
        
        reward = 1.0
        
        # Reward for test generation
        test_count = result.get("test_count", 0)
        if test_count > 0:
            reward += min(test_count * 0.1, 1.0)  # Cap bonus at 1.0
        
        # Reward for endpoint coverage
        endpoint_count = result.get("endpoint_count", 0)
        if endpoint_count > 0 and test_count > 0:
            coverage_ratio = min(test_count / endpoint_count, 3.0)  # Allow up to 3x tests per endpoint
            reward += coverage_ratio * 0.3
        
        # Time efficiency bonus
        if execution_time < 0.3:  # Fast execution in sandbox
            reward += 0.5
        elif execution_time > 1.0:  # Slow execution penalty
            reward -= 0.3
        
        # File creation bonus
        if result.get("output_files"):
            reward += len(result["output_files"]) * 0.1
        
        return max(0.1, reward)
    
    return sandbox_reward_function


if __name__ == "__main__":
    # Test sandbox environment
    print("🏖️  TESTING AGENT LIGHTNING SANDBOX")
    print("=" * 40)
    
    sandbox = AgentLightningSandbox(seed=42)
    
    # Test task
    test_task = {
        "openapi_spec": "examples/banking_api.yaml",
        "spec_title": "Banking API",
        "output_format": "pytest"
    }
    
    print("🧪 Executing test tasks...")
    for i in range(5):
        result = sandbox.execute_agent_task(test_task)
        print(f"Task {i+1}: {'✅ Success' if result.get('success') else '❌ Failed'} "
              f"(Tests: {result.get('test_count', 0)})")
    
    print()
    print("📊 SANDBOX STATISTICS:")
    stats = sandbox.get_sandbox_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print()
    print("🧹 Cleaning up...")
    sandbox.cleanup()
    print("✅ Sandbox test complete!")
