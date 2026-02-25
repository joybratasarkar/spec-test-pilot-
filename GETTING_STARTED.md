# Getting Started with SpecTestPilot

A complete beginner's guide to understanding and using SpecTestPilot.

---

## 🤔 What Does SpecTestPilot Agent Do?

**SpecTestPilot is an AI agent that automatically writes comprehensive test cases for your APIs.**

### Simple Explanation

Think of it like having a **super smart QA engineer** that:

1. **Reads your API documentation** (OpenAPI/Swagger files)
2. **Understands what your API does** (endpoints, parameters, authentication)
3. **Automatically writes 15-20 test cases** covering all scenarios
4. **Gives you perfect JSON output** ready to use in your testing framework

### Input → Output Example

**You give it this** (OpenAPI spec):
```yaml
paths:
  /users:
    get:
      summary: "Get all users"
    post:
      summary: "Create user"
      requestBody:
        required: true
```

**It gives you back this** (Test cases):
```json
{
  "test_suite": [
    {
      "test_id": "T001",
      "name": "GET /users happy path",
      "objective": "Verify users list returns 200",
      "assertions": [{"type": "status_code", "expected": 200}]
    },
    {
      "test_id": "T002", 
      "name": "POST /users missing auth",
      "objective": "Verify returns 401 without token",
      "assertions": [{"type": "status_code", "expected": 401}]
    }
  ]
}
```

### What Makes It Special

| Feature | Benefit |
|---------|---------|
| **🎯 No Hallucination** | Only tests endpoints that actually exist |
| **🧠 Smart Memory** | Remembers API testing best practices |
| **✅ Perfect Format** | Always outputs valid, structured JSON |
| **🔄 Self-Improving** | Gets better through reinforcement learning |

### Types of Tests It Creates

1. **Happy Path Tests** - Normal usage scenarios
2. **Error Tests** - Invalid inputs, missing data
3. **Auth Tests** - Missing tokens, wrong permissions  
4. **Edge Cases** - Boundary conditions, special characters

**Bottom line**: You spend 30 seconds running it, get 20 comprehensive test cases that would take hours to write manually!

---

## 🤔 What is SpecTestPilot?

**SpecTestPilot** is like having an AI assistant that reads your API documentation and automatically writes comprehensive test cases for you.

### The Problem It Solves

When you build an API, you need to test it thoroughly:
- ✅ Does `/users` endpoint work correctly?
- ✅ What happens when I send invalid data?
- ✅ Are authentication errors handled properly?
- ✅ Do all the documented endpoints actually exist?

Writing these tests manually is **boring, time-consuming, and error-prone**. SpecTestPilot does it automatically.

### What It Takes As Input

An **OpenAPI specification** (also called Swagger) - this is a YAML or JSON file that describes your API:

```yaml
# Example: pet-store-api.yaml
openapi: "3.0.3"
info:
  title: "Pet Store API"
  version: "1.0.0"
paths:
  /pets:
    get:
      summary: "List all pets"
      responses:
        "200":
          description: "A list of pets"
    post:
      summary: "Create a new pet"
      responses:
        "201":
          description: "Pet created"
```

### What It Gives You Back

A **comprehensive JSON test suite** with:
- Happy path tests (normal usage)
- Error tests (what happens when things go wrong)
- Authentication tests
- Validation tests (invalid data)

---

## 🏗️ How It Works (Simple Explanation)

Think of SpecTestPilot as a smart robot that follows these steps:

```
1. 📖 READ: "Let me read this API documentation..."
2. 🧠 THINK: "Based on what I know about APIs, what tests should I write?"
3. 🔍 RESEARCH: "Let me look up best practices for testing APIs like this..."
4. ✍️ WRITE: "Here are 20 comprehensive test cases!"
5. ✅ VALIDATE: "Let me double-check these tests are correct..."
```

### The "Brain" Behind It

SpecTestPilot uses several AI technologies:

- **LangGraph**: Orchestrates the thinking process (like a flowchart for AI)
- **GAM Memory**: Remembers best practices and patterns from previous work
- **Pydantic**: Ensures the output is always perfectly formatted
- **Reinforcement Learning**: Gets better over time by learning from feedback

---

## 🚀 Complete Step-by-Step Guide

### Step 1: Setup & Installation

```bash
# 1. Navigate to the project directory
cd /Users/sjoybrata/Desktop/reinforcement-agent

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Verify everything is installed
python3 -c "print('✅ Ready to go!')"

# 4. Check that all dependencies are working
python3 -c "from spec_test_pilot.schemas import SpecTestPilotOutput; print('✅ All modules loaded!')"
```

### Step 2: Run Your First Test (Sample API)

```bash
# Run SpecTestPilot on the included sample Pet Store API
python3 run_agent.py --spec sample_api.yaml --verbose

# This will take 30-60 seconds and show you:
# - Parsing the OpenAPI spec
# - Detecting 7 endpoints  
# - Running the research loop
# - Generating 15-20 test cases
# - Final reward score
```

**What you'll see in the terminal:**
```
============================================================
SpecTestPilot Agent Run
============================================================
Parsing OpenAPI spec...
✅ Found: Pet Store API v1.0.0
✅ Detected 7 endpoints
✅ Authentication: Bearer token required

Running deep research...
📋 Plan: Search for REST API testing conventions
🔍 Search: Found 4 relevant patterns  
🧠 Integrate: Combined research findings
💭 Reflect: Research complete

Generating test cases...
✅ Generated 20 comprehensive test cases
✅ Coverage: Happy paths ✓, Validation ✓, Auth ✓, Errors ✓

Final reward: 0.58 (Good quality)
```

### Step 3: Examine The Results

```bash
# The output is automatically saved to test_output.json
cat test_output.json | head -50

# Or view it in a more readable format
python3 -c "
import json
with open('test_output.json') as f:
    data = json.load(f)
print(f'API: {data[\"spec_summary\"][\"title\"]}')
print(f'Endpoints: {len(data[\"spec_summary\"][\"endpoints_detected\"])}')
print(f'Tests generated: {len(data[\"test_suite\"])}')
print(f'Coverage: {data[\"coverage_checklist\"]}')
"
```

### Step 4: Run All Tests (Verify Everything Works)

```bash
# Run the comprehensive test suite (takes 2-3 minutes)
./run_tests.sh

# This runs:
# - 8 comprehensive system tests
# - 23 unit tests with pytest  
# - Full agent run on sample API
# - All should pass ✅
```

### Step 5: Try With Your Own API

```bash
# Option A: From a file
python3 run_agent.py --spec /path/to/your-api.yaml --output my-tests.json --verbose

# Option B: From stdin (pipe input)
cat your-api.yaml | python3 run_agent.py --stdin --output my-tests.json

# Option C: Quick test without saving
python3 run_agent.py --spec your-api.yaml --verbose
```

### Step 6: Generate Training Data & Train (Optional)

```bash
# Generate synthetic training dataset (500 examples)
python3 data/generate_dataset.py

# Train in mock mode (free, no API calls)
python3 train_agent_lightning.py --mock --epochs 5 --batch-size 16

# Train with real OpenAI (requires API key, costs ~$1-5)
python3 train_agent_lightning.py --epochs 5 --batch-size 16
```

### Step 7: Advanced Usage

```bash
# Run on multiple specs
for spec in *.yaml; do
    echo "Processing $spec..."
    python3 run_agent.py --spec "$spec" --output "tests_${spec%.yaml}.json"
done

# Custom run with specific ID
python3 run_agent.py --spec api.yaml --run-id "my-test-001" --verbose

# Integration with your testing framework
python3 -c "
from spec_test_pilot.graph import run_agent
result = run_agent(open('sample_api.yaml').read())
print('Generated', len(result['output']['test_suite']), 'tests')
"
```

---

## 📋 What You'll Get (Example Output)

After running on the sample API, you'll get a JSON file with:

```json
{
  "spec_summary": {
    "title": "Pet Store API",
    "version": "1.0.0", 
    "base_url": "https://api.petstore.com/v1",
    "auth": {"type": "bearer", "details": "JWT token"},
    "endpoints_detected": [
      {"method": "GET", "path": "/pets", "operation_id": "listPets"},
      {"method": "POST", "path": "/pets", "operation_id": "createPet"},
      {"method": "GET", "path": "/pets/{petId}", "operation_id": "getPetById"}
    ]
  },
  "deep_research": {
    "plan": ["Search for REST API testing conventions", "Find auth patterns"],
    "memory_excerpts": [
      {"source": "convention", "excerpt": "Every endpoint needs happy path test"}
    ],
    "reflection": "Found 5 relevant testing patterns"
  },
  "test_suite": [
    {
      "test_id": "T001",
      "name": "GET /pets happy path",
      "endpoint": {"method": "GET", "path": "/pets"},
      "objective": "Verify listing pets returns 200 with valid data",
      "preconditions": ["Valid authentication"],
      "request": {
        "headers": {"Authorization": "Bearer <token>"},
        "query_params": {"limit": "10"},
        "body": {}
      },
      "assertions": [
        {"type": "status_code", "expected": 200},
        {"type": "schema", "expected": "PetList"}
      ],
      "notes": ""
    },
    {
      "test_id": "T002",
      "name": "GET /pets missing auth",
      "endpoint": {"method": "GET", "path": "/pets"},
      "objective": "Verify returns 401 without authentication",
      "preconditions": ["No authentication provided"],
      "request": {
        "headers": {"Content-Type": "application/json"},
        "query_params": {},
        "body": {}
      },
      "assertions": [
        {"type": "status_code", "expected": 401}
      ],
      "notes": "Tests authentication requirement"
    }
  ],
  "coverage_checklist": {
    "happy_paths": "true",
    "validation_negative": "true", 
    "auth_negative": "true",
    "error_contract": "true",
    "idempotency": "unknown",
    "pagination_filtering": "unknown",
    "rate_limit": "unknown"
  },
  "missing_info": []
}
```

---

## 📁 Understanding The Codebase

Here's what each file does in simple terms:

### Core Files (The Main Logic)

| File | What It Does | Think Of It As |
|------|--------------|----------------|
| `spec_test_pilot/schemas.py` | Defines the exact format of output | The "contract" - what the output must look like |
| `spec_test_pilot/openapi_parse.py` | Reads OpenAPI files | The "reader" - understands API documentation |
| `spec_test_pilot/graph.py` | The main AI workflow | The "brain" - orchestrates the thinking process |
| `spec_test_pilot/memory/gam.py` | Remembers best practices | The "memory" - learns from experience |
| `spec_test_pilot/reward.py` | Judges how good the output is | The "teacher" - grades the AI's work |

### Entry Points (How You Use It)

| File | What It Does | When To Use |
|------|--------------|-------------|
| `run_agent.py` | Run SpecTestPilot on one API spec | "I want to generate tests for my API" |
| `train_agent_lightning.py` | Train the AI to get better | "I want to improve the AI" |
| `test_all.py` | Test that everything works | "I want to make sure nothing is broken" |

### Data & Configuration

| File/Folder | What It Contains |
|-------------|------------------|
| `data/` | Training examples and dataset generator |
| `tests/` | Automated tests to verify the system works |
| `sample_api.yaml` | Example API for you to try |
| `requirements.txt` | List of software dependencies |
| `.env` | Your OpenAI API key (keep secret!) |

---

## 🎯 Common Use Cases

### Use Case 1: "I Have An API, Generate Tests"

```bash
# Put your OpenAPI spec in a file (my-api.yaml)
# Then run:
python3 run_agent.py --spec my-api.yaml --output my-tests.json --verbose

# You'll get comprehensive test cases in my-tests.json
```

### Use Case 2: "I Want To See What It Can Do"

```bash
# Use our sample API
python3 run_agent.py --spec sample_api.yaml --verbose

# Watch it work step by step
```

### Use Case 3: "I Want To Test Everything"

```bash
# Run all tests to make sure everything works
./run_tests.sh

# This runs 31 different tests
```

### Use Case 4: "I Want To Train It To Be Better"

```bash
# Generate training data
python3 data/generate_dataset.py

# Train in mock mode (free, no API calls)
python3 train_agent_lightning.py --mock --epochs 5

# Train with real OpenAI (costs money, but better results)
python3 train_agent_lightning.py --epochs 5
```

---

## 🔧 Configuration Options

### Running The Agent

```bash
# Basic usage
python3 run_agent.py --spec my-api.yaml

# With verbose output (see the thinking process)
python3 run_agent.py --spec my-api.yaml --verbose

# Save output to file
python3 run_agent.py --spec my-api.yaml --output tests.json

# Read from stdin (pipe input)
cat my-api.yaml | python3 run_agent.py --stdin
```

### Training Options

```bash
# Mock mode (free, uses fake AI)
python3 train_agent_lightning.py --mock --epochs 10

# Real mode (uses OpenAI, costs money)
python3 train_agent_lightning.py --epochs 10 --batch-size 16

# Custom dataset
python3 train_agent_lightning.py --train-data my-train.jsonl --test-data my-test.jsonl
```

---

## 🧪 What Makes A Good Test Suite?

SpecTestPilot generates tests that cover:

### 1. Happy Paths ✅
- Normal usage scenarios
- Valid inputs and expected outputs
- "Does it work when used correctly?"

### 2. Validation Tests ❌
- Invalid data types
- Missing required fields
- "What happens with bad input?"

### 3. Authentication Tests 🔐
- Missing auth tokens
- Invalid credentials
- "Is security working?"

### 4. Error Handling Tests ⚠️
- Non-existent resources (404)
- Server errors (500)
- "Are errors handled gracefully?"

### Example Test Case

```json
{
  "test_id": "T001",
  "name": "POST /pets missing required field",
  "endpoint": {"method": "POST", "path": "/pets"},
  "objective": "Verify API returns 400 when name field is missing",
  "request": {
    "headers": {"Content-Type": "application/json"},
    "body": {"status": "available"}  // Missing "name" field
  },
  "assertions": [
    {"type": "status_code", "expected": 400},
    {"type": "field", "path": "$.error", "rule": "exists"}
  ],
  "notes": "Tests required field validation"
}
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'pydantic'"

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### "OpenAI API error: Invalid API key"

```bash
# Check your .env file has the correct key
cat .env

# Should show: OPENAI_API_KEY=sk-...
# If not, edit .env and add your key
```

### "No training data found"

```bash
# Generate the dataset first
python3 data/generate_dataset.py

# This creates data/train.jsonl and data/test.jsonl
```

### "Tests are failing"

```bash
# Run the comprehensive test suite
python3 test_all.py

# This will show you exactly what's broken
```

---

## 🎓 Learning More

### Understanding The Output

The JSON output has these main sections:

1. **`spec_summary`**: What the AI understood about your API
2. **`deep_research`**: What best practices it found
3. **`test_suite`**: The actual test cases (this is what you want!)
4. **`coverage_checklist`**: What types of tests were included
5. **`missing_info`**: What information was missing from your spec

### Key Concepts

- **OpenAPI/Swagger**: Standard format for describing APIs
- **Test Case**: A single test scenario with input and expected output
- **Assertion**: A check that verifies the response is correct
- **Happy Path**: Normal usage scenario
- **Negative Test**: Error scenario testing

### Advanced Usage

Once you're comfortable with the basics:

1. **Custom Training**: Train on your own API specs
2. **Integration**: Use the output in your testing framework
3. **Batch Processing**: Process multiple API specs at once
4. **Custom Prompts**: Modify the AI's behavior

---

## 🤝 Getting Help

### Quick Checks

1. **Is everything installed?** Run `./run_tests.sh`
2. **Is the sample working?** Run `python3 run_agent.py --spec sample_api.yaml`
3. **Are dependencies OK?** Run `pip list | grep pydantic`

### Common Questions

**Q: How much does it cost to run?**
A: Mock mode is free. Real mode uses OpenAI API (~$0.01-0.10 per API spec)

**Q: Can I use it without OpenAI?**
A: Yes! Use `--mock` flag for training and testing

**Q: What if my API spec is incomplete?**
A: SpecTestPilot handles this gracefully and tells you what's missing

**Q: Can I customize the test generation?**
A: Yes, you can modify the prompts in `graph.py` or train with custom data

---

## 🎉 You're Ready!

You now understand:
- ✅ What SpecTestPilot does (generates API tests)
- ✅ How to run it (`python3 run_agent.py --spec my-api.yaml`)
- ✅ What the output looks like (JSON test suite)
- ✅ How to troubleshoot common issues
- ✅ Where each piece of code fits in

**Next steps:**
1. Try it on the sample API
2. Try it on your own API spec
3. Explore the generated test cases
4. Consider training it on your specific domain

Happy testing! 🚀
