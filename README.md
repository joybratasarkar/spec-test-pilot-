# 🚀 SpecTestPilot + Agent Lightning + GAM

**Complete AI-Powered API Testing System** with Microsoft Agent Lightning RL training, GAM intelligent memory, and multi-language test generation.

## ⚡ Key Features

- 🤖 **Professional AI Tester** - Thinks like human QA engineer
- 🌍 **Multi-Language Generation** - Python, JavaScript, Java, cURL
- ⚡ **Agent Lightning RL** - Microsoft Research implementation (arXiv:2508.03680)
- 🧠 **GAM Memory System** - Intelligent context with lossless storage (arXiv:2511.18423)
- 🏖️ **Sandbox Environment** - Safe, isolated execution
- 🔒 **Enterprise Security** - Multi-tenant isolation
- 📊 **Professional Test Coverage** - 8 categories of comprehensive testing
- 🎯 **Zero-Code Integration** - Works with any existing agent

## 🎯 What Makes This Special

**This is the first system that combines:**
1. **Microsoft Agent Lightning** - State-of-the-art RL for agents
2. **GAM Memory System** - Intelligent, lossless memory
3. **Multi-Language Testing** - Professional test generation
4. **Human-Like Testing** - AI that thinks like QA engineers

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 🌍 Generate multi-language tests (Python, JS, Java, cURL)
python demo_multi_language_tester.py

# ⚡ Train with Agent Lightning + GAM
python train_agent_lightning.py --epochs 5 --mock

# 🔬 Test complete integrated system
./run_complete_flow.sh

# 🎯 Multi-language API testing demonstration
./run_complete_api_testing_flow.sh

# 📋 Standard test generation
python run_agent.py examples/banking_api.yaml

# 🧪 Integration testing
python test_complete_system.py
```

## 🏗️ Detailed System Architecture Flow

### **🎯 High-Level Architecture Overview**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  SpecTestPilot  │───▶│ Agent Lightning  │───▶│  GAM Memory     │
│  (Your Agent)   │    │ (RL Training)    │    │  (Intelligence) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│    Sandbox      │    │ Trace Collection │    │ Tenant Scoping  │
│ (Safe Testing)  │    │ (Sidecar Design) │    │ (Multi-tenant)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **🔄 Complete System Data Flow**

```
                           📋 USER INPUT
                    ┌─────────────────────────┐
                    │ OpenAPI Specification   │
                    │ + Target Language       │
                    │ + Tenant Information    │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ⚡ AGENT LIGHTNING SERVER
            ┌─────────────────────────────────────────────┐
            │           🔍 Task Queue Manager              │
            │  ┌─────────────────────────────────────┐   │
            │  │ • Task validation & queuing         │   │
            │  │ • Tenant isolation enforcement      │   │
            │  │ • Resource allocation               │   │
            │  │ • Concurrent task management        │   │
            │  └─────────────────────────────────────┘   │
            │                     │                       │
            │                     ▼                       │
            │           🔍 Sidecar Monitor                │
            │  ┌─────────────────────────────────────┐   │
            │  │ • Non-intrusive trace collection    │   │
            │  │ • Real-time performance monitoring  │   │
            │  │ • Error detection & recovery        │   │
            │  │ • State-action-reward tracking      │   │
            │  └─────────────────────────────────────┘   │
            └─────────────────┬───────────────────────────┘
                              │
                              ▼
                  🤖 AI AGENT EXECUTION LAYER
    ┌─────────────────────────────────────────────────────────────────┐
    │                    🏖️ SANDBOX ENVIRONMENT                      │
    │                                                                 │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │              🌍 Multi-Language Tester                  │   │
    │  │                                                         │   │
    │  │  ┌─────────────────────────────────────────────────┐   │   │
    │  │  │           🧠 AI Professional Tester            │   │   │
    │  │  │                                                 │   │   │
    │  │  │  Step 1: Parse OpenAPI Specification           │   │   │
    │  │  │  ┌─────────────────────────────────────────┐   │   │   │
    │  │  │  │ • Extract endpoints & methods          │   │   │   │
    │  │  │  │ • Identify auth requirements           │   │   │   │
    │  │  │  │ • Parse request/response schemas       │   │   │   │
    │  │  │  │ • Detect parameter constraints         │   │   │   │
    │  │  │  └─────────────────────────────────────────┘   │   │   │
    │  │  │                     │                           │   │   │
    │  │  │                     ▼                           │   │   │
    │  │  │  Step 2: Professional Test Planning             │   │   │
    │  │  │  ┌─────────────────────────────────────────┐   │   │   │
    │  │  │  │ 😊 Happy Path: Normal operations       │   │   │   │
    │  │  │  │ 💥 Error Handling: Graceful failures   │   │   │   │
    │  │  │  │ 🔐 Authentication: Access control      │   │   │   │
    │  │  │  │ ⚖️ Authorization: Permission checks     │   │   │   │
    │  │  │  │ 🛡️ Input Validation: Bad data rejection │   │   │   │
    │  │  │  │ 🎯 Boundary Testing: Edge value limits  │   │   │   │
    │  │  │  │ 🔒 Security Testing: Injection attacks  │   │   │   │
    │  │  │  │ 🔄 Edge Cases: Unusual valid scenarios  │   │   │   │
    │  │  │  └─────────────────────────────────────────┘   │   │   │
    │  │  │                     │                           │   │   │
    │  │  │                     ▼                           │   │   │
    │  │  │  Step 3: Multi-Language Code Generation         │   │   │
    │  │  │  ┌─────────────────────────────────────────┐   │   │   │
    │  │  │  │ 🐍 Python (pytest)                     │   │   │   │
    │  │  │  │ 🟨 JavaScript (Jest/Mocha)              │   │   │   │
    │  │  │  │ ☕ Java (RestAssured/TestNG)            │   │   │   │
    │  │  │  │ 🌐 cURL (Shell scripts)                 │   │   │   │
    │  │  │  │ 📚 Documentation (TEST_PLAN.md)         │   │   │   │
    │  │  │  │ 📦 Package files (deps & setup)         │   │   │   │
    │  │  │  └─────────────────────────────────────────┘   │   │   │
    │  │  └─────────────────────────────────────────────────┘   │   │
    │  │                                                         │   │
    │  │               🔒 Sandbox Safety Layer                   │   │
    │  │  ┌─────────────────────────────────────────────────┐   │   │
    │  │  │ • Isolated temporary directories            │   │   │
    │  │  │ • Mock LLM provider (no external calls)     │   │   │
    │  │  │ • Deterministic test generation             │   │   │
    │  │  │ • Automatic resource cleanup                │   │   │
    │  │  │ • No main project directory pollution       │   │   │
    │  │  └─────────────────────────────────────────────────┘   │   │
    │  └─────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────┬───────────────────────────┘
                                          │
                                          ▼
                            📝 GAM MEMORY INTEGRATION
              ┌─────────────────────────────────────────────────────┐
              │             🧠 Session Management                   │
              │                                                     │
              │  ┌─────────────────────────────────────────────┐   │
              │  │ Session Lifecycle:                          │   │
              │  │ 1. start_session(tenant_id) → session_id   │   │
              │  │ 2. add_to_session(role, content, artifacts) │   │
              │  │ 3. track_tool_outputs(results)             │   │
              │  │ 4. end_session_with_memo(context, decisions)│   │
              │  └─────────────────────────────────────────────┘   │
              │                     │                               │
              │                     ▼                               │
              │          🎯 Dual Memory Creation                    │
              │  ┌─────────────────────┬─────────────────────────┐ │
              │  │  📚 Lossless Pages   │  📝 Contextual Memos     │ │
              │  │                     │                         │ │
              │  │ • Complete session  │ • Smart summaries       │ │
              │  │   transcripts       │ • Key decisions made    │ │
              │  │ • All tool outputs  │ • Technology choices    │ │
              │  │ • Generated files   │ • page_id pointers      │ │
              │  │ • Full context      │ • Searchable headers    │ │
              │  └─────────────────────┴─────────────────────────┘ │
              │                     │                               │
              │                     ▼                               │
              │            🔐 Tenant Isolation                     │
              │  ┌─────────────────────────────────────────────┐   │
              │  │ • Tenant-scoped data storage                │   │
              │  │ • Secure search filtering                   │   │
              │  │ • Cross-tenant data prevention              │   │
              │  │ • Compliance-ready separation               │   │
              │  └─────────────────────────────────────────────┘   │
              └─────────────────┬───────────────────────────────────┘
                                │
                                ▼
                    ⚡ REINFORCEMENT LEARNING PIPELINE
          ┌─────────────────────────────────────────────────────────────┐
          │                🔄 Trace Processing                          │
          │                                                             │
          │  ┌─────────────────────────────────────────────────────┐   │
          │  │           📊 Execution Trace Collection             │   │
          │  │                                                     │   │
          │  │  Raw Traces → RL Transitions:                      │   │
          │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
          │  │  │State_t  │→│Action_t │→│Reward_t │→│State_t+1│   │   │
          │  │  │         │ │         │ │         │ │         │   │   │
          │  │  │API_spec │ │test_gen │ │quality  │ │improved │   │   │
          │  │  │context  │ │decision │ │score    │ │agent    │   │   │
          │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
          │  └─────────────────────────────────────────────────────┘   │
          │                           │                                 │
          │                           ▼                                 │
          │              🧠 Credit Assignment Module                    │
          │  ┌─────────────────────────────────────────────────────┐   │
          │  │ Temporal Credit Assignment:                         │   │
          │  │                                                     │   │
          │  │ R_t = r_t + γ * R_{t+1}                            │   │
          │  │                                                     │   │
          │  │ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐           │   │
          │  │ │ 1.8 │←│ 1.5 │←│ 1.2 │←│ 0.8 │←│ 0.3 │ (Final)   │   │
          │  │ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘           │   │
          │  │    ▲       ▲       ▲       ▲       ▲               │   │
          │  │  Parse   Plan   Generate Code  Validate Quality    │   │
          │  │  API     Tests   Tests      Tests   Result         │   │
          │  └─────────────────────────────────────────────────────┘   │
          │                           │                                 │
          │                           ▼                                 │
          │                🎯 Neural Network Training                   │
          │  ┌─────────────────────────────────────────────────────┐   │
          │  │              LightningRL Algorithm                  │   │
          │  │                                                     │   │
          │  │ Experience Buffer: Store transitions for batch      │   │
          │  │ ┌─────────────────────────────────────────────┐     │   │
          │  │ │ [(s₁,a₁,r₁,s₁'), (s₂,a₂,r₂,s₂'), ...]      │     │   │
          │  │ └─────────────────────────────────────────────┘     │   │
          │  │                     │                               │   │
          │  │                     ▼                               │   │
          │  │ Value Network Architecture:                         │   │
          │  │ ┌─────────────────────────────────────────────┐     │   │
          │  │ │ Input(512) → Hidden(256) → Hidden(128) →   │     │   │
          │  │ │ Output(1)  [Value Estimation]              │     │   │
          │  │ └─────────────────────────────────────────────┘     │   │
          │  │                     │                               │   │
          │  │                     ▼                               │   │
          │  │ Training Process:                                   │   │
          │  │ • Sample mini-batch from experience buffer          │   │
          │  │ • Forward pass: predicted_values = network(states) │   │
          │  │ • Loss calculation: MSE(predicted, target_values)  │   │
          │  │ • Backpropagation: optimizer.step()               │   │
          │  │ • Policy improvement for better test generation    │   │
          │  └─────────────────────────────────────────────────────┘   │
          └─────────────────────────────────┬───────────────────────────┘
                                            │
                                            ▼
                              🔍 INTELLIGENT CONTEXT RETRIEVAL
                ┌─────────────────────────────────────────────────────┐
                │                🧠 GAM Deep Research                 │
                │                                                     │
                │  When agent needs context for better decisions:     │
                │                                                     │
                │  Step 1: PLAN                                      │
                │  ┌─────────────────────────────────────────────┐   │
                │  │ • Analyze current task requirements         │   │
                │  │ • Identify knowledge gaps                   │   │
                │  │ • Select appropriate search strategies      │   │
                │  │ • Plan multi-modal retrieval approach      │   │
                │  └─────────────────────────────────────────────┘   │
                │                     │                               │
                │                     ▼                               │
                │  Step 2: SEARCH                                    │
                │  ┌─────────────────────────────────────────────┐   │
                │  │ 🔍 BM25 Search: "OAuth security testing"    │   │
                │  │ 🧠 Vector Search: Semantic similarity       │   │
                │  │ 🔗 Direct ID: page_id lookup                │   │
                │  │ 🔐 Tenant Filter: Secure data isolation     │   │
                │  └─────────────────────────────────────────────┘   │
                │                     │                               │
                │                     ▼                               │
                │  Step 3: INTEGRATE                                 │
                │  ┌─────────────────────────────────────────────┐   │
                │  │ • Combine memo headers with lossless data   │   │
                │  │ • Follow page_id pointers for full context  │   │
                │  │ • Build comprehensive knowledge base        │   │
                │  │ • Create contextually perfect information   │   │
                │  └─────────────────────────────────────────────┘   │
                │                     │                               │
                │                     ▼                               │
                │  Step 4: REFLECT                                   │
                │  ┌─────────────────────────────────────────────┐   │
                │  │ • Assess information completeness           │   │
                │  │ • Validate context relevance               │   │
                │  │ • Determine if more search needed          │   │
                │  │ • Finalize optimal context compilation     │   │
                │  └─────────────────────────────────────────────┘   │
                └─────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
                        ✅ IMPROVED AGENT OUTPUT
                  ┌─────────────────────────────────────────┐
                  │        🚀 Enhanced Test Generation      │
                  │                                         │
                  │ • Smarter test scenarios               │
                  │ • Better security coverage             │
                  │ • More comprehensive edge cases        │
                  │ • Professional-quality patterns       │
                  │ • Context-aware test names             │
                  │ • Intelligent error detection          │
                  │                                         │
                  │ Result: Production-ready test suites   │
                  │ across multiple programming languages   │
                  └─────────────────────────────────────────┘
```

### **🔗 Component Interaction Details**

#### **1. Agent Lightning Server ⚡**
```python
# Server initialization and task processing
class AgentLightningServer:
    def __init__(self, max_workers=4):
        self.task_queue = TaskQueue()
        self.sidecar_monitor = SidecarMonitor()
        self.rl_algorithm = LightningRLAlgorithm()
        
    def process_task(self, task):
        # 1. Validate and queue task
        task_id = self.task_queue.enqueue(task)
        
        # 2. Start monitoring
        self.sidecar_monitor.start_monitoring(task_id)
        
        # 3. Execute agent (sandbox or real)
        result = self.execute_agent(task)
        
        # 4. Collect traces and train
        traces = self.sidecar_monitor.get_traces(task_id)
        self.rl_algorithm.train_on_traces(traces, result)
        
        return result
```

#### **2. Multi-Language Test Generator 🌍**
```python
# Professional test generation across languages
class MultiLanguageTestGenerator:
    def think_like_tester(self, api_spec):
        scenarios = []
        
        # Professional test categories
        for endpoint in self.parse_endpoints(api_spec):
            scenarios.extend(self.create_happy_path_tests(endpoint))
            scenarios.extend(self.create_security_tests(endpoint))
            scenarios.extend(self.create_auth_tests(endpoint))
            # ... 8 total categories
            
        return scenarios
    
    def generate_all_languages(self, scenarios):
        return {
            'python': self.generate_python_tests(scenarios),
            'javascript': self.generate_js_tests(scenarios), 
            'java': self.generate_java_tests(scenarios),
            'curl': self.generate_curl_tests(scenarios)
        }
```

#### **3. GAM Memory Integration 🧠**
```python
# Intelligent memory with session management
class GAMMemorySystem:
    def intelligent_session_flow(self, task):
        # Start session with tenant isolation
        session_id = self.start_session(tenant_id=task.tenant_id)
        
        # Track all interactions
        self.add_to_session(session_id, "user", task.description)
        self.add_to_session(session_id, "assistant", result.content, 
                          artifacts=result.generated_files)
        
        # Create dual memory: lossless + contextual
        lossless_pages, memo = self.end_session_with_memo(
            session_id, 
            title=task.title,
            decisions=["OAuth 2.0 PKCE", "pytest framework"],
            technologies=["Python", "REST API"]
        )
        
        return session_id, memo.id
```

#### **4. Reinforcement Learning Pipeline ⚡**
```python
# RL training on test generation quality
class LightningRLAlgorithm:
    def train_on_traces(self, traces, final_result):
        # Convert execution traces to RL transitions
        transitions = self.organize_trajectory(traces, final_result)
        
        # Apply temporal credit assignment
        rewards = self.credit_assignment.assign_credit(
            traces, final_result.quality_score, final_result.success
        )
        
        # Train neural network
        for transition, reward in zip(transitions, rewards):
            self.experience_buffer.add(transition, reward)
            
        # Update policy
        if len(self.experience_buffer) >= self.batch_size:
            batch = self.experience_buffer.sample(self.batch_size)
            loss = self.compute_loss(batch)
            self.optimizer.step()
```

## 🔄 Complete Training Flow

### **Step 1: Task Submission** 📋
```python
# Task submitted to Agent Lightning server
task = {
    "openapi_spec": "banking_api.yaml",
    "spec_title": "Banking API", 
    "tenant_id": "bank_corp"
}
```

### **Step 2: Sidecar Monitoring** 🔍
```python
# Non-intrusive trace collection starts
monitor.record_trace(task_id, TraceType.STATE, agent_id, initial_state)
monitor.record_trace(task_id, TraceType.ACTION, agent_id, action_data)
```

### **Step 3: Agent Execution** 🤖
```python
# SpecTestPilot runs in sandbox environment
session_id = gam.start_session(tenant_id="bank_corp")
result = sandbox_agent.execute(task)  # Safe execution
lossless_pages, memo = gam.end_session_with_memo(...)
```

### **Step 4: GAM Integration** 📝
```python
# Lossless storage with intelligent memos
memo_content = f"""
Context: {contextual_header}
Decisions: OAuth 2.0 PKCE; Bearer tokens
Full session data: page_id:{lossless_page.id}
"""
```

### **Step 5: RL Processing** ⚡
```python
# Convert traces to RL transitions
transitions = organizer.organize_trajectory(traces, reward, success)
# Each transition: (state_t, action_t, reward_t, state_t+1)
```

### **Step 6: Credit Assignment** 🧠
```python
# Distribute rewards across actions with temporal discount
rewards = credit_assignment.assign_credit(traces, final_reward, success)
# Backward propagation: R_t = r_t + γ * R_{t+1}
```

### **Step 7: Neural Network Training** 🎯
```python
# Update policy based on performance
loss = criterion(predicted_values, target_rewards)
optimizer.step()  # Agent learns and improves
```

### **Step 8: Next Iteration** 🔄
```python
# Improved agent performance for next task
# GAM provides smarter context from previous sessions
# Agent Lightning enables continuous learning
```

## 🧠 Dual AI Architecture

### **GAM Memory System** (arXiv:2511.18423)
- ✅ Lossless session storage + contextual memos
- ✅ Multi-tenant isolation  
- ✅ Deep research: PLAN → SEARCH → INTEGRATE → REFLECT
- ✅ Intelligent chunking + page_id pointers

### **Agent Lightning RL** (arXiv:2508.03680)  
- ✅ Sidecar monitoring with trace collection
- ✅ Credit assignment + hierarchical RL
- ✅ Training-agent disaggregation
- ✅ Zero-code integration with existing agents

### **Sandbox Environment** 🏖️
- ✅ Isolated file system operations
- ✅ Mock LLM responses for safe training
- ✅ Deterministic outputs for reproducible RL
- ✅ Automatic cleanup prevents directory pollution

### **Multi-Language Testing Agent** 🌍
- ✅ **Python (pytest)** - Backend testing teams
- ✅ **JavaScript (Jest)** - Frontend/Node.js teams
- ✅ **Java (RestAssured)** - Enterprise testing
- ✅ **cURL commands** - CI/CD pipeline integration
- ✅ **Professional documentation** - TEST_PLAN.md with setup instructions
- ✅ **Package files** - requirements.txt, package.json, pom.xml

## 🧠 How AI Thinks Like Professional Tester

### **8 Categories of Professional Testing:**

1. **😊 Happy Path** - What should work normally?
2. **💥 Error Handling** - What should fail gracefully?
3. **🔐 Authentication** - Are access controls working?
4. **⚖️ Authorization** - Can users access what they should?
5. **🛡️ Input Validation** - Are bad inputs rejected?
6. **🎯 Boundary Testing** - What are the limits?
7. **🔒 Security Testing** - Any vulnerabilities?
8. **🔄 Edge Cases** - Unusual but valid scenarios?

## 📁 Project Structure

```
spec_test_pilot/
├── graph.py                # Agent orchestration
├── parsers.py             # OpenAPI parsing  
├── schemas.py             # Data structures
├── agent_lightning.py     # Agent Lightning RL framework
└── memory/gam.py          # GAM memory system

train_agent_lightning.py   # RL training script
tests/                     # Test suite
examples/                  # Sample specs
```

## 🎯 RL Training

```bash
# Train with Agent Lightning + GAM
python train_agent_lightning.py \
    --epochs 10 \
    --data data/train.jsonl \
    --mock

# Features:
# - Non-intrusive trace collection
# - Hierarchical credit assignment  
# - GAM session integration
# - Multi-tenant training isolation
```

## 🔧 Standard Usage

```python
from spec_test_pilot.graph import run_agent

result = run_agent({
    "openapi_spec": "path/to/spec.yaml", 
    "output_format": "pytest"
})
```

## ⚡ Agent Lightning Usage

```python
from spec_test_pilot.memory.gam import GAMMemorySystem
from spec_test_pilot.agent_lightning import AgentLightningTrainer

# Initialize
gam = GAMMemorySystem()
trainer = AgentLightningTrainer(gam)

# Train
result = trainer.train_on_task(
    openapi_spec="examples/banking_api.yaml",
    spec_title="Banking API"
)
```

## 🏆 Research Papers Implemented

### **Microsoft Agent Lightning** (arXiv:2508.03680)
- ✅ Complete RL framework for ANY agent
- ✅ Sidecar design with trace collection
- ✅ Training-agent disaggregation
- ✅ Error monitoring and recovery

### **General Agentic Memory** (arXiv:2511.18423)
- ✅ Lossless memory with contextual intelligence
- ✅ Multi-modal retrieval system
- ✅ Session-based memory management
- ✅ Deep research loop implementation

**Result: State-of-the-art AI agent with RL training + intelligent memory! 🚀**

## 🎯 Production Deployment

Your system is **production-ready** with:

- 🔒 **Complete Security** - Multi-tenant isolation
- ⚡ **High Performance** - Optimized trace collection  
- 🧠 **Intelligent Memory** - Context-aware learning
- 🏖️ **Safe Testing** - Sandbox environment
- 📊 **Full Observability** - Training metrics & monitoring
- 🔄 **Continuous Learning** - RL-based agent improvement

## 📊 API Server (Optional)

```bash
# Run as web service
python api_server.py

# Use via HTTP API
curl -X POST localhost:8000/generate-tests \
  -H "Content-Type: application/json" \
  -d '{"openapi_spec": "path/to/spec.yaml"}'
```

## 🎉 Getting Started

1. **Clone and install:**
   ```bash
   git clone <your-repo>
   cd reinforcement-agent
   pip install -r requirements.txt
   ```

2. **Run the complete demo:**
   ```bash
   ./run_complete_api_testing_flow.sh
   ```

3. **Generate tests for your API:**
   ```bash
   python demo_multi_language_tester.py
   # Point it to your OpenAPI spec
   ```

4. **Train with your data:**
   ```bash
   python train_agent_lightning.py --epochs 10 --mock
   ```

**Your AI API testing agent is ready for production deployment! 🚀**
