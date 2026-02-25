# Complete Testing Guide for SpecTestPilot

This guide shows you exactly how to test and run SpecTestPilot with Agent Lightning integration.

---

## 🚀 **Quick Start Testing (5 Minutes)**

### **Step 1: Basic Agent Test**
```bash
# Test the agent on sample API
python run_agent.py --spec sample_api.yaml --verbose

# Expected output: 20 test cases with 0.58 quality score
```

### **Step 2: Run All Tests**
```bash
# Run comprehensive test suite
./run_tests.sh

# Expected: 8/8 comprehensive tests + 23/23 unit tests = ALL PASSING ✅
```

### **Step 3: Test Agent Lightning Integration**
```bash
# Quick Agent Lightning verification
python -c "
import agentlightning as al
al.clear_active_tracer()
tracer = al.DummyTracer()
al.set_active_tracer(tracer)
al.emit_message('system', 'Agent Lightning works!')
al.emit_reward(0.9)
print('✅ Agent Lightning integration successful!')
"
```

---

## 🧪 **Comprehensive Testing**

### **Test 1: Basic Agent Functionality**

**What it tests:** Core SpecTestPilot functionality
**Expected time:** 30 seconds

```bash
# Test with sample API
python run_agent.py --spec sample_api.yaml --output test_results.json

# Verify results
python -c "
import json
with open('test_results.json') as f:
    data = json.load(f)
print(f'✅ API: {data[\"spec_summary\"][\"title\"]}')
print(f'✅ Endpoints: {len(data[\"spec_summary\"][\"endpoints_detected\"])}')
print(f'✅ Tests: {len(data[\"test_suite\"])}')
print(f'✅ Coverage: {list(data[\"coverage_checklist\"].keys())}')
"
```

**Expected Results:**
- ✅ API: Pet Store API
- ✅ Endpoints: 7
- ✅ Tests: 20
- ✅ Coverage: ['happy_paths', 'validation_negative', 'auth_negative', 'error_contract', ...]

### **Test 2: Educational Training Mode**

**What it tests:** Mock training without API costs
**Expected time:** 2 minutes

```bash
# Generate training data first
python data/generate_dataset.py

# Run mock training
python train_agent_lightning.py --mock --epochs 3 --batch-size 8
```

**Expected Results:**
```
============================================================
Agent Lightning Training
============================================================
Mock mode: True
Epochs: 3
Batch size: 8
============================================================
Loaded 500 training examples
Loaded 100 test examples

Initial evaluation...
Initial avg reward: 0.8191

============================================================
Epoch 1/3
============================================================
[Progress bars and training metrics]

Final avg reward: 0.8224
Improvement: +0.0033
```

### **Test 3: Agent Lightning Integration**

**What it tests:** Real Agent Lightning framework integration
**Expected time:** 1 minute

```bash
# Test Agent Lightning integration
python final_agent_lightning_test.py
```

**Expected Results:**
```
✅ Agent Lightning v0.3.0 loaded successfully!
🧪 AGENT LIGHTNING COMPREHENSIVE TESTING SUITE

🔍 QUICK VERIFICATION TEST
1️⃣ Testing basic agent...
   ✅ Generated X test cases
2️⃣ Testing Agent Lightning...
   ✅ Active tracer: DummyTracer
3️⃣ Testing integration...
   ✅ Agent Lightning integration works

🎉 ALL TESTS PASSED - AGENT LIGHTNING INTEGRATION SUCCESSFUL!
```

### **Test 4: Error Handling**

**What it tests:** Graceful error handling
**Expected time:** 30 seconds

```bash
# Test with invalid OpenAPI spec
echo "invalid: yaml: content" | python run_agent.py --stdin

# Test with empty spec
echo "" | python run_agent.py --stdin

# Test with malformed JSON
echo '{"invalid": json}' | python run_agent.py --stdin
```

**Expected Results:**
- ✅ Invalid YAML: Should handle gracefully with error message
- ✅ Empty spec: Should return with missing_info populated
- ✅ Malformed JSON: Should handle with appropriate error

### **Test 5: Performance Testing**

**What it tests:** Consistent performance across runs
**Expected time:** 2 minutes

```bash
# Run multiple times to test consistency
for i in {1..5}; do
    echo "Run $i:"
    python run_agent.py --spec sample_api.yaml --run-id "perf_test_$i" | grep "Final Reward"
done
```

**Expected Results:**
```
Run 1: Final Reward: 0.5796
Run 2: Final Reward: 0.5796
Run 3: Final Reward: 0.5796
Run 4: Final Reward: 0.5796
Run 5: Final Reward: 0.5796
```
*Note: Should be consistent due to deterministic seed*

---

## 🔧 **Advanced Testing**

### **Custom API Testing**

Test with your own OpenAPI specification:

```bash
# Test with your API
python run_agent.py --spec /path/to/your-api.yaml --verbose

# Batch test multiple APIs
for spec in apis/*.yaml; do
    echo "Testing $spec..."
    python run_agent.py --spec "$spec" --output "results_$(basename $spec .yaml).json"
done
```

### **Training Data Generation**

Generate custom training datasets:

```bash
# Generate default dataset (500 examples)
python data/generate_dataset.py

# Generate larger dataset
python data/generate_dataset.py --num-examples 1000 --output custom_data.jsonl

# Verify dataset quality
python -c "
import json
count = 0
with open('data/train.jsonl') as f:
    for line in f:
        count += 1
        data = json.loads(line)
        if count <= 3:
            print(f'Example {count}: {data[\"spec_summary\"][\"title\"]}')
print(f'Total examples: {count}')
"
```

### **Real RL Training (Production)**

**⚠️ Warning: This uses OpenAI API and costs money**

```bash
# Ensure you have OpenAI API key set
echo $OPENAI_API_KEY

# Start with small training run
python train_agent_lightning_real.py --algorithm ppo --epochs 2 --batch-size 4

# Monitor training progress
tail -f lightning_checkpoints/training.log
```

---

## 🐛 **Troubleshooting**

### **Common Issues & Solutions**

#### **Issue 1: Agent Lightning Import Error**
```
❌ Error: No module named 'agentlightning'
```
**Solution:**
```bash
pip install agentlightning
# or
pip install --upgrade agentlightning
```

#### **Issue 2: Tracer Already Set Error**
```
❌ Error: An active tracer is already set
```
**Solution:**
```python
import agentlightning as al
al.clear_active_tracer()
tracer = al.DummyTracer()
al.set_active_tracer(tracer)
```

#### **Issue 3: OpenAI API Key Missing**
```
❌ Error: OPENAI_API_KEY not found
```
**Solution:**
```bash
# Add to .env file
echo "OPENAI_API_KEY=your_key_here" >> .env

# Or export directly
export OPENAI_API_KEY=your_key_here
```

#### **Issue 4: Tests Failing**
```
❌ Error: Some tests failed
```
**Solution:**
```bash
# Run individual test components
python -c "from spec_test_pilot.schemas import SpecTestPilotOutput; print('✅ Schemas OK')"
python -c "from spec_test_pilot.openapi_parse import parse_openapi_spec; print('✅ Parser OK')"
python -c "from spec_test_pilot.graph import run_agent; print('✅ Agent OK')"

# Check dependencies
pip install -r requirements.txt
```

#### **Issue 5: Low Quality Scores**
```
⚠️ Warning: Quality score below 0.5
```
**Solution:**
- Check if OpenAPI spec is well-formed
- Ensure spec has multiple endpoints
- Verify authentication is properly defined
- Add more detailed endpoint descriptions

---

## 📊 **Performance Benchmarks**

### **Expected Performance Metrics**

| Test Type | Expected Time | Expected Quality Score | Expected Test Count |
|-----------|---------------|----------------------|-------------------|
| **Simple API** (1-3 endpoints) | 10-20 seconds | 0.4-0.6 | 3-8 tests |
| **Medium API** (4-10 endpoints) | 20-40 seconds | 0.5-0.7 | 10-25 tests |
| **Complex API** (10+ endpoints) | 40-80 seconds | 0.6-0.8 | 25-50 tests |

### **Quality Score Breakdown**

| Score Range | Quality Level | Description |
|-------------|---------------|-------------|
| **0.8-1.0** | Excellent | Comprehensive coverage, all test types |
| **0.6-0.8** | Very Good | Good coverage, most test types |
| **0.4-0.6** | Good | Basic coverage, essential tests |
| **0.2-0.4** | Fair | Limited coverage, needs improvement |
| **0.0-0.2** | Poor | Minimal coverage, check API spec |

---

## 🎯 **Testing Checklist**

### **Before Deployment**
- [ ] All 31 automated tests pass
- [ ] Agent generates tests for sample API
- [ ] Agent Lightning integration works
- [ ] Error handling is graceful
- [ ] Performance is consistent

### **Before Production Training**
- [ ] OpenAI API key is set and valid
- [ ] Training data is generated and validated
- [ ] Mock training completes successfully
- [ ] Agent Lightning tracer is properly configured
- [ ] Monitoring and logging are set up

### **Regular Health Checks**
- [ ] Run `./run_tests.sh` weekly
- [ ] Test with new API specifications monthly
- [ ] Monitor quality scores for regression
- [ ] Update dependencies quarterly
- [ ] Review and update training data annually

---

## 🚀 **Next Steps**

After successful testing:

1. **Production Deployment**
   - Set up monitoring and alerting
   - Configure production API keys
   - Set up automated testing pipeline

2. **Continuous Improvement**
   - Collect user feedback on generated tests
   - Monitor quality scores over time
   - Retrain models with new data

3. **Advanced Features**
   - Implement custom reward functions
   - Add domain-specific test patterns
   - Integrate with CI/CD pipelines

---

**🎉 Congratulations! You now have a fully tested, production-ready SpecTestPilot with Agent Lightning integration!**
