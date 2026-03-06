# QA Specialist Execution Report

## Run Metadata
- Spec: `E-commerce API` (1.0.0)
- Spec Path: `/Users/sjoybrata/.spec_test_pilot/customer_default/ecommerce/openapi_ecommerce.yaml`
- Tenant: `customer_default`
- Workspace: `customer_default`
- Run ID: `c65b1d6e310b4d00b8dd72bcf6558968`
- Spec Key: `spec_c3ebc1397f69182e`
- Isolation: `live_http_base_url`
- Environment Profile: `staging`
- LLM Scenario Debug Log: `/Users/sjoybrata/Desktop/reinforcement-agent/.qa_specialist_workspace/llm_scenario_debug.jsonl`
- Runtime Cap (sec): `None`
- Runtime Cap Hit: `False`
- Runtime Skipped: `0`
- Unsafe Actions Blocked: `0`
- LLM Token Cap: `None`
- Stage Metrics (ms): `{'stage_1_spec_intelligence': 3.508, 'stage_2_gam_memory_research': 8285.375, 'stage_3_scenario_generation': 39701.74, 'stage_4_mutation_selection': 38.025, 'stage_6_execute_verify': 50.059, 'stage_7_reward_training': 8.673, 'stage_8_reporting_rl': 29.433}`
- Generated At: `2026-03-05 10:49:32`
- Execution Time: `49.137s`

## Summary
- Total Scenarios: `16`
- Passed: `0`
- Failed: `16`
- Suspect: `0`
- Blocked: `0`
- Pass Rate: `0.0`
- True Pass Rate: `0.0`
- Quality Gate (0.7): `False`
- Quality Gate Fail Reasons: `['pass_rate_below_threshold']`
- Avg Duration: `0.675 ms`
- Contract Checks Run: `16`
- Contract Check Failures: `0`
- Corrected Expectations: `0`

## Test Type Breakdown
- `happy_path`: total=6, passed=0, failed=6, suspect=0, blocked=0
- `authentication`: total=3, passed=0, failed=3, suspect=0, blocked=0
- `input_validation`: total=1, passed=0, failed=1, suspect=0, blocked=0
- `error_handling`: total=2, passed=0, failed=2, suspect=0, blocked=0
- `boundary_testing`: total=2, passed=0, failed=2, suspect=0, blocked=0
- `performance`: total=1, passed=0, failed=1, suspect=0, blocked=0
- `security`: total=1, passed=0, failed=1, suspect=0, blocked=0

## Generated Script Execution
- Script Kind: `python_pytest`
- Status: `executed`
- Executed: `True`
- Script Path: `/Users/sjoybrata/Desktop/reinforcement-agent/.qa_specialist_workspace/generated_tests/test_api.py`
- Total Tests: `16`
- Passed Tests: `0`
- Failed Tests: `16`
- Pass Rate: `0.0`

## Learning Loop
- Run Reward: `0.6228`
- Average Decision Reward: `-0.0001`
- Rewarded Decisions: `8`
- Penalized Decisions: `8`
- Learning Delta Status: `regressed`
- Learning Delta Reason: `weak_pattern_failure_rate_regressed`
- Learning Run Count: `3`
- Learning State File: `/Users/sjoybrata/Desktop/reinforcement-agent/.qa_specialist_workspace/agent_lightning_checkpoint_learning_state.json`
- RL Checkpoint File: `/Users/sjoybrata/Desktop/reinforcement-agent/.qa_specialist_workspace/agent_lightning_checkpoint.pt`
- Weak Pattern Deltas: `14`
- Weak Improved: `0`
- Weak Regressed: `14`
- Repro Artifacts: `16`
- Policy Movement Status: `shifted`
- Policy Turnover Ratio: `0.7407`
- Policy Jaccard Similarity: `0.2593`

## Spec Intelligence
- Operations Total: `4`
- Dependency Edge Count: `1`
- Workflow Candidates: `1`

## OSS Tooling
- Python Packages: `{'schemathesis': False, 'hypothesis': False, 'openapi_core': False, 'openapi_spec_validator': False, 'pact': False, 'locust': False, 'testcontainers': False}`
- CLI Binaries: `{'restler': False, 'evomaster': False, 'zap_cli': False, 'k6': False}`
- Spec Validation: `{'validator': 'openapi_spec_validator', 'available': False, 'valid': None, 'error': ''}`

## Adaptive Selection Policy
- Algorithm: `contextual_linear_ucb`
- Candidate Scenarios: `61`
- Base Candidates: `20`
- RL Mutated Candidates: `41`
- Selected Scenarios: `16`
- Base Max Scenarios: `16`
- Effective Budget: `16`
- Expanded For Uncertainty: `False`
- Uncertain Candidates: `13`
- Uncertain Selected: `6`
- Uncertainty Threshold: `0.48868`
- Policy Feature Dim: `96`
- Scenario Patterns Tracked: `33`
- Active Repair Rules: `0`

## Scenario Learning Context
- Run Count Before: `2`
- Run Count After: `3`
- Base Generated: `20`
- Candidate Pool: `61`
- Selected Total: `16`
- Selected New vs History: `14`
- Selected Historical Weak Patterns: `2`
- Selected Source LLM Base: `5`
- Selected Source RL Mutation: `5`
- Selected Source RL History Seed: `6`
- Sample Selected Scenarios:
- `GET /orders/{orderId} - Happy Path` source=llm_base strategy= reason=forced_weak_replay expected=200 actual=404 passed=False
- `GET /products - Happy Path` source=llm_base strategy= reason=forced_weak_replay expected=200 actual=404 passed=False
- `GET /products - Boundary Test Large Page Size` source=llm_base strategy= reason=uncertainty_coverage expected=401 actual=404 passed=False
- `GET /products - Performance Test` source=llm_base strategy= reason=uncertainty_coverage expected=200 actual=404 passed=False
- `POST /orders - Dependency Failure` source=llm_base strategy= reason=uncertainty_coverage expected=503 actual=404 passed=False
- `test_post_products_rl_history_seed_input_validation_400` source=rl_history_seed strategy=history_seed reason=core_type_coverage expected=400 actual=404 passed=False
- `test_get_orders_orderId_rl_history_seed_happy_path_200` source=rl_history_seed strategy=history_seed reason=happy_path_coverage expected=200 actual=404 passed=False
- `test_get_products_rl_history_seed_happy_path_200` source=rl_history_seed strategy=history_seed reason=happy_path_coverage expected=200 actual=404 passed=False
- `test_post_orders_rl_history_seed_happy_path_201` source=rl_history_seed strategy=history_seed reason=happy_path_coverage expected=201 actual=404 passed=False
- `test_post_products_rl_history_seed_happy_path_201` source=rl_history_seed strategy=history_seed reason=happy_path_coverage expected=201 actual=404 passed=False
- Decision 1: `test_get_orders_orderId_rl_history_seed_happy_path_200` score=0.0205 uncertainty=0.4384 failure_focus=0.4
- Decision 2: `test_get_products_rl_history_seed_happy_path_200` score=0.1674 uncertainty=0.4472 failure_focus=0.4
- Decision 3: `test_post_orders_rl_history_seed_happy_path_201` score=0.142 uncertainty=0.4254 failure_focus=0.4
- Decision 4: `test_post_products_rl_history_seed_happy_path_201` score=0.1085 uncertainty=0.381 failure_focus=0.4
- Decision 5: `GET /products - Happy Path_rl_missing_auth` score=1.695 uncertainty=0.4329 failure_focus=0.4

## RL Mutation Stage
- Enabled: `True`
- Base Candidates: `20`
- Targeted Candidates: `20`
- Mutated Added: `41`
- Final Candidates: `61`
- Priority Threshold: `0.08`

## Learned Repairs
- Active Rules: `0`
- Applied Repairs: `0`
- Status Overrides: `0`
- Request Body Repairs: `0`

## Top Failures
- `test_get_orders_orderId_rl_history_seed_happy_path_200` expected=200 actual=404 endpoint=/orders/123
- `test_get_products_rl_history_seed_happy_path_200` expected=200 actual=404 endpoint=/products
- `test_post_orders_rl_history_seed_happy_path_201` expected=201 actual=404 endpoint=/orders
- `test_post_products_rl_history_seed_happy_path_201` expected=201 actual=404 endpoint=/products
- `GET /products - Happy Path_rl_missing_auth` expected=401 actual=404 endpoint=/products
- `test_post_products_rl_history_seed_input_validation_400` expected=400 actual=404 endpoint=/products
- `POST /products - Happy Path_rl_missing_required_name` expected=400 actual=404 endpoint=/products
- `POST /orders - Happy Path_rl_learned_below_min_quantity` expected=400 actual=404 endpoint=/orders
- `GET /orders/{orderId} - Happy Path` expected=200 actual=404 endpoint=/orders/abc123
- `GET /products - Happy Path` expected=200 actual=404 endpoint=/products

## GAM Memory
- Session ID: `fc6e2afb-10c9-448c-b95d-f3cbc025eba4`
- Memo Page ID: `b3da1924806a857b`
- Memo Title: `Iteration 4 Analysis: updated api endpoints and parameters: E-commerce API (fc6e2afb)`
- Memory Store Path: `/Users/sjoybrata/Desktop/reinforcement-agent/.qa_specialist_workspace/gam_memory_pages.json`
- Research Plan Items: `16`
- Research Excerpts: `8`
- Research Reflection: `Completed 2 research iterations. Found 6 relevant excerpts.`
- Quality Score: `1.0`
- Convention Excerpts: `0`
- Non-Convention Excerpts: `8`
- Actionable Excerpts: `5`
- Machine-Like Excerpts: `0`
- Warnings: `none`

## Agent Lightning RL
- Registered Agents: `1`
- Traces Collected: `18`
- Replay Buffer Size: `53`
- Training Steps: `3`
- Training Enabled: `True`

## References
- Agent Lightning: https://arxiv.org/pdf/2508.03680
- GAM: https://arxiv.org/pdf/2511.18423
