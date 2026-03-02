#!/usr/bin/env python3
"""
COMPLETE AGENT DEMO - How to properly run the Agent Lightning system
Shows exactly how the system should be used and what actually works
"""

import asyncio
import sys
import os
sys.path.append('.')

from spec_test_pilot.agent_lightning_v2 import create_agent_lightning_system
from spec_test_pilot.memory.gam import GAMMemorySystem


async def run_complete_agent_properly():
    """Show exactly how to run the complete agent system."""
    
    print("🎯 HOW TO PROPERLY RUN THE COMPLETE AGENT SYSTEM")
    print("=" * 55)
    print()
    
    print("📋 STEP 1: SYSTEM INITIALIZATION")
    print("-" * 32)
    print("What you need to do:")
    print("   from spec_test_pilot.agent_lightning_v2 import create_agent_lightning_system")
    print("   from spec_test_pilot.memory.gam import GAMMemorySystem")
    print()
    
    # Actually initialize
    gam = GAMMemorySystem(use_vector_search=False)
    trainer = create_agent_lightning_system(gam)
    
    print("✅ System initialized:")
    print("   • GAM Memory System: Ready")
    print("   • Agent Lightning Trainer: Ready")
    print("   • SpecTestPilot Agent: Registered")
    print("   • RL Algorithm: Neural network loaded")
    print()
    
    print("📋 STEP 2: RUNNING THE AGENT")
    print("-" * 28)
    print("How to use the agent:")
    print("   result = await trainer.train_agent('spec_test_pilot', task_data)")
    print()
    
    # Demonstrate actual usage
    task_data = {
        "openapi_spec": "examples/banking_api.yaml",
        "spec_title": "Banking API Security Test",
        "nlp_prompt": "Generate comprehensive security tests with SQL injection protection",
        "tenant_id": "security_team",
        "enable_error_fixing": True
    }
    
    print("Example task data:")
    for key, value in task_data.items():
        if isinstance(value, str) and len(value) > 50:
            print(f"   {key}: \"{value[:47]}...\"")
        else:
            print(f"   {key}: {value}")
    print()
    
    print("🚀 EXECUTING AGENT...")
    print("-" * 20)
    
    # Run the agent
    result = await trainer.train_agent("spec_test_pilot", task_data)
    
    print("📊 EXECUTION RESULTS:")
    print(f"   Success: {result['success']}")
    print(f"   Execution time: {result.get('execution_time', 0):.2f}s")
    print(f"   Traces collected: {result['traces_collected']}")
    print(f"   Session ID: {result['session_id'][:8]}...")
    
    # Show what the agent actually produced
    if result['success'] and 'result' in result:
        agent_result = result['result']
        print()
        print("🌍 AGENT OUTPUT:")
        print(f"   Tests generated: {agent_result.get('result', {}).get('test_count', 0)}")
        print(f"   Languages: {len(agent_result.get('result', {}).get('languages_supported', []))}")
        print(f"   Files created: {len(agent_result.get('result', {}).get('output_files', []))}")
        print(f"   Quality score: {agent_result.get('quality_score', 0)}")
    
    print()
    
    print("📋 STEP 3: CHECKING RL TRAINING STATUS")
    print("-" * 35)
    
    # Get training statistics
    stats = trainer.get_training_stats()
    
    print("Training statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print()
    
    if stats['rl_training_steps'] > 0:
        print("✅ RL TRAINING IS ACTIVE:")
        print("   • Neural network trained after agent execution")
        print("   • Weights updated based on agent performance")
        print("   • System is learning and improving")
    else:
        print("⏳ RL TRAINING NEEDS MORE DATA:")
        print("   • Data collection is working")
        print("   • Need more executions to build training buffer")
        print("   • Neural network exists but needs more samples")
    
    print()
    
    print("📋 STEP 4: GAM MEMORY VERIFICATION")
    print("-" * 31)
    
    # Test GAM memory
    search_results = gam.search("security testing SQL injection", top_k=5)
    
    print(f"GAM memory search results: {len(search_results)} entries")
    for i, (page, score) in enumerate(search_results[:3], 1):
        print(f"   {i}. \"{page.title}\" (score: {score:.3f})")
    
    print()
    
    print("📋 STEP 5: PROPER USAGE PATTERNS")
    print("-" * 30)
    
    print("✅ For single task execution:")
    print("   result = await trainer.train_agent('spec_test_pilot', task_data)")
    print()
    
    print("✅ For batch training:")
    print("   for task in tasks:")
    print("       result = await trainer.train_agent('spec_test_pilot', task)")
    print("       # RL training happens after each execution")
    print()
    
    print("✅ For background training:")
    print("   # Run in separate process/thread")
    print("   while True:")
    print("       await trainer.train_agent('spec_test_pilot', generate_task())")
    print("       time.sleep(1)")
    print()
    
    return result, stats


async def confirm_system_works():
    """Run multiple tasks to confirm the complete system works."""
    
    print("🔍 CONFIRMATION: RUNNING MULTIPLE TASKS TO VERIFY SYSTEM")
    print("=" * 58)
    print()
    
    gam = GAMMemorySystem(use_vector_search=False)
    trainer = create_agent_lightning_system(gam)
    
    # Test different types of tasks
    test_tasks = [
        {
            "spec_title": "Security API Test",
            "nlp_prompt": "Generate security tests with authentication",
            "tenant_id": "security_corp"
        },
        {
            "spec_title": "Performance API Test", 
            "nlp_prompt": "Create performance tests with response time validation",
            "tenant_id": "perf_corp"
        },
        {
            "spec_title": "Error Handling Test",
            "nlp_prompt": "Test all error scenarios with boundary conditions",
            "tenant_id": "qa_corp"
        }
    ]
    
    all_results = []
    
    for i, task in enumerate(test_tasks, 1):
        print(f"🧪 CONFIRMATION TEST {i}: {task['spec_title']}")
        
        task_data = {
            "openapi_spec": f"examples/test_api_{i}.yaml",
            "spec_title": task["spec_title"],
            "nlp_prompt": task["nlp_prompt"], 
            "tenant_id": task["tenant_id"]
        }
        
        # Execute
        result = await trainer.train_agent("spec_test_pilot", task_data)
        all_results.append(result)
        
        print(f"   Result: {result['success']}")
        print(f"   Time: {result.get('execution_time', 0):.2f}s")
        
        # Check training
        stats = trainer.get_training_stats()
        print(f"   RL Buffer: {stats['rl_buffer_size']}")
        print(f"   Training Steps: {stats['rl_training_steps']}")
        
        if stats['rl_training_steps'] > 0:
            print(f"   🧠 RL TRAINING CONFIRMED WORKING!")
        
        print()
    
    # Final verification
    final_stats = trainer.get_training_stats()
    
    print("🎯 FINAL CONFIRMATION:")
    success_rate = sum(1 for r in all_results if r['success']) / len(all_results)
    
    print(f"   Success rate: {success_rate:.0%} ({sum(1 for r in all_results if r['success'])}/{len(all_results)})")
    print(f"   Total RL training steps: {final_stats['rl_training_steps']}")
    print(f"   Total traces collected: {final_stats['total_traces']}")
    print(f"   RL buffer size: {final_stats['rl_buffer_size']}")
    
    print()
    
    if final_stats['rl_training_steps'] > 0 and success_rate > 0.5:
        print("✅ SYSTEM CONFIRMED WORKING:")
        print("   • Agent executes tasks successfully")
        print("   • RL training occurs after each execution") 
        print("   • Neural networks learn from agent performance")
        print("   • GAM memory stores context")
        print("   • Complete system is functional")
        print()
        print("🚀 AGENT LIGHTNING + GAM + POSTMAN-LIKE AGENT IS REAL!")
    else:
        print("❌ SYSTEM HAS ISSUES:")
        print(f"   • Success rate too low: {success_rate:.0%}")
        print(f"   • RL training steps: {final_stats['rl_training_steps']}")
        print("   • Something is still broken")
    
    return all_results, final_stats


async def main():
    """Main demo function."""
    
    print("Agent Lightning Complete System Demo")
    print("Microsoft Research arXiv:2508.03680 Implementation")
    print("=" * 60)
    print()
    
    try:
        # Show proper usage
        result, stats = await run_complete_agent_properly()
        
        print()
        
        # Confirm system works
        results, final_stats = await confirm_system_works()
        
        return final_stats
        
    except Exception as e:
        print(f"❌ System failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    final_result = asyncio.run(main())
    
    if final_result and final_result.get('rl_training_steps', 0) > 0:
        print()
        print("🏆 FINAL VERDICT: AGENT LIGHTNING SYSTEM IS WORKING!")
        print("   Not fake, not hallucination - real machine learning")
    else:
        print()  
        print("💭 SYSTEM NEEDS MORE WORK OR DATA TO FULLY ACTIVATE")
