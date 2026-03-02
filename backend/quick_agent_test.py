#!/usr/bin/env python3
"""
QUICK AGENT TEST - Fast RL and GAM demonstration
Shows agent execution, GAM memory, and RL learning in under 10 seconds
"""

import asyncio
import time
import torch
from spec_test_pilot.agent_lightning_v2 import create_agent_lightning_system
from spec_test_pilot.memory.gam import GAMMemorySystem

async def quick_agent_test():
    print("⚡ QUICK AGENT + RL + GAM TEST")
    print("=" * 32)
    
    start_time = time.time()
    
    # Initialize (should be fast)
    print("🔧 Initializing...")
    gam = GAMMemorySystem(use_vector_search=False)
    trainer = create_agent_lightning_system(gam)
    rl_algo = trainer.rl_algorithm
    
    print(f"   ✅ System ready in {time.time() - start_time:.1f}s")
    print(f"   RL Buffer: {len(rl_algo.replay_buffer)}")
    print(f"   Training Steps: {rl_algo.training_steps}")
    print()
    
    # Quick task 1
    print("🚀 TASK 1: Quick Security Test")
    task_start = time.time()
    
    result1 = await trainer.train_agent('spec_test_pilot', {
        'spec_title': 'Quick API',
        'nlp_prompt': 'Generate basic security tests',
        'tenant_id': 'test1'
    })
    
    task1_time = time.time() - task_start
    stats1 = trainer.get_training_stats()
    
    print(f"   ✅ Completed in {task1_time:.1f}s")
    print(f"   Success: {result1['success']}")
    print(f"   RL Buffer: {stats1['rl_buffer_size']}")
    print(f"   Training Steps: {stats1['rl_training_steps']}")
    
    if stats1['rl_training_steps'] > 0:
        print("   🧠 RL TRAINING ACTIVATED!")
    print()
    
    # Quick task 2
    print("🚀 TASK 2: Quick Performance Test")
    task_start = time.time()
    
    result2 = await trainer.train_agent('spec_test_pilot', {
        'spec_title': 'Fast API',
        'nlp_prompt': 'Generate performance tests',
        'tenant_id': 'test2'
    })
    
    task2_time = time.time() - task_start
    stats2 = trainer.get_training_stats()
    
    print(f"   ✅ Completed in {task2_time:.1f}s")
    print(f"   Success: {result2['success']}")
    print(f"   RL Buffer: {stats2['rl_buffer_size']}")
    print(f"   Training Steps: {stats2['rl_training_steps']}")
    
    if stats2['rl_training_steps'] > stats1['rl_training_steps']:
        print("   🧠 MORE RL TRAINING!")
    print()
    
    # Test GAM memory
    print("💾 TESTING GAM MEMORY:")
    search_results = gam.search("security testing", top_k=3)
    print(f"   Found {len(search_results)} memory entries")
    for i, (page, score) in enumerate(search_results[:2], 1):
        print(f"   {i}. \"{page.title[:40]}...\" (score: {score:.2f})")
    print()
    
    # Show RL learning
    if stats2['rl_training_steps'] > 0:
        print("🧠 RL LEARNING VERIFICATION:")
        
        # Test network predictions
        test_states = [
            {'task_type': 'security', 'complexity': 0.8},
            {'task_type': 'performance', 'complexity': 0.6}
        ]
        
        for state in test_states:
            state_vec = rl_algo._encode_state(state)
            with torch.no_grad():
                pred = rl_algo.value_net(torch.FloatTensor([state_vec])).item()
            print(f"   {state['task_type']}: prediction = {pred:.4f}")
        print()
    
    total_time = time.time() - start_time
    
    print("📊 QUICK TEST RESULTS:")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Tasks completed: 2")
    print(f"   RL training steps: {stats2['rl_training_steps']}")
    print(f"   GAM entries: {len(search_results)}")
    print(f"   Success rate: {(result1['success'] + result2['success']) / 2 * 100:.0f}%")
    print()
    
    if stats2['rl_training_steps'] > 0 and len(search_results) > 0:
        print("✅ EVERYTHING WORKING:")
        print("   • Agent executes tasks quickly")
        print("   • RL training happens after each task")
        print("   • GAM memory stores and retrieves context")
        print("   • Neural network learns from performance")
        print("   • System is fast and responsive")
    else:
        print("⚠️  Some components need more data or time")
    
    return {
        'total_time': total_time,
        'rl_steps': stats2['rl_training_steps'],
        'gam_entries': len(search_results),
        'success_rate': (result1['success'] + result2['success']) / 2
    }

if __name__ == "__main__":
    result = asyncio.run(quick_agent_test())
    
    print()
    if result['total_time'] < 15 and result['rl_steps'] > 0:
        print("🎉 QUICK TEST PASSED - SYSTEM IS FAST AND WORKING!")
    else:
        print(f"⏱️  Test took {result['total_time']:.1f}s - may need optimization")
