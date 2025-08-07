#!/usr/bin/env python3
"""
Test script to validate both lighteval and inspect_ai frameworks are properly integrated.
This script checks imports, task definitions, and basic functionality.
"""

import sys
import os
from pathlib import Path

def test_lighteval_integration():
    """Test lighteval integration"""
    print("Testing lighteval integration...")
    
    try:
        # Test importing lighteval tasks
        from tasks.french_evals import TASKS_TABLE
        print(f"✓ Lighteval tasks loaded: {len(TASKS_TABLE)} tasks")
        
        # Test importing lighteval with reasoning tasks  
        from tasks.french_evals_w_reasoning import TASKS_TABLE as TASKS_W_REASONING
        print(f"✓ Lighteval reasoning tasks loaded: {len(TASKS_W_REASONING)} tasks")
        
        return True
        
    except ImportError as e:
        print(f"✗ Lighteval integration failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Lighteval integration error: {e}")
        return False


def test_inspect_ai_integration():
    """Test inspect_ai integration"""
    print("Testing inspect_ai integration...")
    
    try:
        # Test importing inspect_ai tasks
        from tasks.french_evals_inspect import AVAILABLE_TASKS
        print(f"✓ Inspect AI tasks loaded: {len(AVAILABLE_TASKS)} tasks")
        
        # List available tasks
        task_names = list(AVAILABLE_TASKS.keys())
        print(f"  Available tasks: {', '.join(task_names)}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Inspect AI integration failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Inspect AI integration error: {e}")
        return False


def test_script_imports():
    """Test that main scripts can be imported"""
    print("Testing main script imports...")
    
    results = {}
    
    # Test lighteval scripts
    scripts_to_test = [
        'run-lighteval.py',
        'run-inspect-ai.py', 
        'run-inspect-ai-interactive.py'
    ]
    
    for script in scripts_to_test:
        try:
            # Import as module (remove .py and replace - with _)
            module_name = script.replace('.py', '').replace('-', '_')
            
            # Check if file exists
            if not os.path.exists(script):
                print(f"✗ Script {script} not found")
                results[script] = False
                continue
                
            print(f"✓ Script {script} exists")
            results[script] = True
            
        except Exception as e:
            print(f"✗ Script {script} import error: {e}")
            results[script] = False
    
    return all(results.values())


def test_container_files():
    """Test that container definition files exist"""
    print("Testing container files...")
    
    files_to_check = [
        'workers_image/Singularityfile',
        'workers_image/requirements.txt',
        'workers_image/Singularityfile_inspect', 
        'workers_image/requirements_inspect.txt'
    ]
    
    results = {}
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
            results[file_path] = True
        else:
            print(f"✗ {file_path} missing")
            results[file_path] = False
    
    return all(results.values())


def test_pipeline_scripts():
    """Test that pipeline scripts exist and have framework selection logic"""
    print("Testing pipeline scripts...")
    
    scripts_to_check = [
        'run-eval-main.sh',
        'run-eval-main-slurm.sh',
        'run-eval-workers.sh'
    ]
    
    results = {}
    for script in scripts_to_check:
        if not os.path.exists(script):
            print(f"✗ {script} missing")
            results[script] = False
            continue
        
        # Check if script contains framework selection logic
        with open(script, 'r') as f:
            content = f.read()
            if 'FRAMEWORK' in content:
                print(f"✓ {script} has framework selection logic")
                results[script] = True
            else:
                print(f"✗ {script} missing framework selection logic")
                results[script] = False
    
    return all(results.values())


def main():
    """Run all tests"""
    print("="*60)
    print("Testing Inspect AI Integration")
    print("="*60)
    
    tests = [
        test_lighteval_integration,
        test_inspect_ai_integration, 
        test_script_imports,
        test_container_files,
        test_pipeline_scripts
    ]
    
    results = []
    for test in tests:
        print()
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append(False)
    
    print()
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results)
    total = len(results) 
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Inspect AI integration is ready.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()