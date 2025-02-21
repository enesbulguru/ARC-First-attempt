import os
import json
from arc_solver import HyperDimensionalSolver

def run_training_tests(training_dir: str = r"C:\Users\Acer\Desktop\Project Simurg\Graveyard\Arc\arcprize\data\training"):
    """Run solver on all training files"""
    solver = HyperDimensionalSolver()
    results = []
    
    for filename in os.listdir(training_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(training_dir, filename)
            
            with open(file_path, 'r') as f:
                task = json.load(f)
            
            correct = 0
            total = len(task['test'])
            
            for test_case in task['test']:
                solution = solver.solve_puzzle(test_case['input'])
                if 'output' in test_case:
                    # Convert numpy arrays to lists for comparison
                    if all(all(a == b for a, b in zip(row1, row2)) 
                          for row1, row2 in zip(solution, test_case['output'])):
                        correct += 1
            
            results.append({
                'task': filename,
                'score': f"{correct}/{total}",
                'passed': correct == total
            })
            
            print(f"Task {filename}: {correct}/{total} correct")
    
    passed = sum(1 for r in results if r['passed'])
    print(f"\nOverall: {passed}/{len(results)} tasks passed")
    
    return results

def test_single_task(json_str: str):
    """Test solver on a single task"""
    solver = HyperDimensionalSolver()
    task = json.loads(json_str)
    
    print("Testing training examples...")
    for i, train_case in enumerate(task['train']):
        prediction = solver.solve_puzzle(train_case['input'])
        # Compare lists element by element
        success = all(all(a == b for a, b in zip(row1, row2)) 
                     for row1, row2 in zip(prediction, train_case['output']))
        print(f"Training example {i+1}: {'✓' if success else '✗'}")
        
        if not success:
            print("\nExpected:")
            print_grid(train_case['output'])
            print("\nGot:")
            print_grid(prediction)
    
    print("\nTesting test cases...")
    for i, test_case in enumerate(task['test']):
        prediction = solver.solve_puzzle(test_case['input'])
        if 'output' in test_case:
            # Compare lists element by element
            success = all(all(a == b for a, b in zip(row1, row2)) 
                         for row1, row2 in zip(prediction, test_case['output']))
            print(f"Test case {i+1}: {'✓' if success else '✗'}")
            
            if not success:
                print("\nExpected:")
                print_grid(test_case['output'])
                print("\nGot:")
                print_grid(prediction)
        else:
            print("\nPrediction for test case:")
            print_grid(prediction)

def print_grid(grid):
    """Helper function to pretty-print grids"""
    for row in grid:
        print(' '.join(str(x) for x in row))

if __name__ == "__main__":
    # Option 1: Test a single task
    sample_task = '''
    {"train": [{"input": [[0, 0, 0], [0, 3, 0], [0, 0, 0]], 
                "output": [[0, 0, 0], [0, 4, 0], [0, 0, 0]]}],
     "test": [{"input": [[0, 0, 0], [0, 3, 0], [0, 0, 0]]}]}
    '''
    test_single_task(sample_task)
    
    # Option 2: Run all training tests
    print("\nRunning all training tests:")
    run_training_tests()