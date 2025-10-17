"""
Test that all required packages are installed and working correctly.
Run this after setting up your environment to verify everything is working.
"""

def test_core_packages():
    """Test core simulation packages"""
    print("Testing core packages...")
    
    try:
        import simpy
        print(f"✅ SimPy {simpy.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
        
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
        
        import yaml
        print(f"✅ PyYAML {yaml.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False

def test_simpy_basic():
    """Test basic SimPy functionality"""
    print("\nTesting SimPy basic functionality...")
    
    try:
        import simpy
        
        def test_process(env):
            print(f"  Process started at time {env.now}")
            yield env.timeout(5)
            print(f"  Process finished at time {env.now}")
        
        env = simpy.Environment()
        env.process(test_process(env))
        env.run(until=10)
        
        print("✅ SimPy basic test passed")
        return True
    except Exception as e:
        print(f"❌ SimPy test failed: {e}")
        return False

def test_data_analysis():
    """Test data analysis packages"""
    print("\nTesting data analysis packages...")
    
    try:
        import numpy as np
        import pandas as pd
        
        # Create test data
        data = np.random.randn(100)
        df = pd.DataFrame({'values': data})
        
        # Basic operations
        mean_val = df['values'].mean()
        std_val = df['values'].std()
        
        print(f"✅ Data analysis test passed (mean: {mean_val:.2f}, std: {std_val:.2f})")
        return True
    except Exception as e:
        print(f"❌ Data analysis test failed: {e}")
        return False

def test_plotting():
    """Test plotting functionality (non-interactive)"""
    print("\nTesting plotting functionality...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create simple plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        plt.figure(figsize=(8, 6))
        plt.plot(x, y)
        plt.title('Test Plot')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # Save to temporary location (won't show up in results/ due to .gitignore)
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()
            os.unlink(tmp.name)  # Clean up
        
        print("✅ Plotting test passed")
        return True
    except Exception as e:
        print(f"❌ Plotting test failed: {e}")
        return False

def test_optional_packages():
    """Test optional packages"""
    print("\nTesting optional packages...")
    
    optional_tests = []
    
    try:
        import networkx as nx
        print(f"✅ NetworkX {nx.__version__}")
        optional_tests.append(True)
    except ImportError:
        print("⚠️  NetworkX not installed (optional)")
        optional_tests.append(False)
    
    try:
        import tqdm
        print(f"✅ tqdm {tqdm.__version__}")
        optional_tests.append(True)
    except ImportError:
        print("⚠️  tqdm not installed (optional)")
        optional_tests.append(False)
    
    try:
        import scipy
        print(f"✅ SciPy {scipy.__version__}")
        optional_tests.append(True)
    except ImportError:
        print("⚠️  SciPy not installed (optional)")
        optional_tests.append(False)
    
    return any(optional_tests)

def main():
    """Run all installation tests"""
    print("🧪 Testing Underwater DTN Simulation Installation")
    print("=" * 50)
    
    tests = [
        test_core_packages(),
        test_simpy_basic(),
        test_data_analysis(),
        test_plotting(),
    ]
    
    # Optional packages (don't fail if missing)
    test_optional_packages()
    
    print("\n" + "=" * 50)
    
    if all(tests):
        print("🎉 All tests passed! Your environment is ready for simulation development.")
        print("\nNext steps:")
        print("1. Start coding in experiments/quick_tests/playground.py")
        print("2. Build your simulation components in src/")
        print("3. Run simulations and analyze results")
    else:
        print("❌ Some tests failed. Please install missing packages:")
        print("   pip install -r requirements.txt")
        print("   pip install -e .")
    
    return all(tests)

if __name__ == "__main__":
    main()