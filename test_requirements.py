#!/usr/bin/env python3
"""
Test script to verify all required packages are working correctly.
"""

print("Testing required packages...")

try:
    import numpy as np
    print(f"✅ numpy {np.__version__} - OK")
    
    import pandas as pd
    print(f"✅ pandas {pd.__version__} - OK")
    
    import yfinance as yf
    print(f"✅ yfinance - OK")
    
    import matplotlib.pyplot as plt
    import matplotlib
    print(f"✅ matplotlib {matplotlib.__version__} - OK")
    
    print("\n🎉 All packages imported successfully!")
    
    # Quick functionality test
    print("\nRunning quick functionality tests...")
    
    # Test numpy
    arr = np.array([1, 2, 3, 4, 5])
    print(f"NumPy array mean: {arr.mean()}")
    
    # Test pandas
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    print(f"Pandas DataFrame shape: {df.shape}")
    
    # Test yfinance (without making actual API call to avoid rate limits)
    ticker = yf.Ticker("AAPL")
    print("YFinance ticker object created successfully")
    
    # Test matplotlib
    plt.figure(figsize=(6, 4))
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.title("Test Plot")
    print("Matplotlib plot created successfully")
    
    print("\n✨ All functionality tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"⚠️  Error during testing: {e}")