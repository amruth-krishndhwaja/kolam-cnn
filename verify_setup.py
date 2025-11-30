"""
================================================================================
SETUP VERIFICATION SCRIPT
================================================================================
Purpose: Verify that all dependencies and setup are correct before training
================================================================================
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("=" * 80)
    print("CHECKING PYTHON VERSION")
    print("=" * 80)
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ ERROR: Python 3.7 or higher is required!")
        return False
    else:
        print("✓ Python version is compatible")
        return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "=" * 80)
    print("CHECKING DEPENDENCIES")
    print("=" * 80)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'Torchvision',
        'PIL': 'Pillow',
        'numpy': 'NumPy'
    }
    
    all_installed = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} is installed")
        except ImportError:
            print(f"❌ {name} is NOT installed")
            print(f"   Install with: pip install {package}")
            all_installed = False
    
    return all_installed

def check_gpu():
    """Check GPU availability"""
    print("\n" + "=" * 80)
    print("CHECKING GPU AVAILABILITY")
    print("=" * 80)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU is available!")
            print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            return True
        else:
            print("⚠ GPU is not available (will use CPU)")
            print("  Training will be slower but will work")
            return False
    except ImportError:
        print("⚠ Cannot check GPU (PyTorch not installed)")
        return False

def check_project_structure():
    """Check if project structure is correct"""
    print("\n" + "=" * 80)
    print("CHECKING PROJECT STRUCTURE")
    print("=" * 80)
    
    base_dir = Path(__file__).parent
    required_files = [
        'config.py',
        'train.py',
        'inference.py',
        'extract_features.py',
        'models/kolam_cnn.py',
        'data/dataset.py'
    ]
    
    all_present = True
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} is missing!")
            all_present = False
    
    return all_present

def check_dataset_structure():
    """Check if dataset structure exists"""
    print("\n" + "=" * 80)
    print("CHECKING DATASET STRUCTURE")
    print("=" * 80)
    
    base_dir = Path(__file__).parent
    train_dir = base_dir / "data" / "train"
    
    if not train_dir.exists():
        print("❌ data/train/ directory does not exist!")
        print("   Create it and add your image folders:")
        print("   - data/train/elephant_kolam/")
        print("   - data/train/not_recognized/")
        return False
    
    # Check for class folders
    elephant_dir = train_dir / "elephant_kolam"
    not_recognized_dir = train_dir / "not_recognized"
    
    if elephant_dir.exists():
        image_count = len(list(elephant_dir.glob("*.jpg")) + 
                         list(elephant_dir.glob("*.jpeg")) +
                         list(elephant_dir.glob("*.png")))
        print(f"✓ elephant_kolam/ folder exists ({image_count} images)")
    else:
        print("❌ elephant_kolam/ folder does not exist!")
        print("   Create: data/train/elephant_kolam/ and add images")
    
    if not_recognized_dir.exists():
        image_count = len(list(not_recognized_dir.glob("*.jpg")) + 
                         list(not_recognized_dir.glob("*.jpeg")) +
                         list(not_recognized_dir.glob("*.png")))
        print(f"✓ not_recognized/ folder exists ({image_count} images)")
    else:
        print("❌ not_recognized/ folder does not exist!")
        print("   Create: data/train/not_recognized/ and add images")
    
    if elephant_dir.exists() and not_recognized_dir.exists():
        return True
    else:
        print("\n⚠ Dataset structure incomplete. Add images to train the model.")
        return False

def main():
    """Run all verification checks"""
    print("\n" + "=" * 80)
    print("KOLAM CLASSIFICATION CNN - SETUP VERIFICATION")
    print("=" * 80)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("GPU", check_gpu),
        ("Project Structure", check_project_structure),
        ("Dataset Structure", check_dataset_structure)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error checking {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL CHECKS PASSED! You're ready to train.")
        print("\nNext step: python train.py")
    else:
        print("⚠ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before training.")
        print("See SETUP_GUIDE.md for detailed instructions.")
    print("=" * 80)

if __name__ == "__main__":
    main()

