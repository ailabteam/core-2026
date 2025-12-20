#!/usr/bin/env python3
"""
Diagnostic script to check if all dependencies are installed correctly
"""

import sys

print("=" * 60)
print("DCAD Dependency Checker")
print("=" * 60)
print(f"\nPython executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"\nChecking dependencies...\n")

dependencies = {
    'cv2': 'opencv-python',
    'numpy': 'numpy',
    'PIL': 'Pillow',
    'pdf2image': 'pdf2image',
    'skimage': 'scikit-image',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib'
}

missing = []
installed = []

for module_name, package_name in dependencies.items():
    try:
        if module_name == 'PIL':
            import PIL
            version = getattr(PIL, '__version__', 'unknown')
        elif module_name == 'cv2':
            import cv2
            version = cv2.__version__
        elif module_name == 'skimage':
            import skimage
            version = skimage.__version__
        else:
            mod = __import__(module_name)
            version = getattr(mod, '__version__', 'unknown')
        
        print(f"✅ {package_name:20s} - {module_name:10s} (version: {version})")
        installed.append(package_name)
    except ImportError as e:
        print(f"❌ {package_name:20s} - {module_name:10s} - MISSING")
        missing.append(package_name)

print("\n" + "=" * 60)
if missing:
    print(f"\n⚠️  Missing {len(missing)} package(s):")
    for pkg in missing:
        print(f"   - {pkg}")
    print(f"\n💡 Install missing packages:")
    print(f"   {sys.executable} -m pip install {' '.join(missing)}")
    print(f"\n   Or install all requirements:")
    print(f"   {sys.executable} -m pip install -r requirements.txt")
    sys.exit(1)
else:
    print("\n✅ All dependencies are installed!")
    print("\nYou can now run:")
    print("   python3 app.py <image_path>")
    sys.exit(0)
