#!/usr/bin/env python3
"""
Simple test/demo script to verify core functionality.

This script tests the core modules without requiring the full dependencies.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_text_comparator():
    """Test TextComparator module."""
    print("\n" + "="*60)
    print("Testing TextComparator")
    print("="*60)
    
    try:
        from core.text_comparator import TextComparator
        
        tc = TextComparator()
        
        # Test 1: Exact match
        result = tc.compare_texts("hello world", "hello world")
        assert result["match"] == True
        print("✓ Test 1: Exact match works")
        
        # Test 2: Different texts
        result = tc.compare_texts("hello world", "goodbye world")
        assert result["match"] == False
        assert result["similarity"] < 1.0
        print(f"✓ Test 2: Similarity calculation works (similarity: {result['similarity']:.2f})")
        
        # Test 3: Word similarity
        sim = tc.calculate_word_similarity("hello", "helo")
        assert 0.5 < sim < 1.0
        print(f"✓ Test 3: Word similarity works (similarity: {sim:.2f})")
        
        # Test 4: Word analysis
        result = tc.compare_texts("the cat sat", "the dog sat")
        assert "cat" in result["missing_words"]
        assert "dog" in result["extra_words"]
        print("✓ Test 4: Word difference detection works")
        
        print("\n✅ All TextComparator tests passed!")
        return True
        
    except ImportError as e:
        print(f"⚠️  Skipped (dependencies not installed): {e}")
        return None
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*60)
    print("Testing Module Imports")
    print("="*60)
    
    modules = [
        ("core.transcriber", "WhisperTranscriber"),
        ("core.aligner", "PhonemeAligner"),
        ("core.scorer", "PronunciationScorer"),
        ("core.text_comparator", "TextComparator"),
        ("core.voice_cloner", "VoiceCloner"),
    ]
    
    results = []
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
            results.append(True)
        except ImportError as e:
            print(f"⚠️  {module_name}.{class_name} (dependencies missing)")
            results.append(None)
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: {e}")
            results.append(False)
    
    if all(r is not False for r in results):
        print("\n✅ All imports successful (or skipped)")
        return True
    else:
        print("\n❌ Some imports failed")
        return False


def test_file_structure():
    """Test that required files and directories exist."""
    print("\n" + "="*60)
    print("Testing File Structure")
    print("="*60)
    
    required_items = [
        ("File", "README.md"),
        ("File", "requirements.txt"),
        ("File", "app.py"),
        ("Dir", "core"),
        ("Dir", "models"),
        ("Dir", "assets"),
        ("Dir", "scripts"),
        ("File", "core/__init__.py"),
        ("File", "core/transcriber.py"),
        ("File", "core/aligner.py"),
        ("File", "core/scorer.py"),
        ("File", "core/text_comparator.py"),
        ("File", "core/voice_cloner.py"),
        ("File", "scripts/download_models.py"),
    ]
    
    base_dir = Path(__file__).parent.parent
    all_exist = True
    
    for item_type, item_path in required_items:
        full_path = base_dir / item_path
        
        if item_type == "File":
            exists = full_path.is_file()
        else:
            exists = full_path.is_dir()
        
        if exists:
            print(f"✓ {item_path}")
        else:
            print(f"❌ {item_path} (missing)")
            all_exist = False
    
    if all_exist:
        print("\n✅ All required files and directories exist!")
        return True
    else:
        print("\n❌ Some required files/directories are missing")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Pronunciation Scoring MVP - System Test")
    print("="*60)
    
    results = {
        "File Structure": test_file_structure(),
        "Module Imports": test_imports(),
        "TextComparator": test_text_comparator(),
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is None:
            status = "⚠️  SKIP (dependencies)"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "❓ UNKNOWN"
        
        print(f"{test_name}: {status}")
    
    # Overall result
    failures = sum(1 for r in results.values() if r is False)
    
    if failures == 0:
        print("\n✅ All tests passed or skipped!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download models: python scripts/download_models.py")
        print("3. Run application: streamlit run app.py")
        return 0
    else:
        print(f"\n❌ {failures} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
