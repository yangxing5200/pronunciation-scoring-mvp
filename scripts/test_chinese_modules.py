#!/usr/bin/env python
"""
Test script for Chinese pronunciation scoring modules.

Tests each module independently without requiring full dependency installation.
"""

import sys
import importlib.util
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_module(module_name, file_path):
    """Load a module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinyin_mapper():
    """Test Task 1: Pinyin Mapper"""
    print("\n=== Testing Task 1: Pinyin Mapper ===")
    try:
        pm_module = load_module(
            "pinyin_mapper",
            project_root / "core/chinese/pinyin_mapper.py"
        )
        
        mapper = pm_module.PinyinMapper()
        print(f"✓ PinyinMapper initialized: {mapper.is_available()}")
        
        if mapper.is_available():
            # Test basic conversion
            result = mapper.text_to_pinyin("你好，这是一个测试。")
            print(f"✓ Text to pinyin: {len(result)} characters mapped")
            print(f"  Sample: {result[:2]}")
            
            # Test tone extraction
            tone = mapper.get_tone("ni3")
            print(f"✓ Tone extraction: 'ni3' -> tone {tone}")
            
            # Test initial/final splitting
            parts = mapper.get_initial_final("hao3")
            print(f"✓ Initial/final split: 'hao3' -> {parts}")
            
        return True
    except Exception as e:
        print(f"✗ PinyinMapper failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_module_structure():
    """Test module structure and imports."""
    print("\n=== Testing Module Structure ===")
    
    modules = [
        "pinyin_mapper.py",
        "audio_aligner.py",
        "audio_slicer.py",
        "acoustic_scorer.py",
        "tone_scorer.py",
        "duration_scorer.py",
        "pause_scorer.py",
        "error_classifier.py",
        "final_scorer.py",
        "pipeline.py",
    ]
    
    chinese_dir = project_root / "core/chinese"
    
    for module_file in modules:
        module_path = chinese_dir / module_file
        if module_path.exists():
            print(f"✓ {module_file} exists ({module_path.stat().st_size} bytes)")
        else:
            print(f"✗ {module_file} missing")
    
    # Check __init__.py
    init_file = chinese_dir / "__init__.py"
    if init_file.exists():
        print(f"✓ __init__.py exists")
    else:
        print(f"✗ __init__.py missing")
    
    return True


def test_duration_scorer():
    """Test Task 6: Duration Scorer (no external dependencies)"""
    print("\n=== Testing Task 6: Duration Scorer ===")
    try:
        ds_module = load_module(
            "duration_scorer",
            project_root / "core/chinese/duration_scorer.py"
        )
        
        scorer = ds_module.DurationScorer()
        print(f"✓ DurationScorer initialized")
        
        # Test scoring without reference
        test_data = [
            {"char": "你", "pinyin": "ni3", "duration": 0.25, "start": 0.0, "end": 0.25},
            {"char": "好", "pinyin": "hao3", "duration": 0.30, "start": 0.25, "end": 0.55},
        ]
        
        results = scorer.score_durations(test_data)
        print(f"✓ Duration scoring: {len(results)} characters scored")
        for r in results:
            print(f"  {r['char']}: duration={r['duration']:.2f}s, score={r['duration_score']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ DurationScorer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pause_scorer():
    """Test Task 7: Pause Scorer (no external dependencies)"""
    print("\n=== Testing Task 7: Pause Scorer ===")
    try:
        ps_module = load_module(
            "pause_scorer",
            project_root / "core/chinese/pause_scorer.py"
        )
        
        scorer = ps_module.PauseScorer()
        print(f"✓ PauseScorer initialized")
        
        # Test scoring
        test_data = [
            {"char": "你", "pinyin": "ni3", "start": 0.0, "end": 0.25},
            {"char": "好", "pinyin": "hao3", "start": 0.30, "end": 0.55},
            {"char": "世", "pinyin": "shi4", "start": 0.60, "end": 0.85},
        ]
        
        results = scorer.score_pauses(test_data)
        print(f"✓ Pause scoring: {len(results)} characters scored")
        for r in results:
            print(f"  {r['char']}: pause_after={r['pause_after']:.3f}s, score={r['pause_score']:.2f}")
        
        # Test overall fluency
        fluency = scorer.calculate_overall_fluency(results)
        print(f"✓ Overall fluency: {fluency['overall_fluency_score']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ PauseScorer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_final_scorer():
    """Test Task 9: Final Scorer (no external dependencies)"""
    print("\n=== Testing Task 9: Final Scorer ===")
    try:
        fs_module = load_module(
            "final_scorer",
            project_root / "core/chinese/final_scorer.py"
        )
        
        scorer = fs_module.FinalScorer()
        print(f"✓ FinalScorer initialized")
        
        # Test scoring
        test_data = [
            {
                "char": "你", "pinyin": "ni3",
                "acoustic_score": 0.92,
                "tone_score": 0.95,
                "duration_score": 0.98,
                "pause_score": 1.0
            },
            {
                "char": "好", "pinyin": "hao3",
                "acoustic_score": 0.88,
                "tone_score": 0.90,
                "duration_score": 0.85,
                "pause_score": 0.95
            },
        ]
        
        results = scorer.calculate_final_scores(test_data)
        print(f"✓ Final scoring: {len(results)} characters scored")
        for r in results:
            print(f"  {r['char']}: final_score={r['final_score']}/100")
        
        # Test overall metrics
        overall = scorer.calculate_overall_score(results)
        print(f"✓ Overall score: {overall['overall_score']}/100")
        
        # Test feedback
        feedback = scorer.generate_feedback(results, overall)
        print(f"✓ Feedback generated: {len(feedback)} messages")
        for f in feedback:
            print(f"  - {f}")
        
        return True
    except Exception as e:
        print(f"✗ FinalScorer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Chinese Pronunciation Scoring Module Tests")
    print("=" * 60)
    
    tests = [
        ("Module Structure", test_module_structure),
        ("Pinyin Mapper (Task 1)", test_pinyin_mapper),
        ("Duration Scorer (Task 6)", test_duration_scorer),
        ("Pause Scorer (Task 7)", test_pause_scorer),
        ("Final Scorer (Task 9)", test_final_scorer),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
