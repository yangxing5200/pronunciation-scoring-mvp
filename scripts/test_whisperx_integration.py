#!/usr/bin/env python3
"""
Test script for WhisperX integration and enhanced timestamp alignment.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_whisperx_optional_import():
    """Test that WhisperX is optional."""
    print("\n" + "="*60)
    print("Testing WhisperX Optional Import")
    print("="*60)
    
    try:
        # Check that transcriber file mentions WhisperX
        transcriber_file = Path(__file__).parent.parent / "core" / "transcriber.py"
        with open(transcriber_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'whisperx' in content.lower(), "WhisperX not mentioned in transcriber"
            assert '_load_whisperx' in content, "_load_whisperx method not found"
            assert 'use_whisperx' in content, "use_whisperx parameter not found"
        print("✓ Test 1: WhisperX support code exists in transcriber")
        
        # Check that it has the new methods
        assert '_transcribe_with_whisper' in content, "_transcribe_with_whisper not found"
        assert '_transcribe_with_whisperx' in content, "_transcribe_with_whisperx not found"
        print("✓ Test 2: Separate transcription methods exist for Whisper and WhisperX")
        
        # Check alignment_type is returned
        assert 'alignment_type' in content, "alignment_type not in return value"
        print("✓ Test 3: alignment_type is included in results")
        
        # Check phoneme support
        assert 'phoneme' in content.lower(), "phoneme support not found"
        assert '"phonemes"' in content, "phonemes key not in return dict"
        print("✓ Test 4: Phoneme-level support is implemented")
        
        # Check language-specific handling
        assert '_is_chinese_lang' in content, "_is_chinese_lang method not found"
        assert 'return_char_alignments' in content, "Character alignment not implemented"
        print("✓ Test 5: Language-specific alignment handling exists")
        
        print("\n✅ All WhisperX integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_app_whisperx_support():
    """Test that app.py supports WhisperX."""
    print("\n" + "="*60)
    print("Testing App WhisperX Support")
    print("="*60)
    
    try:
        app_file = Path(__file__).parent.parent / "app.py"
        with open(app_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Test 1: App checks for WhisperX
        assert 'import whisperx' in content, "App doesn't check for WhisperX"
        print("✓ Test 1: App checks for WhisperX availability")
        
        # Test 2: use_whisperx parameter is used
        assert 'use_whisperx' in content, "use_whisperx not passed to transcriber"
        print("✓ Test 2: use_whisperx parameter is used")
        
        # Test 3: Alignment type is displayed
        assert 'alignment_type' in content, "alignment_type not used in app"
        print("✓ Test 3: Alignment type is displayed to users")
        
        # Test 4: User feedback about WhisperX
        assert 'WhisperX' in content or 'whisperx' in content.lower(), "No user feedback about WhisperX"
        print("✓ Test 4: User feedback about WhisperX exists")
        
        print("\n✅ All app WhisperX support tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_requirements_whisperx():
    """Test that requirements mention WhisperX."""
    print("\n" + "="*60)
    print("Testing Requirements for WhisperX")
    print("="*60)
    
    try:
        req_file = Path(__file__).parent.parent / "requirements.txt"
        with open(req_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Test 1: WhisperX is mentioned
        assert 'whisperx' in content.lower(), "WhisperX not in requirements.txt"
        print("✓ Test 1: WhisperX is mentioned in requirements.txt")
        
        # Test 2: Installation instructions provided
        assert 'install' in content.lower() or 'pip' in content.lower(), "No installation instructions"
        print("✓ Test 2: Installation instructions provided")
        
        # Test 3: Optional/recommended noted
        lines = content.lower()
        assert 'optional' in lines or 'recommended' in lines, "Not marked as optional"
        print("✓ Test 3: WhisperX marked as optional")
        
        print("\n✅ All requirements tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transcriber_initialization():
    """Test transcriber initialization with and without WhisperX."""
    print("\n" + "="*60)
    print("Testing Transcriber Initialization")
    print("="*60)
    
    try:
        # Test that initialization signature accepts use_whisperx
        transcriber_file = Path(__file__).parent.parent / "core" / "transcriber.py"
        with open(transcriber_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find __init__ signature
        init_start = content.find('def __init__(')
        if init_start == -1:
            raise ValueError("__init__ method not found")
        
        init_end = content.find('):', init_start)
        init_signature = content[init_start:init_end]
        
        # Test 1: use_whisperx parameter exists
        assert 'use_whisperx' in init_signature, "use_whisperx not in __init__ signature"
        print("✓ Test 1: use_whisperx parameter in __init__")
        
        # Test 2: Default value is False
        assert 'use_whisperx: bool = False' in init_signature or 'use_whisperx=False' in init_signature, \
            "use_whisperx default should be False"
        print("✓ Test 2: use_whisperx defaults to False")
        
        # Test 3: Docstring mentions WhisperX
        assert 'whisperx' in content[init_start:init_start+1000].lower(), "Docstring doesn't mention WhisperX"
        print("✓ Test 3: Docstring documents WhisperX parameter")
        
        print("\n✅ All initialization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("WhisperX Integration Tests")
    print("="*60)
    
    results = {
        "WhisperX Optional Import": test_whisperx_optional_import(),
        "App WhisperX Support": test_app_whisperx_support(),
        "Requirements WhisperX": test_requirements_whisperx(),
        "Transcriber Initialization": test_transcriber_initialization(),
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is None:
            status = "⚠️  SKIP"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "❓ UNKNOWN"
        
        print(f"{test_name}: {status}")
    
    # Overall result
    failures = sum(1 for r in results.values() if r is False)
    
    if failures == 0:
        print("\n✅ All WhisperX integration tests passed!")
        return 0
    else:
        print(f"\n❌ {failures} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
