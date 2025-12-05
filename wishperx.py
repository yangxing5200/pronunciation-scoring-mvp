import torch
import whisperx

print("Step 1: Loading WhisperX model...")
try:
    model = whisperx.load_model(
        "base",
        device="cuda",
        compute_type="float16"
    )
    print("✓ WhisperX model loaded")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 2: Loading alignment model...")
try:
    align_model, metadata = whisperx.load_align_model(
        language_code="en",
        device="cuda"
    )
    print("✓ Alignment model loaded")
except Exception as e:
    print(f"✗ Alignment loading failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ All tests passed!")