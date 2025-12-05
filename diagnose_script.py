"""
诊断模型加载问题的脚本
运行此脚本找出具体哪里出错
"""

import sys
import traceback

def test_imports():
    """测试基础导入"""
    print("=" * 50)
    print("测试 1: 基础库导入")
    print("=" * 50)
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"✗ PyTorch 导入失败: {e}")
        return False
    
    try:
        import torchaudio
        print(f"✓ torchaudio: {torchaudio.__version__}")
    except Exception as e:
        print(f"✗ torchaudio 导入失败: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ transformers: {transformers.__version__}")
    except Exception as e:
        print(f"✗ transformers 导入失败: {e}")
        return False
    
    try:
        import whisperx
        print(f"✓ WhisperX 可用")
    except Exception as e:
        print(f"✗ WhisperX 导入失败: {e}")
    
    return True

def test_torchaudio_api():
    """测试 torchaudio API"""
    print("\n" + "=" * 50)
    print("测试 2: torchaudio API")
    print("=" * 50)
    
    import torchaudio
    
    # 检查 AudioMetaData
    if hasattr(torchaudio, 'AudioMetaData'):
        print("✓ torchaudio.AudioMetaData 存在")
    else:
        print("✗ torchaudio.AudioMetaData 不存在（新版本已移除）")
        print("  建议: 使用 torchaudio.info() 替代")
    
    # 检查替代方法
    if hasattr(torchaudio, 'info'):
        print("✓ torchaudio.info() 可用（推荐使用）")
    
    # 列出可用的属性
    print("\ntorchaudio 主要属性:")
    attrs = [attr for attr in dir(torchaudio) if not attr.startswith('_')]
    for attr in sorted(attrs)[:10]:
        print(f"  - {attr}")

def test_transformers_models():
    """测试 transformers 模型加载"""
    print("\n" + "=" * 50)
    print("测试 3: Transformers 模型")
    print("=" * 50)
    
    try:
        from transformers import AutoModel, AutoConfig
        print("✓ AutoModel 和 AutoConfig 导入成功")
        
        # 测试 WavLM
        try:
            from transformers import WavLMModel
            print("✓ WavLMModel 可导入")
            
            # 尝试加载配置（不下载模型）
            try:
                config = AutoConfig.from_pretrained("microsoft/wavlm-base")
                print("✓ 可以访问 WavLM 配置")
            except Exception as e:
                print(f"✗ 无法访问 WavLM 配置: {e}")
                print("  可能需要网络连接或本地模型文件")
        
        except ImportError as e:
            print(f"✗ WavLMModel 导入失败: {e}")
            print("\n详细错误信息:")
            traceback.print_exc()
    
    except Exception as e:
        print(f"✗ transformers 模型测试失败: {e}")
        traceback.print_exc()

def test_module_loading():
    """测试你的模块加载"""
    print("\n" + "=" * 50)
    print("测试 4: 自定义模块加载")
    print("=" * 50)
    
    modules = [
        ("audio_aligner", "ChineseAudioAligner"),
        ("acoustic_scorer", "AcousticScorer"),
        ("tone_scorer", "ToneScorer"),
        ("error_classifier", "ErrorClassifier"),
    ]
    
    for module_name, class_name in modules:
        try:
            print(f"\n测试 {module_name}.{class_name}...")
            exec(f"from core.chinese.{module_name} import {class_name}")
            print(f"✓ {class_name} 导入成功")
            
            # 尝试实例化
            try:
                exec(f"obj = {class_name}(device='cpu')")
                print(f"✓ {class_name} 实例化成功")
                
                # 检查是否可用
                try:
                    exec(f"available = obj.is_available()")
                    exec(f"print(f'  可用性: {{available}}')")
                except:
                    pass
                
            except Exception as e:
                print(f"✗ {class_name} 实例化失败: {e}")
                traceback.print_exc()
        
        except ImportError as e:
            print(f"✗ {class_name} 导入失败: {e}")
            print("  详细错误:")
            traceback.print_exc()

def main():
    print("开始诊断...\n")
    
    # 测试 1: 基础导入
    if not test_imports():
        print("\n基础导入失败，请先解决 PyTorch/torchaudio/transformers 安装问题")
        return
    
    # 测试 2: torchaudio API
    test_torchaudio_api()
    
    # 测试 3: transformers 模型
    test_transformers_models()
    
    # 测试 4: 自定义模块
    test_module_loading()
    
    print("\n" + "=" * 50)
    print("诊断完成")
    print("=" * 50)

if __name__ == "__main__":
    main()
