#!/usr/bin/env python3
"""
环境检查工具 - 检查当前环境与 requirements 文件的差异

用法：
    python check_env.py                    # 检查所有 requirements 文件
    python check_env.py requirements.txt   # 检查指定文件
    python check_env.py --export           # 导出当前环境的精确版本
"""

import subprocess
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def get_installed_packages() -> Dict[str, str]:
    """获取当前环境所有已安装包的版本"""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        capture_output=True,
        text=True
    )
    
    packages = {}
    for line in result.stdout.strip().split("\n"):
        if "==" in line:
            name, version = line.split("==", 1)
            # 统一包名为小写
            packages[name.lower().replace("_", "-")] = version
    
    return packages


def parse_requirements(file_path: str) -> List[Dict]:
    """解析 requirements 文件"""
    requirements = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith("#") or line.startswith("--"):
                continue
            
            # 解析包名和版本约束
            # 支持: package, package==1.0, package>=1.0, package>=1.0,<2.0
            match = re.match(r'^([a-zA-Z0-9_\-\[\]]+)\s*([<>=!~].*)?$', line)
            
            if match:
                name = match.group(1).lower().replace("_", "-")
                # 移除 extras，如 package[extra]
                name = re.sub(r'\[.*\]', '', name)
                version_spec = match.group(2) or ""
                
                requirements.append({
                    "name": name,
                    "spec": version_spec,
                    "line": line,
                    "line_num": line_num
                })
    
    return requirements


def parse_version_spec(spec: str) -> List[Tuple[str, str]]:
    """解析版本约束，返回 [(operator, version), ...]"""
    if not spec:
        return []
    
    constraints = []
    # 匹配 >=1.0, ==1.0, <2.0 等
    pattern = r'([<>=!~]+)\s*([0-9a-zA-Z\.\-\+]+)'
    for match in re.finditer(pattern, spec):
        constraints.append((match.group(1), match.group(2)))
    
    return constraints


def version_to_tuple(version: str) -> Tuple:
    """将版本字符串转换为可比较的元组"""
    # 移除后缀如 +cu118, .post1 等
    version = re.sub(r'[\+\.]?(cu\d+|cpu|post\d+|dev\d*)$', '', version)
    
    parts = []
    for part in version.split("."):
        # 尝试转换为数字
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(part)
    
    return tuple(parts)


def check_version_constraint(installed_version: str, constraints: List[Tuple[str, str]]) -> Tuple[bool, str]:
    """检查已安装版本是否满足约束"""
    if not constraints:
        return True, "无版本要求"
    
    installed_tuple = version_to_tuple(installed_version)
    
    for op, required_version in constraints:
        required_tuple = version_to_tuple(required_version)
        
        if op == "==":
            # 精确匹配（允许后缀差异）
            if installed_tuple != required_tuple:
                return False, f"需要 =={required_version}，已安装 {installed_version}"
        elif op == ">=":
            if installed_tuple < required_tuple:
                return False, f"需要 >={required_version}，已安装 {installed_version}"
        elif op == "<=":
            if installed_tuple > required_tuple:
                return False, f"需要 <={required_version}，已安装 {installed_version}"
        elif op == ">":
            if installed_tuple <= required_tuple:
                return False, f"需要 >{required_version}，已安装 {installed_version}"
        elif op == "<":
            if installed_tuple >= required_tuple:
                return False, f"需要 <{required_version}，已安装 {installed_version}"
        elif op == "!=":
            if installed_tuple == required_tuple:
                return False, f"不能是 {required_version}，已安装 {installed_version}"
        elif op == "~=":
            # 兼容版本，如 ~=1.4.2 表示 >=1.4.2, ==1.4.*
            if installed_tuple < required_tuple:
                return False, f"需要 ~={required_version}，已安装 {installed_version}"
    
    return True, "✓"


def check_requirements(req_file: str, installed: Dict[str, str]) -> Dict:
    """检查 requirements 文件与当前环境的差异"""
    requirements = parse_requirements(req_file)
    
    results = {
        "file": req_file,
        "matched": [],      # 版本匹配
        "mismatched": [],   # 版本不匹配
        "missing": [],      # 未安装
        "extra_installed": []  # 已安装但不在 requirements 中（可选）
    }
    
    required_names = set()
    
    for req in requirements:
        name = req["name"]
        required_names.add(name)
        
        if name in installed:
            installed_version = installed[name]
            constraints = parse_version_spec(req["spec"])
            is_ok, message = check_version_constraint(installed_version, constraints)
            
            if is_ok:
                results["matched"].append({
                    "name": name,
                    "required": req["spec"] or "any",
                    "installed": installed_version
                })
            else:
                results["mismatched"].append({
                    "name": name,
                    "required": req["spec"],
                    "installed": installed_version,
                    "message": message
                })
        else:
            results["missing"].append({
                "name": name,
                "required": req["spec"] or "any"
            })
    
    return results


def print_report(results: Dict):
    """打印检查报告"""
    print("\n" + "=" * 70)
    print(f"📋 检查文件: {results['file']}")
    print("=" * 70)
    
    # 统计
    total = len(results["matched"]) + len(results["mismatched"]) + len(results["missing"])
    print(f"\n📊 总计: {total} 个包")
    print(f"   ✅ 匹配: {len(results['matched'])}")
    print(f"   ⚠️  版本不符: {len(results['mismatched'])}")
    print(f"   ❌ 未安装: {len(results['missing'])}")
    
    # 版本不匹配
    if results["mismatched"]:
        print(f"\n⚠️  版本不匹配 ({len(results['mismatched'])} 个):")
        print("-" * 50)
        for item in results["mismatched"]:
            print(f"  {item['name']}")
            print(f"    要求: {item['required']}")
            print(f"    已装: {item['installed']}")
            print(f"    问题: {item['message']}")
    
    # 未安装
    if results["missing"]:
        print(f"\n❌ 未安装 ({len(results['missing'])} 个):")
        print("-" * 50)
        for item in results["missing"]:
            print(f"  {item['name']} {item['required']}")
    
    # 匹配的包（可选显示）
    if results["matched"] and len(results["mismatched"]) == 0 and len(results["missing"]) == 0:
        print(f"\n✅ 所有 {len(results['matched'])} 个包版本匹配！")
    
    print()


def export_current_env(output_file: str = "requirements-lock.txt"):
    """导出当前环境的精确版本"""
    installed = get_installed_packages()
    
    # 按名称排序
    sorted_packages = sorted(installed.items())
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Auto-generated requirements lock file\n")
        f.write("# Generated from current working environment\n")
        f.write(f"# Python version: {sys.version.split()[0]}\n")
        f.write("#\n")
        f.write("# This file contains exact versions that are known to work together.\n")
        f.write("# Use: pip install -r requirements-lock.txt\n")
        f.write("#\n\n")
        
        for name, version in sorted_packages:
            f.write(f"{name}=={version}\n")
    
    print(f"✅ 已导出 {len(sorted_packages)} 个包到 {output_file}")
    return output_file


def export_filtered_requirements(
    base_requirements: str,
    output_file: str = "requirements-exact.txt"
):
    """基于 requirements 文件导出精确版本（只包含指定的包）"""
    installed = get_installed_packages()
    requirements = parse_requirements(base_requirements)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Exact versions based on {base_requirements}\n")
        f.write(f"# Python version: {sys.version.split()[0]}\n")
        f.write("#\n\n")
        
        for req in requirements:
            name = req["name"]
            if name in installed:
                f.write(f"{name}=={installed[name]}\n")
            else:
                f.write(f"# NOT INSTALLED: {req['line']}\n")
    
    print(f"✅ 已导出精确版本到 {output_file}")
    return output_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="检查环境与 requirements 文件的差异")
    parser.add_argument("files", nargs="*", default=[], help="要检查的 requirements 文件")
    parser.add_argument("--export", action="store_true", help="导出当前环境的精确版本")
    parser.add_argument("--export-filtered", type=str, help="基于指定文件导出精确版本")
    parser.add_argument("--all", action="store_true", help="检查所有 requirements*.txt 文件")
    
    args = parser.parse_args()
    
    # 导出模式
    if args.export:
        export_current_env()
        return
    
    if args.export_filtered:
        export_filtered_requirements(args.export_filtered)
        return
    
    # 确定要检查的文件
    files_to_check = args.files
    
    if args.all or not files_to_check:
        # 查找所有 requirements 文件
        current_dir = Path(".")
        files_to_check = list(current_dir.glob("requirements*.txt"))
        files_to_check = [str(f) for f in files_to_check]
    
    if not files_to_check:
        print("❌ 未找到 requirements 文件")
        print("   请指定文件或在当前目录放置 requirements.txt")
        return
    
    # 获取已安装的包
    print("🔍 正在获取已安装的包...")
    installed = get_installed_packages()
    print(f"   找到 {len(installed)} 个已安装的包")
    
    # 检查每个文件
    all_ok = True
    for req_file in files_to_check:
        if Path(req_file).exists():
            results = check_requirements(req_file, installed)
            print_report(results)
            
            if results["mismatched"] or results["missing"]:
                all_ok = False
        else:
            print(f"⚠️  文件不存在: {req_file}")
    
    # 总结建议
    print("\n" + "=" * 70)
    print("💡 建议")
    print("=" * 70)
    
    if all_ok:
        print("✅ 环境检查通过！")
        print("\n建议：导出精确版本供他人使用")
        print("  python check_env.py --export")
    else:
        print("⚠️  存在版本差异，建议：")
        print("\n1. 安装缺失的包：")
        print("   pip install -r requirements.txt")
        print("\n2. 或者导出当前工作环境的精确版本：")
        print("   python check_env.py --export")
        print("   然后让其他人使用 requirements-lock.txt")


if __name__ == "__main__":
    main()
