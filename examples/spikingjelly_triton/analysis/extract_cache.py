import os
import shutil
from pathlib import Path

# Triton 的编译产物默认存储在 ~/.triton/cache 中
cache_dir = Path.home() / ".triton" / "cache"
target_dir = Path("examples/spikingjelly_triton/analysis/triton_passes/cache_dump")

if not target_dir.exists():
    target_dir.mkdir(parents=True, exist_ok=True)

print(f"正在从 {cache_dir} 提取 Triton 编译产生的中间 IR...")

found = False
if cache_dir.exists():
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            if file.endswith((".ttir", ".ttgir", ".llir", ".ptx")):
                src = Path(root) / file
                # 为了防止重名，我们将 hash 目录名加上
                hash_name = Path(root).name
                dst = target_dir / f"{hash_name}_{file}"
                shutil.copy2(src, dst)
                print(f"提取: {dst.name}")
                found = True

if not found:
    print("未找到任何 .ttir / .ttgir / .llir 文件！可能是刚才的环境变量未生效或缓存被清空。")
else:
    print(f"提取成功！请查看 {target_dir} 目录。")
