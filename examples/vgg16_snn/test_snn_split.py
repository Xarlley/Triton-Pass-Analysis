import triton
import triton.language as tl
import torch

@triton.jit
def snn_kernel(out_ptr, SNN_FLAG: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # 模拟一个 T=4 的时间步循环
    # 在 Triton 中 for 循环可以通过 tl.range 来触发
    acc = 0.0
    for t in tl.static_range(0, 4, 1):
        acc += 1.0
        
    tl.store(out_ptr + pid, acc)

def main():
    print("Compiling SNN kernel with SNN_FLAG=True...")
    out = torch.zeros(10, device='cuda')
    # 触发 Triton 编译
    snn_kernel[(10,)](out, SNN_FLAG=True)
    print("Execution finished.")

if __name__ == "__main__":
    main()
