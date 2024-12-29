#import pynvml
#print(f"Allocated: {torch.cuda.memory_allocated()} bytes")
#print(f"Cached: {torch.cuda.memory_reserved()} bytes")
# Initialize NVML
#pynvml.nvmlInit()

# Get the first GPU handle (adjust index for multiple GPUs)
#handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Get memory info
#info = pynvml.nvmlDeviceGetMemoryInfo(handle)

#print(f"Total VRAM: {info.total / (1024**2):.2f} MB")
#print(f"Used VRAM: {info.used / (1024**2):.2f} MB")
#print(f"Free VRAM: {info.free / (1024**2):.2f} MB")
