import sys
import torch
import ultralytics

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    print("Device count:", torch.cuda.device_count())
    print("Device memory (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
else:
    print("PyTorch is running on CPU.")

print("Ultralytics version:", ultralytics.__version__)

