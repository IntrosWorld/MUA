import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from src.separation import separate
print("Testing SepFormer-WHAMR-16K + IRM pipeline...")
result = separate(mixed_path='data/mixed/demo_mixed.wav', num_speakers=2, demo_oracle=False)
print('Mode   :', result['mode'])
print('Warning:', result.get('warning'))
for p in result['separated_paths']:
    size = os.path.getsize(p) // 1024
    print(f'Output : {p}  ({size} KB)')
print("DONE")
