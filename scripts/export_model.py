import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_engine.audio_engine.benchmark import run_benchmark

X_train = np.load(ROOT / 'audio_engine' / 'data' / 'train_features.npy')
y_train = np.load(ROOT / 'audio_engine' / 'data' / 'train_labels.npy')
X_val = np.load(ROOT / 'audio_engine' / 'data' / 'val_features.npy')
y_val = np.load(ROOT / 'audio_engine' / 'data' / 'val_labels.npy')
X_test = np.load(ROOT / 'audio_engine' / 'data' / 'test_features.npy')
y_test = np.load(ROOT / 'audio_engine' / 'data' / 'test_labels.npy')

model_path = ROOT / 'models' / 'pcg_classifier.onnx'
model_path.parent.mkdir(parents=True, exist_ok=True)
features = np.concatenate([X_train, X_val, X_test], axis=0)
labels = np.concatenate([y_train, y_val, y_test], axis=0)
report = run_benchmark(
    features,
    labels,
    model_path=model_path,
    experiment_name='pcg-benchmark',
)
print(report)
