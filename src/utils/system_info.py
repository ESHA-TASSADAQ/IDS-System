import platform
import sys
import json
from typing import Dict


def collect_system_info() -> Dict[str, str]:
	info = {
		"python_version": sys.version.split()[0],
		"platform": platform.platform(),
		"processor": platform.processor(),
	}
	try:
		import tensorflow as tf  # noqa: F401
		info["tensorflow_version"] = tf.__version__
	except Exception:
		info["tensorflow_version"] = "unknown"
	return info


def save_system_info(path: str) -> None:
	info = collect_system_info()
	with open(path, 'w') as f:
		json.dump(info, f, indent=2)