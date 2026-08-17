#!/usr/bin/env bash
# Install Python packages needed for the homotopy training scripts.
# Run this after a container restart: bash setup_env.sh
set -e

echo "=== Bootstrapping pip ==="
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --break-system-packages

echo ""
echo "=== Installing packages ==="
# jax 0.10.2 + jaxlib 0.10.2 (CPU build) — versions used in original session.
# --break-system-packages is needed on Debian-managed Python.
python3 -m pip install --break-system-packages \
    "jax==0.10.2" \
    "jaxlib==0.10.2" \
    numpy \
    scipy \
    matplotlib \
    pandas

echo ""
echo "=== Verifying ==="
python3 -c "
import jax, jax.numpy as jnp, numpy as np, matplotlib
print('jax      ', jax.__version__)
print('numpy    ', np.__version__)
print('matplotlib', matplotlib.__version__)
print('devices  ', jax.local_devices())
"

echo ""
echo "Done. To run training:"
echo "  cd /workspace/project"
echo "  python3 visualize_training.py"
echo "  python3 test_cases.py"
