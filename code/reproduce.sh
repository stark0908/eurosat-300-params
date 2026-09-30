#!/usr/bin/env bash
# Reproduction script for EuroSAT Sub-306 Parameter Minimization Breakthrough
#
# Hardware requirements: NVIDIA GPU with CUDA or multicore CPU
# Environment: conda activate torch
# Expected runtime: ~1-2 minutes total

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "================================================================================"
echo "EuroSAT Sub-306 Parameter Minimization: Breakthrough Reproduction Suite"
echo "Project Root: ${PROJECT_ROOT}"
echo "Script Dir:   ${SCRIPT_DIR}"
echo "================================================================================"

# Verify conda environment
if ! command -v python &> /dev/null; then
    echo "Error: Python is not in PATH. Please run 'conda activate torch' first."
    exit 1
fi

echo "[1/3] Running 32-Feature Exhaustive Leave-One-Out Ablation (StandardScaler vs Yeo-Johnson)..."
PYTHONPATH="${PROJECT_ROOT}" python "${SCRIPT_DIR}/ablation_32_leave_one_out.py"

echo ""
echo "[2/3] Running Multi-Scale Benchmark Comparison (33, 32, 31, 29 Features)..."
PYTHONPATH="${PROJECT_ROOT}" python "${SCRIPT_DIR}/benchmark_comparison.py"

echo ""
echo "[3/3] Generating All Publication Figures..."
PYTHONPATH="${PROJECT_ROOT}" python "${SCRIPT_DIR}/generate_figures.py"

echo ""
echo "================================================================================"
echo "Reproduction Suite Completed Successfully!"
echo "Generated CSV Tables:"
echo "  1. ${PROJECT_ROOT}/mohit/results/leave_one_out_32_ablation.csv"
echo "  2. ${PROJECT_ROOT}/mohit/results/benchmark_33_32_31_29.csv"
echo "Generated Publication Figures:"
echo "  1. ${PROJECT_ROOT}/mohit/figures/pareto_frontier_sub306.png"
echo "  2. ${PROJECT_ROOT}/mohit/figures/confusion_matrices_comparison.png"
echo "  3. ${PROJECT_ROOT}/mohit/figures/distribution_pathology_transforms.png"
echo "  4. ${PROJECT_ROOT}/mohit/figures/leave_one_out_32_ranking.png"
echo "================================================================================"
