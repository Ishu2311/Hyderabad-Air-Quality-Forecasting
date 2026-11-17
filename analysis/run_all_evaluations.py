"""
Master Pipeline Runner for Paper Evaluation Analysis
Runs Steps 1 through 5 in sequence.
"""

import sys
from pathlib import Path

# Add current folder to path
cur_dir = Path(__file__).resolve().parent
if str(cur_dir) not in sys.path:
    sys.path.insert(0, str(cur_dir))

from step1_bollaram_safe import run_bollaram_safe_evaluation
from step2_seasonal_robustness import run_seasonal_robustness
from step3_significance_tests import run_significance_tests
from step4_spatial_baselines import run_spatial_baselines
from step5_shap_scale import run_shap_scale_evaluation


def main():
    print("=" * 70)
    print("RUNNING ALL EVALUATION MODULES FOR PAPER REVISION")
    print("=" * 70)
    
    print("\n>>> EXECUTING STEP 1: Bollaram-Safe Multi-Horizon Evaluation...")
    run_bollaram_safe_evaluation()
    
    print("\n>>> EXECUTING STEP 2: Seasonal Robustness and Skill Scores...")
    run_seasonal_robustness()
    
    print("\n>>> EXECUTING STEP 3: Statistical Significance Testing...")
    run_significance_tests()
    
    print("\n>>> EXECUTING STEP 4: Leave-One-Station-Out Spatial Baselines...")
    run_spatial_baselines()
    
    print("\n>>> EXECUTING STEP 5: Large-Sample Stratified TreeSHAP Explainability...")
    run_shap_scale_evaluation()
    
    print("\n" + "=" * 70)
    print("ALL EVALUATION STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
