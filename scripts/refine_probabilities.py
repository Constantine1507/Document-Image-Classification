"""Run reproducible nested validation and generate a Kaggle submission."""
import argparse
import json
import sys
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from docvision_calibration.data import load_review,write_json
from docvision_calibration.validation import run_nested


def main():
    """Parse explicit artifact paths and execute the fixed experiment plan."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--review-zip',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=PROJECT/'outputs/nested')
    parser.add_argument('--plan',type=Path,default=PROJECT/'configs/refinement/experiment_plan.json')
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    valid,sample,probability,testing,metadata=load_review(args.review_zip,args.output/'review_source')
    write_json(args.output/'data_audit.json',metadata)
    print('Verified baseline, rows:',len(valid),len(sample),flush=True)
    run_nested(valid,sample,probability,testing,metadata,plan,args.output)


if __name__=='__main__':main()
