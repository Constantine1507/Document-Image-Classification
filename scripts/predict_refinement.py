"""Recreate the final CSV from saved PyTorch correctors without retraining."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from docvision_calibration.data import load_review,file_sha256
from docvision_calibration.models import predict
from docvision_calibration.validation import load_model


def main():
    """Load the original review and the selected fold models; enforce data identity."""
    parser=argparse.ArgumentParser();parser.add_argument('--review-zip',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summary=json.loads((args.run_dir/'summary.json').read_text())
    if file_sha256(args.review_zip)!=summary['data_provenance']['archive_sha256']:raise ValueError('Review archive changed')
    _,sample,_,testing,_=load_review(args.review_zip,args.run_dir/'review_source')
    if summary['promotion_gate_passed']:
        predictions=[]
        for fold in range(summary['outer_folds']):
            config,model=load_model(args.run_dir/f'outer_{fold}/corrector.pt')
            predictions.append(predict(config,model,testing))
        probability=np.mean(predictions,axis=0)
    else:probability=testing.mean(1)
    result=sample[['id']].copy();result['label']=probability.argmax(1)
    args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False)
    if file_sha256(args.output)!=summary['submission_sha256']:raise AssertionError('Submission differs from recorded output')
    print('Reproduced exact submission:',args.output)


if __name__=='__main__':main()
