"""Write the simple primary submission and a clearly labeled exploratory alternative."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from docvision_calibration.data import load_review,write_json,file_sha256
from docvision_calibration.power import PowerMeanPool
from docvision_calibration.validation import metrics,paired_diagnostics


def main():
    """Produce test labels using exact ID alignment and report validation scope honestly."""
    parser=argparse.ArgumentParser();parser.add_argument('--review-zip',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=PROJECT/'outputs/refinement_final')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    valid,sample,p,t,metadata=load_review(args.review_zip,args.output/'review_source')
    y=valid.label.to_numpy();baseline=p.mean(1);baseline/=baseline.sum(1,keepdims=True)
    records=[]
    for filename,weight,power,status in [
        ('submission.csv',.5,0.,'Primary fixed geometric mean; selected after exploratory comparison, not new Kaggle accuracy'),
        ('submission_exploratory_power.csv',.35,-1.,'Exploratory parameters selected using the whole validation set; score is optimistic selection accuracy')]:
        model=PowerMeanPool(weight,power)
        with torch.inference_mode():
            val=model(torch.as_tensor(p,dtype=torch.float64)).numpy()
            test=model(torch.as_tensor(t,dtype=torch.float64)).numpy()
        frame=sample[['id']].copy();frame['label']=test.argmax(1).astype(int)
        if len(frame)!=7994 or frame.id.duplicated().any() or not np.array_equal(frame.id,sample.id):raise AssertionError('Submission ID failure')
        if list(frame.columns)!=['id','label'] or not frame.label.between(0,15).all():raise AssertionError('Submission schema failure')
        frame.to_csv(args.output/filename,index=False)
        prefix=Path(filename).stem
        differences=frame.label.to_numpy()!=sample.label.to_numpy()
        pd.DataFrame({'id':sample.id[differences],'baseline_label':sample.label[differences],
                      'new_label':frame.label[differences]}).to_csv(args.output/f'{prefix}_changes.csv',index=False)
        np.savez_compressed(args.output/f'{prefix}_validation.npz',ids=valid.id.to_numpy(dtype=str),labels=y,probabilities=val)
        np.savez_compressed(args.output/f'{prefix}_test.npz',ids=sample.id.to_numpy(dtype=str),probabilities=test)
        torch.save({'weights':model.weights,'power':power,'framework':'PyTorch','base_models':metadata['member_names']},args.output/f'{prefix}_pool.pt')
        record={'file':filename,'weight_convnext':weight,'weight_efficientnet':1-weight,'power':power,
                'validation_metrics':metrics(y,val),'paired':paired_diagnostics(y,baseline,val,valid.pixel_sha256.to_numpy()),
                'changed_test_labels':int(differences.sum()),'rows':len(frame),'sha256':file_sha256(args.output/filename),
                'status':status,'kaggle_score':None}
        records.append(record)
    write_json(args.output/'submission_manifest.json',{'primary':records[0],'exploratory':records[1],
               'baseline_validation':metrics(y,baseline),'reported_baseline_kaggle_score':.88456,
               'data_provenance':metadata,'target_above_0_90_achieved':False,
               'choice_reason':'Fixed 50:50 geometric pooling has 5659 correct validation labels and lower log loss than the original. Learned nested selectors failed their promotion gate. Global grid selection has only 5658 honest cross-fitted correct labels, despite 5672 at its in-sample optimum. The simple primary candidate avoids additional fitted parameters; uncertainty remains.'})
    print(json.dumps(records,indent=2),flush=True)


if __name__=='__main__':main()
