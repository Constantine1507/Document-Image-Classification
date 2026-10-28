"""Compare global power-mean pooling without claiming test-set accuracy."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from docvision_calibration.data import load_review,write_json,file_sha256
from docvision_calibration.power import grid_predictions,select_grid
from docvision_calibration.validation import partitions,metrics,paired_diagnostics


def main():
    """Fit global pooling rules on fold training rows and assess unseen fold rows."""
    parser=argparse.ArgumentParser();parser.add_argument('--review-zip',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=PROJECT/'outputs/power')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((PROJECT/'configs/refinement/power_plan.json').read_text())
    valid,sample,p,t,metadata=load_review(args.review_zip,args.output/'review_source')
    labels=valid.label.to_numpy();groups=valid.pixel_sha256.to_numpy()
    grid=[{'weight':w,'power':r} for w in plan['weights'] for r in plan['powers']]
    values=grid_predictions(p,grid);test=grid_predictions(t,grid)
    outer=partitions(labels,groups,plan['outer_folds'],plan['seed'])
    baseline=p.mean(1);baseline/=baseline.sum(1,keepdims=True)
    records=[]
    for tolerance in plan['training_selection_tolerances_in_correct_rows']:
        oof=np.zeros_like(baseline);predictions=[];choices=[]
        for fold,(fit,assessment) in enumerate(outer):
            selected=select_grid(values[:,fit],labels[fit],grid,tolerance)
            oof[assessment]=values[selected][:,assessment].mean(0)
            predictions.append(test[selected].mean(0))
            choices.append({'fold':fold,'parameter_indices':selected,'parameters':[grid[i] for i in selected],
                            'assessment_accuracy':metrics(labels[assessment],oof[assessment])['accuracy'],
                            'baseline_accuracy':metrics(labels[assessment],baseline[assessment])['accuracy']})
        candidate='power_best' if tolerance==0 else 'power_plateau'
        probability=np.mean(predictions,axis=0)
        sample[['id']].assign(label=probability.argmax(1)).to_csv(args.output/f'submission_{candidate}.csv',index=False)
        np.savez_compressed(args.output/f'{candidate}_oof.npz',ids=valid.id.to_numpy(dtype=str),labels=labels,probabilities=oof)
        np.savez_compressed(args.output/f'{candidate}_test.npz',ids=sample.id.to_numpy(dtype=str),probabilities=probability)
        report={'candidate':candidate,**metrics(labels,oof),'paired':paired_diagnostics(labels,baseline,oof,groups),
                'choices':choices,'scope':plan['scope'],'kaggle_score':None,'data_provenance':metadata}
        write_json(args.output/f'{candidate}.json',report);records.append({'candidate':candidate,**metrics(labels,oof)})
    # All full-holdout grid scores are exploratory selection diagnostics, not OOF fitting results.
    full=[]
    for cfg,probs in zip(grid,values):full.append({**cfg,**metrics(labels,probs)})
    pd.DataFrame(full).sort_values(['accuracy','log_loss'],ascending=[False,True]).to_csv(args.output/'grid_selection_only.csv',index=False)
    pd.DataFrame(records).to_csv(args.output/'crossfit_comparison.csv',index=False)
    write_json(args.output/'plan.json',plan)
    print(pd.DataFrame(records).to_string(index=False),flush=True)


if __name__=='__main__':main()
