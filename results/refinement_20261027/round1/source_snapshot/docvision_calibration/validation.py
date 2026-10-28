"""Nested group cross-validation, reproducible selection and prediction export."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import binomtest

from .data import file_sha256, write_json
from .models import fit, predict, ProbabilityCorrector


def metrics(labels, probability):
    """Report accuracy and proper probability scores for the 16 fixed classes."""
    prediction=probability.argmax(1)
    return {'accuracy':float(accuracy_score(labels,prediction)),
            'correct':int(np.sum(prediction==labels)),
            'macro_f1':float(f1_score(labels,prediction,average='macro',labels=list(range(16)),zero_division=0)),
            'log_loss':float(log_loss(labels,probability,labels=list(range(16))))}


def partitions(labels, groups, folds, seed):
    """Build stratified folds that never separate identical-pixel groups."""
    splitter=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed)
    result=list(splitter.split(np.zeros(len(labels)),labels,groups))
    assigned=np.zeros(len(labels),dtype=int)
    for train,validation in result:
        if set(groups[train])&set(groups[validation]):raise AssertionError('Group leakage')
        if set(labels[train])!=set(range(16)):raise ValueError('Training fold lacks a class')
        assigned[validation]+=1
    if not np.all(assigned==1):raise AssertionError('Each assessment row must occur exactly once')
    return result


def save_model(path, model, config, metadata):
    """Save tensors and primitive metadata compatible with weights_only loading."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'format_version':1,'config':config,'metadata':metadata,
                'state_dict':None if model is None else model.state_dict()},path)


def load_model(path):
    """Restore an already-fitted corrector without fitting on additional labels."""
    content=torch.load(path,map_location='cpu',weights_only=True);config=content['config']
    if content['state_dict'] is None:return config,None
    dtype=torch.float32 if config['method']=='mlp' else torch.float64
    model=ProbabilityCorrector(config).to(dtype=dtype)
    model.load_state_dict(content['state_dict'],strict=True);model.eval()
    return config,model


def paired_diagnostics(labels, baseline, candidate, groups):
    """Compute descriptive group bootstrap intervals and paired error transitions."""
    base_ok=baseline.argmax(1)==labels;new_ok=candidate.argmax(1)==labels
    recovered=int(np.sum(~base_ok&new_ok));broken=int(np.sum(base_ok&~new_ok))
    _,inverse=np.unique(groups,return_inverse=True)
    counts=np.bincount(inverse);delta=np.bincount(inverse,weights=new_ok.astype(float)-base_ok)
    generator=np.random.default_rng(1729);differences=[]
    for _ in range(2000):
        sample=generator.integers(0,len(counts),size=len(counts))
        differences.append(delta[sample].sum()/counts[sample].sum())
    interval=np.quantile(differences,[.025,.975])
    return {'recovered':recovered,'broken':broken,'net_correct_gain':recovered-broken,
            'still_wrong':int(np.sum(~base_ok&~new_ok)),
            'descriptive_95pct_delta_interval':interval.tolist(),
            'mcnemar_exact_p_descriptive':float(binomtest(recovered,recovered+broken,.5).pvalue) if recovered+broken else 1.,
            'inference_limitation':'Descriptive only; shared base-model selection and overlapping CV training sets limit formal inference.'}


def run_nested(valid, sample, probability, testing, metadata, plan, output):
    """Select corrections inside each outer fold; preserve all honest assessment predictions."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    labels=valid.label.to_numpy(dtype=int);groups=valid.pixel_sha256.to_numpy(dtype=str)
    source_root=Path(__file__).parent
    signature={'data':metadata,'plan':plan,'source_sha256':{p.name:file_sha256(p) for p in sorted(source_root.glob('*.py'))}}
    signature_path=output/'run_signature.json'
    if signature_path.is_file() and json.loads(signature_path.read_text())!=signature:
        raise ValueError('Data/config/source changed; use a new output directory')
    write_json(signature_path,signature)
    outer=partitions(labels,groups,plan['outer_folds'],plan['outer_seed'])
    outer_fold_ids=np.full(len(labels),-1,dtype=int)
    for fold,(_,assessment) in enumerate(outer):outer_fold_ids[assessment]=fold
    pd.DataFrame({'id':valid.id,'label':labels,'pixel_sha256':groups,'outer_fold':outer_fold_ids}).to_csv(output/'folds.csv',index=False)
    baseline=probability.mean(1);oof=np.zeros_like(baseline);test_predictions=[];fold_records=[];all_inner=[]
    started=time.perf_counter()
    for fold,(training,assessment) in enumerate(outer):
        folder=output/f'outer_{fold}';folder.mkdir(parents=True,exist_ok=True)
        cached=folder/'completed.json'
        if cached.is_file():
            record=json.loads(cached.read_text())
            with np.load(folder/'assessment.npz',allow_pickle=False) as content:
                if not np.array_equal(content['indices'],assessment):raise ValueError('Cache assessment indices changed')
                oof[assessment]=content['probabilities']
            with np.load(folder/'test.npz',allow_pickle=False) as content:test_predictions.append(content['probabilities'])
            fold_records.append(record);all_inner.append(pd.read_csv(folder/'inner_comparison.csv'))
            print('Reused completed outer fold',fold,flush=True);continue
        print(f'OUTER {fold+1}/{len(outer)} | meta train={len(training)} | assessment={len(assessment)}',flush=True)
        inner=partitions(labels[training],groups[training],plan['inner_folds'],plan['outer_seed']+100+fold)
        comparison=[]
        for config in plan['methods']:
            path=folder/f"inner_{config['name']}.npz"
            if path.is_file():
                with np.load(path,allow_pickle=False) as content:prediction=content['probabilities']
            else:
                prediction=np.zeros((len(training),16))
                for inner_fold,(inner_fit,inner_assess) in enumerate(inner):
                    train_indices=training[inner_fit];assess_indices=training[inner_assess]
                    model=fit(config,probability[train_indices],labels[train_indices],seed=1729+fold*100+inner_fold)
                    prediction[inner_assess]=predict(config,model,probability[assess_indices])
                    del model
                np.savez_compressed(path,probabilities=prediction)
            values=metrics(labels[training],prediction)
            comparison.append(dict(outer_fold=fold,candidate=config['name'],**values))
            print('  inner',config['name'],round(values['accuracy'],6),round(values['log_loss'],6),flush=True)
        table=pd.DataFrame(comparison).sort_values(['accuracy','log_loss'],ascending=[False,True])
        reference=float(table.loc[table.candidate.eq('baseline_equal'),'accuracy'].iloc[0])
        selected=str(table.iloc[0].candidate)
        if float(table.iloc[0].accuracy)<reference+plan['minimum_gain_for_selection']:selected='baseline_equal'
        config=next(c for c in plan['methods'] if c['name']==selected)
        # Selection is frozen here, before computing any assessment metric.
        write_json(folder/'frozen_selection.json',{'selected':selected,'config':config,'inner_metrics':table.to_dict('records')})
        model=fit(config,probability[training],labels[training],seed=50000+fold)
        selected_probability=predict(config,model,probability[assessment]);oof[assessment]=selected_probability
        test_probability=predict(config,model,testing);test_predictions.append(test_probability)
        save_model(folder/'corrector.pt',model,config,{'fold':fold,'fingerprint':metadata['fingerprint'],
                   'training_ids':valid.id.iloc[training].tolist(),'assessment_ids':valid.id.iloc[assessment].tolist()})
        np.savez_compressed(folder/'assessment.npz',indices=assessment,probabilities=selected_probability)
        np.savez_compressed(folder/'test.npz',ids=sample.id.to_numpy(dtype=str),probabilities=test_probability)
        table.to_csv(folder/'inner_comparison.csv',index=False);all_inner.append(table)
        values=metrics(labels[assessment],selected_probability);reference=metrics(labels[assessment],baseline[assessment])
        record={'outer_fold':fold,'selected':selected,'rows':len(assessment),**values,
                'baseline_accuracy':reference['accuracy'],'delta':values['accuracy']-reference['accuracy']}
        write_json(cached,record);fold_records.append(record)
        print('OUTER RESULT',json.dumps(record),flush=True)
        del model
    comparison=pd.DataFrame(fold_records);comparison.to_csv(output/'outer_comparison.csv',index=False)
    inner_table=pd.concat(all_inner,ignore_index=True);inner_table.to_csv(output/'all_inner_comparisons.csv',index=False)
    inner_table.groupby('candidate')[['accuracy','log_loss']].mean().sort_values('accuracy',ascending=False).to_csv(output/'inner_candidate_averages.csv')
    baseline_metrics=metrics(labels,baseline);nested_metrics=metrics(labels,oof)
    gain=nested_metrics['accuracy']-baseline_metrics['accuracy']
    positive_folds=int(np.sum(comparison.delta>=-1e-12))
    passed=gain>=plan['minimum_gain_for_selection'] and positive_folds>=3
    corrected_test=np.mean(test_predictions,axis=0)
    final_probability=corrected_test if passed else testing.mean(1)
    result=sample[['id']].copy();result['label']=final_probability.argmax(1).astype(int)
    if result.isna().any().any() or not np.array_equal(result.id,sample.id):raise AssertionError('Submission IDs changed')
    result.to_csv(output/'submission.csv',index=False)
    sample[['id']].assign(label=corrected_test.argmax(1).astype(int)).to_csv(output/'submission_nested_candidate.csv',index=False)
    sample.to_csv(output/'submission_original_baseline.csv',index=False)
    np.savez_compressed(output/'nested_oof.npz',ids=valid.id.to_numpy(dtype=str),labels=labels,probabilities=oof)
    np.savez_compressed(output/'test_probabilities.npz',ids=sample.id.to_numpy(dtype=str),probabilities=final_probability)
    diagnostics=paired_diagnostics(labels,baseline,oof,groups)
    transitions=pd.DataFrame({'id':valid.id,'label':labels,'baseline_prediction':baseline.argmax(1),
                              'nested_prediction':oof.argmax(1),'outer_fold':outer_fold_ids,'pixel_sha256':groups})
    transitions['status']=np.select([(transitions.baseline_prediction!=labels)&(transitions.nested_prediction==labels),
                                    (transitions.baseline_prediction==labels)&(transitions.nested_prediction!=labels),
                                    transitions.nested_prediction!=labels],['recovered','broken','still_wrong'],default='still_correct')
    transitions.to_csv(output/'validation_transitions.csv',index=False)
    per_class=[]
    for name,probs in [('baseline_equal',baseline),('nested_selected',oof)]:
        precision,recall,f1,support=precision_recall_fscore_support(labels,probs.argmax(1),labels=list(range(16)),zero_division=0)
        for i in range(16):per_class.append({'method':name,'label':i,'class_name':metadata['class_names'][i],
                   'precision':precision[i],'recall':recall[i],'f1':f1[i],'support':int(support[i]),
                   'correct':int(np.sum((labels==i)&(probs.argmax(1)==i)))})
        pd.DataFrame(confusion_matrix(labels,probs.argmax(1),labels=list(range(16))),
                     index=metadata['class_names'],columns=metadata['class_names']).to_csv(output/f'confusion_{name}.csv')
    pd.DataFrame(per_class).to_csv(output/'per_class_comparison.csv',index=False)
    changed=result.label.to_numpy()!=sample.label.to_numpy()
    pd.DataFrame({'id':sample.id[changed],'baseline_label':sample.label[changed],
                  'new_label':result.label[changed]}).to_csv(output/'test_changed_predictions.csv',index=False)
    report={'baseline_validation':baseline_metrics,'nested_meta_validation':nested_metrics,'accuracy_gain':gain,
            'paired_diagnostics':diagnostics,'nondecreasing_outer_folds':positive_folds,'outer_folds':len(outer),
            'selected_methods':[r['selected'] for r in fold_records],'promotion_gate_passed':passed,
            'submission_policy':'mean of the selected outer-fit corrector probabilities' if passed else 'unchanged original equal ensemble',
            'changed_test_predictions':int(np.sum(changed)),'submission_rows':len(result),
            'submission_sha256':file_sha256(output/'submission.csv'),'reported_baseline_kaggle_score':.88456,
            'new_kaggle_score':None,'target_above_0_90_on_nested_validation':nested_metrics['accuracy']>.9,
            'scope':plan['scope'],'test_policy':plan['test_policy'],'elapsed_seconds':time.perf_counter()-started,
            'data_provenance':metadata,'torch':str(torch.__version__)}
    write_json(output/'summary.json',report)
    print('FINAL',json.dumps(report,indent=2),flush=True)
    return report
