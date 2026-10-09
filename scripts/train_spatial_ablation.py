#!/usr/bin/env python3
"""Train a prospective, paired uniform-vs-spatial MIL experiment, offline.

T4x2: one independent arm per GPU, with identical initialisation, UID split,
window samples, sampler order, epoch RNG seed and optimisation schedule. The
first pair runs fold0 before the remaining folds. Partial models are retained
for exact resume, but never counted as complete or used for the final holdout.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from rsna.cache_data import (Bags, Corpus, TARGETS, UID, CONTRACT, DEFAULT_CONFIG, atomic_json,
                             discover_inputs, make_plan, sha256, volume_fingerprint)
from rsna.spatial_mil import SpatialTargetMIL, load_checkpoint, save_checkpoint


def versions():
    return {'python':platform.python_version(),
            **{p:importlib.metadata.version(p) for p in ('torch','timm','numpy','pandas','scikit-learn')},
            'cuda':str(torch.version.cuda)}


def query_diagnostics(model):
    """Read the spatial parameters and clipped gradient without changing training."""
    query=model.spatial_queries
    gradient=query.grad
    def finite_norm(tensor):
        if tensor is None:
            return None
        value=float(tensor.detach().float().norm().item())
        return value if math.isfinite(value) else None
    return {'spatial_query_l2':finite_norm(query),
            'spatial_query_gradient_l2_after_clip':finite_norm(gradient)}


def relay_worker_logs(children, offsets, fold, final=False):
    """Relay each complete unread worker line; retain an incomplete line for later."""
    for _,log,arm in children:
        with Path(log.name).open('r',encoding='utf-8',errors='replace') as reader:
            reader.seek(offsets.get(arm,0))
            while True:
                position=reader.tell()
                line=reader.readline()
                if not line:
                    break
                if not line.endswith('\n') and not final:
                    reader.seek(position)
                    break
                print('WORKER_LOG',f'{arm}_fold{fold}',line.rstrip('\n'),flush=True)
            offsets[arm]=reader.tell()


def weak_scores(truth, pred):
    scores={}
    for j,target in enumerate(TARGETS):
        known=np.isfinite(truth[:,j]) & (np.abs(truth[:,j]-.5)>1e-6)
        binary=(truth[known,j]>.5).astype(np.uint8)
        if len(np.unique(binary))<2:
            raise ValueError(f'Both weak-label classes absent for {target}')
        scores[target]=float(roc_auc_score(binary,pred[known,j]))
    return float(np.mean(list(scores.values()))),scores


def source_paths():
    base=Path(__file__).resolve().parents[1]
    return {'train_spatial_ablation.py':Path(__file__).resolve(),
            'cache_data.py':base/'src/rsna/cache_data.py','spatial_mil.py':base/'src/rsna/spatial_mil.py'}


def verify_plan(plan_path):
    plan_path=Path(plan_path)
    plan=json.loads(plan_path.read_text())
    if plan['pixel_contract']!=CONTRACT:
        raise ValueError('Pixel contract changed')
    for name,path in source_paths().items():
        if plan['source_sha256'].get(name)!=sha256(path):
            raise ValueError(f'Source changed after split lock: {name}')
    for name,key in [('split_manifest.csv','split_manifest_sha256'),('targets.npz','targets_sha256')]:
        if sha256(plan_path.parent/name)!=plan[key]:
            raise ValueError('Split or target table changed after lock')
    for name,digest in plan['inputs']['alignment'].items():
        if sha256(plan['corpus_paths'][name])!=digest:
            raise ValueError(f'Cache alignment changed: {name}')
    for name,fingerprint in plan['inputs']['volumes'].items():
        if volume_fingerprint(plan['corpus_paths'][name],fingerprint['method']=='full_sha256')!=fingerprint:
            raise ValueError(f'Cache volume fingerprint changed: {name}')
    return plan


def initialized_model(config, arm, init_path, seed):
    if sha256(init_path)!=config['initialization_sha256']:
        raise ValueError('Independent ImageNet initialisation SHA256 differs from locked configuration')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model=SpatialTargetMIL(config, spatial=arm=='spatial', pretrained=False)
    initial=torch.load(init_path,map_location='cpu',weights_only=True)
    if initial.get('architecture')!=config['architecture'] or 'state_dict' not in initial:
        raise ValueError('Expected the offline pure ImageNet backbone payload, not a trained competition model')
    pretrained=initial.get('pretrained_config',{})
    if list(pretrained.get('mean',[]))!=config['mean'] or list(pretrained.get('std',[]))!=config['std']:
        raise ValueError('Initial backbone normalisation differs from the fixed image contract')
    model.backbone.load_state_dict(initial['state_dict'],strict=True)
    return model


def make_loader(dataset, config, train, fold, epoch):
    generator=torch.Generator().manual_seed(config['seed']+fold*997+epoch*10007)
    return DataLoader(dataset,batch_size=config['batch_size'] if train else config['eval_batch_size'],
                      shuffle=train,num_workers=config['workers'],pin_memory=True,drop_last=False,
                      persistent_workers=False,prefetch_factor=2 if config['workers'] else None,
                      generator=generator)


@torch.inference_mode()
def predict(model, corpus, rows, values, config, fold, device):
    dataset=Bags(corpus,rows,values['targets'],values['weights'],config,train=False,fold=fold)
    loader=make_loader(dataset,config,False,fold,0)
    model.eval(); indices=[]; probabilities=[]
    for x,s,m,p,_,_,r in loader:
        with torch.autocast('cuda',dtype=torch.float16):
            z=model(x.to(device,non_blocking=True),s.to(device),m.to(device),p.to(device),chunk=32)
        prob=z.float().sigmoid().cpu().numpy()
        if not np.isfinite(prob).all():
            raise FloatingPointError('Nonfinite evaluation scores')
        indices.extend(r.tolist()); probabilities.append(prob)
    ix=np.asarray(indices,np.int64)
    if len(ix)!=len(rows) or len(set(ix))!=len(rows) or set(ix)!=set(rows):
        raise ValueError('Evaluation UID coverage failure')
    return ix,np.concatenate(probabilities)


def train_arm(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Actual training requires CUDA; CPU tests do not train the model')
    torch.cuda.set_device(0); device=torch.device('cuda',0)
    plan=verify_plan(args.plan); config=plan['config']; fold=args.fold; arm=args.arm
    corpus=Corpus(plan['corpus_paths'])
    values=np.load(Path(args.plan).parent/'targets.npz',allow_pickle=False)
    if not np.array_equal(values['ids'],corpus.ids):
        raise ValueError('Cache/target UID order differs')
    roles,folds,groups,gold=values['roles'],values['folds'],values['groups'],values['gold']
    tr=np.flatnonzero((roles=='development') & (folds!=fold))
    va=np.flatnonzero((roles=='development') & (folds==fold))
    held=np.flatnonzero(roles=='holdout')
    if not len(tr) or not len(va) or gold[tr].any() or gold[va].any() \
        or set(groups[tr]) & set(groups[va]) or set(groups[tr]) & set(groups[held]):
        raise ValueError('Gold/holdout/group isolation violation')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    prefix=f'{arm}_fold{fold}'
    status_path=out/(prefix+'_status.json')
    seed=config['seed']+fold*997
    model=initialized_model(config,arm,args.init,seed).to(device)
    # Packages are part of the resume contract; RNG streams cannot be compared
    # across an silently changed torch/timm training implementation.
    contract={'arm':arm,'fold':fold,'config':config,'pixel_contract':CONTRACT,
              'split_manifest_sha256':plan['split_manifest_sha256'],'targets_sha256':plan['targets_sha256'],
              'source_sha256':plan['source_sha256'],'initialization_sha256':config['initialization_sha256'],
              'train_uid_sha256':hashlib.sha256('\n'.join(corpus.ids[tr]).encode()).hexdigest(),
              'validation_uid_sha256':hashlib.sha256('\n'.join(corpus.ids[va]).encode()).hexdigest(),
              'packages':versions(),'gold_gradient_or_selection_rows':0,'holdout_gradient_or_selection_rows':0,
              'selection':'development weak-label macro-AUC; holdout never selects an epoch'}
    if status_path.exists():
        old=json.loads(status_path.read_text())
        if old.get('contract')!=contract:
            raise ValueError('Existing arm status differs from locked experiment contract')
        if old['status']=='complete':
            load_checkpoint(model,out/(prefix+'_best.pt'),contract=contract)
            print('ALREADY_COMPLETE',prefix,flush=True); return
    bb=[p for p in model.backbone.parameters() if p.requires_grad]
    head=[p for n,p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    optimizer=torch.optim.AdamW([{'params':bb,'lr':config['backbone_lr']},
                                {'params':head,'lr':config['head_lr']}],weight_decay=config['weight_decay'])
    dataset=Bags(corpus,tr,values['targets'],values['weights'],config,train=True,fold=fold)
    steps_per_epoch=math.ceil(len(dataset)/config['batch_size'])
    total_steps=steps_per_epoch*config['epochs']
    def schedule(step):
        warmup=max(1,round(total_steps*.15))
        if step<warmup:
            return (step+1)/warmup
        phase=(step-warmup)/max(1,total_steps-warmup)
        return .05+.95*(1+math.cos(math.pi*min(1,phase)))/2
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    scaler=torch.amp.GradScaler('cuda')
    y,w=values['targets'],values['weights']
    prevalence=(y[tr]*w[tr]).sum(0)/w[tr].sum(0).clip(1e-6)
    positive=torch.tensor(np.clip((1-prevalence)/prevalence.clip(.02),1,config['positive_weight_cap']),device=device)
    epoch=next_batch=0; best=-1.; history=[]; accumulated={'loss':0.,'batches':0,'overflows':0,'updates':0}
    latest=out/(prefix+'_latest.pt')
    def checkpoint(epoch_value,batch_value):
        save_checkpoint(model,latest,contract=contract,extras={
            'epoch':epoch_value,'next_batch':batch_value,'best_score':best,'history':history,
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
            'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state(device).cpu(),
            'accumulated':accumulated,'train_prevalence':prevalence.tolist()})
    if latest.exists():
        resume=load_checkpoint(model,latest,contract=contract)
        optimizer.load_state_dict(resume['optimizer']); scheduler.load_state_dict(resume['scheduler']); scaler.load_state_dict(resume['scaler'])
        epoch=int(resume['epoch']); next_batch=int(resume['next_batch']); best=float(resume['best_score']); history=resume['history']
        accumulated=resume['accumulated']; torch.set_rng_state(resume['rng_cpu']); torch.cuda.set_rng_state(resume['rng_cuda'],device)
    else:
        # Reset after model construction so paired arms begin identical RNG streams.
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # Recover the narrow crash window after the last latest checkpoint was
    # written but before the final status JSON was committed.
    if epoch >= config['epochs']:
        if len(history) != config['epochs'] or not (out/(prefix+'_best.pt')).is_file():
            raise ValueError('Completed checkpoint lacks its epoch history or selected model')
        load_checkpoint(model,out/(prefix+'_best.pt'),contract=contract)
        atomic_json({'status':'complete','contract':contract,'completed_epochs':epoch,
                     'best_weak_auc':best,'history':history,'recovered_final_status':True},status_path)
        return
    torch.backends.cudnn.benchmark=False
    started=time.time(); reserve=config['reserve_minutes']*60
    atomic_json({'status':'running','contract':contract,'completed_epochs':epoch,'history':history},status_path)
    while epoch<config['epochs']:
        if time.time()>args.deadline-reserve:
            checkpoint(epoch,next_batch)
            atomic_json({'status':'partial','contract':contract,'completed_epochs':epoch,'next_batch':next_batch,'history':history,
                         'reason':'GPU budget stopped; resumable; not deployable'},status_path)
            return
        dataset.epoch=epoch
        loader=make_loader(dataset,config,True,fold,epoch)
        model.train(); epoch_started=time.time(); consecutive_overflows=0
        for batch_index,(x,s,m,p,target,weight,_) in enumerate(loader):
            if batch_index<next_batch:
                continue
            # A mid-epoch resume skips deterministic loader batches without
            # consuming augmentation/dropout RNG; the checkpoint stores that RNG.
            if time.time()>args.deadline-reserve:
                checkpoint(epoch,batch_index)
                atomic_json({'status':'partial','contract':contract,'completed_epochs':epoch,'next_batch':batch_index,'history':history,
                             'reason':'GPU budget stopped mid epoch; resumable; not deployable'},status_path)
                return
            target,weight=target.to(device),weight.to(device)
            smoothed=target*(1-config['label_smoothing'])+config['label_smoothing']/2
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(x.to(device,non_blocking=True),s.to(device),m.to(device),p.to(device),augment=True)
                loss=(F.binary_cross_entropy_with_logits(logits.float(),smoothed,reduction='none',pos_weight=positive)*weight).sum()/weight.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            before=scaler.get_scale(); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'])
            scaler.step(optimizer); scaler.update()
            overflow=scaler.get_scale()<before
            accumulated['loss']+=float(loss.detach()); accumulated['batches']+=1
            if overflow:
                accumulated['overflows']+=1; consecutive_overflows+=1
                if consecutive_overflows>=10:
                    raise FloatingPointError('Ten consecutive FP16 gradient overflows')
            else:
                consecutive_overflows=0; accumulated['updates']+=1; scheduler.step()
            next_batch=batch_index+1
            if next_batch%100==0:
                checkpoint(epoch,next_batch)
                print(json.dumps({'arm':arm,'fold':fold,'epoch':epoch+1,'batch':next_batch,'of':len(loader),
                                  'loss':float(loss),'updates':accumulated['updates'],'overflows':accumulated['overflows'],
                                  'peak_gpu_gib':torch.cuda.max_memory_allocated(device)/2**30,
                                  **query_diagnostics(model)}),flush=True)
        if accumulated['overflows']>max(10,.05*accumulated['batches']):
            raise FloatingPointError('More than5% of updates overflowed; refusing degraded model')
        ix,probs=predict(model,corpus,va,values,config,fold,device)
        score,by_target=weak_scores(y[ix],probs)
        record={'epoch':epoch+1,'weak_macro_auc':score,'weak_per_target_auc':by_target,
                'loss':accumulated['loss']/max(1,accumulated['batches']),
                'seconds_this_session':time.time()-epoch_started,'updates':accumulated['updates'],
                'overflows':accumulated['overflows'],**query_diagnostics(model)}
        history.append(record)
        if score>best:
            best=score
            save_checkpoint(model,out/(prefix+'_best.pt'),contract=contract,extras={'epoch':epoch+1,'weak_macro_auc':score,
                            'train_prevalence':prevalence.tolist()})
            frame=pd.DataFrame(probs,columns=TARGETS); frame.insert(0,UID,corpus.ids[ix]);frame.to_csv(out/(prefix+'_validation.csv'),index=False)
        epoch+=1; next_batch=0; accumulated={'loss':0.,'batches':0,'overflows':0,'updates':0}
        checkpoint(epoch,0)
        atomic_json({'status':'complete' if epoch==config['epochs'] else 'running','contract':contract,
                     'completed_epochs':epoch,'best_weak_auc':best,'history':history,
                     'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device),'seconds_this_session':time.time()-started},status_path)
        print('EPOCH_COMPLETE',prefix,json.dumps(record),flush=True)
        if epoch<config['epochs'] and time.time()+record['seconds_this_session']*1.2>args.deadline-reserve:
            atomic_json({'status':'partial','contract':contract,'completed_epochs':epoch,'next_batch':0,
                         'best_weak_auc':best,'history':history,'reason':'Insufficient budget for another complete epoch'},status_path)
            return


def completed_pairs(out,config):
    result=[]
    for fold in range(config['n_folds']):
        statuses=[]
        for arm in ('uniform','spatial'):
            p=out/f'{arm}_fold{fold}_status.json'
            statuses.append(json.loads(p.read_text()) if p.exists() else {'status':'missing'})
        if all(s['status']=='complete' and s['completed_epochs']==config['epochs'] for s in statuses):
            result.append(fold)
    return result


@torch.inference_mode()
def holdout_once(plan,out,init_path):
    marker=out/'holdout_once_started.json'
    if marker.exists():
        raise RuntimeError('Holdout already opened; no repeat selection loop is allowed')
    config=plan['config']; complete=completed_pairs(out,config)
    if len(complete)!=config['n_folds']:
        raise ValueError('All paired folds must complete before the locked holdout is opened')
    # Claim the once-only gate atomically before exposing held-out scores.
    with marker.open('x') as f:
        json.dump({'config':config,'pairs':complete,'selection_frozen':True},f,indent=2)
    corpus=Corpus(plan['corpus_paths']); values=np.load(out/'targets.npz',allow_pickle=False)
    rows=np.flatnonzero(values['roles']=='holdout'); y=values['targets'][rows]
    device=torch.device('cuda',0); results={}; predictions={}
    for arm in ('uniform','spatial'):
        family=[]
        for fold in complete:
            status=json.loads((out/f'{arm}_fold{fold}_status.json').read_text())
            model=initialized_model(config,arm,init_path,config['seed']+fold*997).to(device)
            checkpoint=out/f'{arm}_fold{fold}_best.pt'
            load_checkpoint(model,checkpoint,contract=status['contract'])
            ix,p=predict(model,corpus,rows,values,config,fold,device)
            order={int(r):i for i,r in enumerate(ix)}
            family.append(p[[order[int(r)] for r in rows]])
            del model; torch.cuda.empty_cache()
        mean=np.mean(family,axis=0); predictions[arm]=mean
        score,by_target=weak_scores(y,mean)
        results[arm]={'weak_macro_auc':score,'weak_per_target_auc':by_target,
                      'checkpoints':[{ 'file':f'{arm}_fold{fold}_best.pt','sha256':sha256(out/f'{arm}_fold{fold}_best.pt')} for fold in complete]}
        frame=pd.DataFrame(mean,columns=TARGETS); frame.insert(0,UID,corpus.ids[rows]);frame.to_csv(out/f'{arm}_holdout_predictions.csv',index=False)
    groups=values['groups'][rows]; unique=np.unique(groups); rng=np.random.default_rng(config['seed']+73)
    bootstrap=[]
    # Cluster bootstrap respects the same report/known-relation isolation unit.
    for _ in range(1000):
        selected=rng.choice(unique,len(unique),replace=True)
        indices=np.concatenate([np.flatnonzero(groups==g) for g in selected])
        try:
            baseline,_=weak_scores(y[indices],predictions['uniform'][indices])
            spatial,_=weak_scores(y[indices],predictions['spatial'][indices])
        except ValueError:
            continue
        bootstrap.append(spatial-baseline)
    delta=results['spatial']['weak_macro_auc']-results['uniform']['weak_macro_auc']
    receipt={'schema':'rsna_spatial_ablation_holdout_v1','evaluation_count':1,'weak_label_proxy_not_expert_accuracy':True,
             'gold_used_for_gradient_selection_or_this_evaluation':False,'pairs':complete,'rows':len(rows),
             'results':results,'spatial_minus_uniform_weak_macro_auc':delta,
             'group_bootstrap_replicates':len(bootstrap),
             'group_bootstrap_delta_95pct':None if not bootstrap else np.quantile(bootstrap,[.025,.975]).tolist(),
             'promotion_rule':'Research evidence only: no automatic production promotion or promised Kaggle score.'}
    atomic_json(receipt,out/'holdout_comparison.json')
    print('HOLDOUT_ONCE_COMPLETE',json.dumps(receipt),flush=True)


def launch(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to train; CPU CI checks architecture/contracts only')
    config=dict(DEFAULT_CONFIG)
    if args.config:
        config.update(json.loads(Path(args.config).read_text()))
    if args.full_volume_sha256:
        config['full_volume_sha256']=True
    if args.init_sha256:
        config['initialization_sha256']=args.init_sha256
    if args.corpus_paths:
        paths=json.loads(Path(args.corpus_paths).read_text())
        if not args.train_csv or not args.teacher_csv:
            raise ValueError('Explicit corpus paths require train.csv and weak-label CSV paths')
        train_csv,teachers=args.train_csv,args.teacher_csv
    else:
        paths,train_csv,teachers=discover_inputs(args.input_root)
        train_csv=args.train_csv or train_csv; teachers=args.teacher_csv or teachers
    if sha256(args.init)!=config['initialization_sha256']:
        raise ValueError('Initial ImageNet checkpoint hash differs')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    plan=make_plan(config,out,paths=paths,train_csv=train_csv,teacher_csvs=teachers,
                   relations_csv=args.relations_csv,excluded_csv=args.excluded_csv)
    deadline=time.time()+config['max_hours']*3600
    atomic_json({'status':'running','mode':args.mode,'packages':versions(),'config':config,
                 'input_initialization_sha256':sha256(args.init),'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                 'gold_used_for_gradient_or_selection':False,'heldout_used_for_gradient_or_selection':False},out/'experiment_status.json')
    folds=[0] if args.mode=='paired-first' else list(range(config['n_folds']))
    for fold in folds:
        if time.time()>deadline-config['reserve_minutes']*60:
            break
        children=[]; log_offsets={}
        for arm,gpu in [('uniform',0),('spatial',1 if torch.cuda.device_count()>1 else 0)]:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            command=[sys.executable,str(Path(__file__).resolve()),'--worker','--plan',str(out/'plan.json'),
                     '--out',str(out),'--init',args.init,'--arm',arm,'--fold',str(fold),'--deadline',str(deadline)]
            log=open(out/f'{arm}_fold{fold}.log','a',encoding='utf-8')
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append((child,log,arm))
            if torch.cuda.device_count()==1:
                remaining=max(1,deadline-time.time())
                try:
                    child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    child.terminate(); child.wait(timeout=20)
        # Parent provides a hard deadline even if a driver/library hangs.
        while any(p.poll() is None for p,_,_ in children):
            relay_worker_logs(children,log_offsets,fold)
            if time.time()>deadline:
                for p,_,_ in children:
                    if p.poll() is None:
                        p.terminate()
                break
            print('PAIR_RUNNING',fold,{arm:p.poll() for p,_,arm in children},flush=True)
            time.sleep(20)
        failures=[]
        for p,log,arm in children:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
            log.close()
            if p.returncode:
                failures.append({'arm':arm,'fold':fold,'exit_code':p.returncode})
        relay_worker_logs(children,log_offsets,fold,final=True)
        complete=completed_pairs(out,config)
        if failures or fold not in complete:
            atomic_json({'status':'partial_or_failed','completed_pairs':complete,'failures':failures,
                         'partial_models_deployable':False,'training_receipts':str(out)},out/'experiment_status.json')
            print('PAIR_NOT_COMPLETE',fold,'See arm logs/status; resume the same output directory.',flush=True)
            return
        print('PAIR_COMPLETE',fold,flush=True)
    complete=completed_pairs(out,config)
    checkpoints=[]
    for fold in complete:
        for arm in ('uniform','spatial'):
            p=out/f'{arm}_fold{fold}_best.pt'
            checkpoints.append({'arm':arm,'fold':fold,'path':str(p),'sha256':sha256(p)})
    _status='all_pairs_complete' if len(complete)==config['n_folds'] else ('paired_first_complete' if complete else 'no_completed_pairs')
    atomic_json({'status':_status,
                 'completed_pairs':complete,'planned_pairs':config['n_folds'],'config':config,'checkpoints':checkpoints,
                 'partial_models_deployable':False,'gold_used_for_gradient_or_selection':False,
                 'heldout_opened':False,'validation_boundary':plan['validation_boundary']},out/'experiment_status.json')
    if args.holdout_once:
        # Keep the once-only gate unopened when this training session has too
        # little time for a conservative full-family evaluation. A later run
        # skips the complete pairs and receives a fresh evaluation budget.
        if time.time() + max(3600,config['reserve_minutes']*120) > deadline:
            atomic_json({'status':'deferred','reason':'Insufficient remaining GPU budget; rerun --mode all --holdout-once with the same output',
                         'holdout_opened':False},out/'holdout_deferred.json')
            print('HOLDOUT_DEFERRED; completed training preserved; once-only gate remains unopened',flush=True)
        else:
            holdout_once(plan,out,args.init)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True); p.add_argument('--init',required=True)
    p.add_argument('--mode',choices=['paired-first','all'],default='paired-first')
    p.add_argument('--config');p.add_argument('--input-root',default='/kaggle/input')
    p.add_argument('--corpus-paths');p.add_argument('--train-csv');p.add_argument('--teacher-csv',action='append')
    p.add_argument('--relations-csv');p.add_argument('--excluded-csv');p.add_argument('--init-sha256')
    p.add_argument('--full-volume-sha256',action='store_true');p.add_argument('--holdout-once',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--plan');p.add_argument('--arm',choices=['uniform','spatial'])
    p.add_argument('--fold',type=int);p.add_argument('--deadline',type=float)
    a=p.parse_args()
    if a.worker:
        if a.plan is None or a.arm is None or a.fold is None or a.deadline is None:
            p.error('Worker requires --plan, --arm, --fold, --deadline')
        try:
            train_arm(a)
        except Exception as exc:
            atomic_json({'status':'failed','arm':a.arm,'fold':a.fold,'error':repr(exc),
                         'partial_models_deployable':False,'resume_checkpoint_may_exist':True},
                        Path(a.out)/f'{a.arm}_fold{a.fold}_failure.json')
            raise
    else:
        launch(a)


if __name__=='__main__':
    main()
