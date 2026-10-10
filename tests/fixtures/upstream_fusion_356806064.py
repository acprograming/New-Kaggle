# Public DINOsaur immutable version 356806064: healthy fusion reference.
# Attribution retained in THIRD_PARTY.md.
# DINOsaur V16: preregistered Lateral-OA specialist ablation, exact V14 control.
if ACTIVE_MODE == 'apex950':
    from scipy.stats import rankdata
    from pathlib import Path
    import json

    BASE_W={"Baker's":.39,"Contusion":.33,"Medial OA":.25,"ACL":.27,
            "Medial Meniscus":.20,"MCL":.22,"Lateral Meniscus":.08,
            "Fracture":.08,"Lateral OA":.18,"PF OA":.08,
            "Effusion":.04,"Synovitis":.03}
    assert set(BASE_W)==set(TARGETS)
    # This is the sole experimental change. All other 11 columns remain exact.
    EXPERIMENT_W=BASE_W.copy()
    EXPERIMENT_W['Lateral OA']=.52

    def pct_rank_matrix(table, columns):
        x=table[columns].to_numpy(dtype=np.float64)
        assert x.ndim==2 and x.shape[0]>0 and np.isfinite(x).all()
        return (rankdata(x,axis=0,method='average')-.5)/len(x)

    assert 'reader' in globals() and reader is not None, 'Refusing RAPTOR-only 0.949 fallback'
    assert reader['StudyInstanceUID'].astype(str).tolist()==ids
    assert anchor['StudyInstanceUID'].astype(str).tolist()==ids
    a=pct_rank_matrix(anchor,TARGETS)
    b=pct_rank_matrix(reader,TARGETS)

    def make_fusion(wdict):
        w=np.array([wdict[k] for k in TARGETS],dtype=np.float64)[None,:]
        assert ((w>=0)&(w<1)).all()
        mixed=(1-w)*a + w*b
        mixed=(rankdata(mixed,axis=0,method='average')-.5)/len(ids)
        ret=anchor[['StudyInstanceUID']].copy()
        ret[TARGETS]=mixed
        assert np.isfinite(ret[TARGETS].to_numpy(np.float64)).all()
        return ret

    control=make_fusion(BASE_W)
    candidate=make_fusion(EXPERIMENT_W)
    control_path=WORK/'_dinov16_exact_v14_0950_control.csv'
    candidate_path=WORK/'_dinov16_lateral_oa_candidate.csv'
    control.to_csv(control_path,index=False)
    candidate.to_csv(candidate_path,index=False)

    stable_cols=[k for k in TARGETS if k!='Lateral OA']
    assert np.array_equal(control[stable_cols].to_numpy(np.float64),candidate[stable_cols].to_numpy(np.float64)), \
        'Experiment unexpectedly changed a non-Lateral-OA target'
    assert candidate['StudyInstanceUID'].tolist()==ids

    # Report sensitivity without selecting on the hidden-test ranks.
    # These alternative weights are diagnostics only; not alternative submissions.
    sensitivity={}
    for w in (.18,.35,.52,.70):
        cfg=BASE_W.copy();cfg['Lateral OA']=w
        z=make_fusion(cfg)
        delta=(z['Lateral OA']-control['Lateral OA']).to_numpy(float)
        sensitivity[str(w)]={'mean_abs_rank_shift':round(float(np.mean(np.abs(delta))),7),
                             'changed_rank_count':int(np.count_nonzero(delta))}

    result=candidate
    SUB=WORK/'submission.csv'
    result.to_csv(SUB,index=False)
    assert list(result.columns)==['StudyInstanceUID']+TARGETS
    assert result.StudyInstanceUID.is_unique and len(result)==len(ids)
    assert np.isfinite(result[TARGETS].to_numpy(np.float64)).all()
    assert ((result[TARGETS].to_numpy(np.float64)>=0)&(result[TARGETS].to_numpy(np.float64)<=1)).all()
    assert list(pd.read_csv(SUB,nrows=0).columns)==['StudyInstanceUID']+TARGETS
    receipt={'v14_control':'exact model and 12-target blend',
             'modified_target':'Lateral OA',
             'v14_lateral_oa_weight':.18,
             'candidate_lateral_oa_weight':.52,
             'other_targets_byte_equivalent':True,
             'reader_models':3,
             'source':'public checkpoints only',
             'n_studies':len(ids),
             'lateral_oa_rank_sensitivity':sensitivity,
             'elapsed_min':round((time.monotonic()-T0)/60,2),
             'actual_leaderboard_score':None}
    (WORK/'_dinov16_research_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('V16 completed:',len(ids),'studies | all reader folds complete | other 11 targets EXACT V14',flush=True)
    print('Lateral OA specialist sensitivity:',sensitivity,flush=True)
    print('submission.csv ready; experimental score NOT YET verified',flush=True)
