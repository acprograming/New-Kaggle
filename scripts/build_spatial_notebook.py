"""Embed the repository training modules into a standalone, offline Kaggle notebook."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES=['src/rsna/spatial_mil.py','src/rsna/cache_data.py','scripts/train_spatial_ablation.py']
embedded={name:(ROOT/name).read_text() for name in FILES}
checks={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}

def code(text):
    compile(text,'<notebook-cell>','exec')
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
            'source':text.splitlines(keepends=True)}

markdown='''# RSNA original spatial attention: prospective training ablation

This notebook trains a new model. It does not copy a competition checkpoint or promise a leaderboard score.

The single hypothesis is **per-finding spatial attention**, compared with uniform spatial pooling. Both arms start from the same independently pretrained ConvNeXt backbone and shared MIL parameters. Zero query vectors reproduce the control output while remaining learnable. The ConvNeXt post-pool normalisation is retained.

A fresh deterministic split reserves 20% of report/known-relation groups as a locked holdout. Four development folds exclude all 58 expert-labelled studies and their known connected groups. Only development weak-label AUC selects checkpoints. The holdout is opened once, after every paired fold has completed; it measures agreement with weak labels, not expert accuracy. Known patient relations may be supplied to strengthen isolation.

**Attach these inputs before running, with T4 ×2 and Internet off:**

- Competition `rsna-knee-abnormality-detection` (`train.csv`).
- `dreaddevelopment/knee-raptor-corpus`: `all_vols.npy`, `all_masks.npy`, `all_ids.npy`.
- `dreaddevelopment/knee-raptor-corpus-ext`: `extra_vols.npy`, `extra_masks.npy`, `extra_ids.npy`.
- Dataset `rsna-knee-hybrid-report-labels`: `report_labels_v4hybrid.csv`.
- `stevenleehans/rsna-knee-llm-report-labels`: `llm_labels_v4_blend.csv`.
- Your existing output `tdeng24/rsna-knee-overnight-convnext-td`, containing the pure `imagenet_initialization.pt`. Required SHA256: `6270cae1e860a267e3b9a6f73932d627edaf3529a32bf9b80e79f04c936aa7b9`. No competition-trained weights are loaded.

The public caches contain 44 **sampled** planes at 336 pixels. A triplet never crosses a series slot, and all three masks must be positive. These are not claimed to be adjacent native DICOM slices. Source processing is the original Dread corpus44 contract; this experiment cannot serve a native384/96-plane checkpoint.

Default run: both arms of fold0, eight epochs each. If both complete, retain the entire output directory and extend the same locked plan with `MODE='all'`. `EVALUATE_HOLDOUT_ONCE=True` is valid only after all four pairs complete. Partial models have resume checkpoints and cannot count as a completed family. To resume across Kaggle sessions, attach the preceding output and set `RESUME_OUTPUT` to its `original_spatial` folder.

Only a checkpoint/receipt folder is generated; there is no competition submission in this training notebook. Do not publish reports, target tables, study IDs, or image arrays with the source repository.

Attribution: public corpus inputs by dreaddevelopment; public weak labels by their dataset authors. The per-finding spatial architecture, controlled comparison, split protocol, and training implementation in this notebook are original project code. All upstream input licences and competition rules still apply.
'''
setup='''import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'Enable the T4 GPU accelerator'
assert torch.cuda.device_count()==2, 'Use T4 ×2 for paired arms'
CODE_ROOT=Path('/kaggle/working/original_spatial_code')
OUTPUT=Path('/kaggle/working/original_spatial')
MODE='paired-first'  # After the first complete pair, extend the same plan with 'all'.
EVALUATE_HOLDOUT_ONCE=False
RESUME_OUTPUT=''  # Optional attached previous original_spatial output directory.
RELATIONS_CSV=''  # Optional UID,RelatedStudyInstanceUID local table.
EXCLUDED_CSV=''   # Optional StudyInstanceUID local exclusion table.
if RESUME_OUTPUT:
    old=Path(RESUME_OUTPUT)
    assert (old/'plan.json').is_file(), 'Resume folder must contain the locked plan'
    shutil.copytree(old,OUTPUT,dirs_exist_ok=True)
OUTPUT.mkdir(parents=True,exist_ok=True)
os.environ['HF_HUB_OFFLINE']='1'
os.environ['PIP_NO_INDEX']='1'
'''
write='EMBEDDED='+repr(embedded)+'\nEXPECTED_SHA256='+repr(checks)+'''
for name,text in EMBEDDED.items():
    path=CODE_ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==EXPECTED_SHA256[name],name
    compile(text,str(path),'exec')
(CODE_ROOT/'src/rsna/__init__.py').write_text('"""Original RSNA spatial ablation package."""\\n')
sys.path.insert(0,str(CODE_ROOT/'src'))
from rsna.cache_data import discover_inputs,find_dataset_file,sha256,DEFAULT_CONFIG
paths,train_csv,teachers=discover_inputs()
INIT=Path(find_dataset_file('rsna-knee-overnight-convnext-td','imagenet_initialization.pt'))
assert sha256(INIT)==DEFAULT_CONFIG['initialization_sha256'],'Wrong ImageNet initialization bytes'
print('OFFLINE_INPUTS_READY',{'corpus_files':paths,'train_csv':train_csv,'teachers':teachers,
                             'initialization':str(INIT),'source_sha256':EXPECTED_SHA256},flush=True)
'''
run='''env=dict(os.environ,PYTHONPATH=str(CODE_ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH',''),
         OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
command=[sys.executable,'-u',str(CODE_ROOT/'scripts/train_spatial_ablation.py'),
         '--out',str(OUTPUT),'--init',str(INIT),'--mode',MODE]
if RELATIONS_CSV:command+=['--relations-csv',RELATIONS_CSV]
if EXCLUDED_CSV:command+=['--excluded-csv',EXCLUDED_CSV]
if EVALUATE_HOLDOUT_ONCE:command+=['--holdout-once']
started=time.time()
rc=subprocess.run(command,env=env,timeout=int(DEFAULT_CONFIG['max_hours']*3600+300)).returncode
assert rc==0,'Training failed: inspect per-arm logs/status before resuming'
status=json.loads((OUTPUT/'experiment_status.json').read_text())
print('ACTUAL_TRAINING_STATUS',json.dumps(status),flush=True)
print('Notebook elapsed hours:',round((time.time()-started)/3600,3),flush=True)
'''
nb={'nbformat':4,'nbformat_minor':4,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.12'},'rsna_original_spatial':{'sources':checks,
        'actual_training_executed':False,'holdout_proxy':'weak labels','competition_submission':False}},
    'cells':[{'cell_type':'markdown','metadata':{},'source':markdown.splitlines(keepends=True)},code(setup),code(write),code(run)]}
(ROOT/'notebooks/RSNA_Spatial_Ablation_Train.ipynb').write_text(json.dumps(nb,indent=1))
print('Built self-contained notebook; source hashes',checks)
