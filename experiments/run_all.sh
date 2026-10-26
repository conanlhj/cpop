#!/bin/bash
# Runner for the revised evaluation protocol. Every step runs in its own Docker container.
#   ./run_all.sh prep | smoke | main | extra | post | all
#   prep  = ImageNet val cache, class-stratified splits, GroupNorm check   (needs IMAGENET)
#   all   = smoke, then main, extra and post only if no smoke step failed
# A step whose ok_* marker exists is skipped (scripts also resume from their parts).
# Settings (environment):
#   DATA      host directory mounted as /ext; outputs go to $DATA/r1, logs to $DATA/r1_logs
#   IMAGENET  host ILSVRC2012 root (val_full/*.JPEG + ILSVRC2012_devkit_t12/), used by prep only
#   IMAGE     Docker image with requirements.txt installed (default cpop:latest)
#   ADM_PATH  ADM checkpoint inside the container (default: Hugging Face id, see common.py)
STAGE=${1:-main}
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:?set DATA to the data/output directory}; IMAGE=${IMAGE:-cpop:latest}
if [ "$STAGE" = all ]; then "$0" smoke && "$0" main && "$0" extra && "$0" post; exit $?; fi
L=$DATA/r1_logs; mkdir -p $L; echo $$ > $L/run.pid
SM=0; [ "$STAGE" = smoke ] && SM=1
P=$( [ $SM = 1 ] && echo smoke_ )
CLFS="resnet50 vgg16 densenet121 mobilenet_v3_large convnext_tiny vit_b_16 swin_t maxvit_t"
CLF4="resnet50 densenet121 vit_b_16 swin_t"

run() {   # run <tag> <python args...>
  local tag=$P$1; shift
  [ -f $L/ok_$tag ] && return 0
  echo "$(date +%F_%T) start $tag" >> $L/progress.txt
  docker run --rm --name cpop-r1 --gpus all --shm-size 8g -e SMOKE=$SM -e SMOKE_N=6 \
    -e TORCH_HOME=/ext/torch_cache -e HF_HOME=/ext/hf_cache -e OMP_NUM_THREADS=4 ${ADM_PATH:+-e ADM_PATH=$ADM_PATH} \
    ${IMAGENET:+-v $IMAGENET:/im:ro} -v $REPO:/repo:ro -v $DATA:/ext -w /repo/experiments \
    $IMAGE python "$@" > $L/$tag.log 2>&1
  local rc=$?
  echo "$(date +%F_%T) end   $tag rc=$rc" >> $L/progress.txt
  if [ $rc -eq 0 ]; then touch $L/ok_$tag; else touch $L/FAILED_$tag; fi
  return $rc
}

if [ "$STAGE" = prep ]; then           # data cache, splits, checkpoint check
  run prep prep_cache.py
  run splits make_splits.py
  run g0 g0_groupnorm_check.py
  exit 0
fi

if [ "$STAGE" = smoke ]; then          # GPU smoke of everything not yet smoke-tested on CPU
  run s0 s0_bench.py
  run s1_vit s1_clean_clf.py --clf vit_b_16
  run s2_vit_pgd s2_attack.py --clf vit_b_16 --kind PGD --eps 4
  for k in APGD AutoAttack CW Square TIFGSM; do run s2_r50_$k s2_attack.py --clf resnet50 --kind $k --eps 4 --n 2; done
  run s3 s3_score.py clean seeds sigma ALL_ADV
  run s6 s6_adaptive.py --clf resnet50 --obj hinge --lam 10 --steps 4 --eot 2 --n 2 --check 2 --bs 1
  run s6_sum s6_adaptive.py --clf resnet50 --obj sum --lam 20 --steps 4 --eot 2 --n 2 --check 2 --bs 1 --alpha_div 4
  run s7 s7_cifar.py --clf resnet20
  run s5_cifar s5_feature_baselines.py --dataset cifar --clf resnet20
  touch $L/${P}DONE; ls $L/FAILED_smoke_* >/dev/null 2>&1 && exit 1; exit 0
fi

if [ "$STAGE" = main ]; then           # clean scores, PGD at three budgets, benign controls, feature baselines
  run s0 s0_bench.py
  for c in $CLFS; do run s1_$c s1_clean_clf.py --clf $c; done
  run s5_resnet50_clean s5_feature_baselines.py --clf resnet50            # early check of the feature baselines
  run s3_clean s3_score.py clean seeds --bs 32
  for e in 4 2 8; do for c in $CLFS; do run s2_${c}_PGD_e$e s2_attack.py --clf $c --kind PGD --eps $e; done; done
  for c in resnet50 vit_b_16; do for k in sign gauss phase shuffle jpeg75 blur; do
    run s2_${c}_CTRL_${k} s2_attack.py --clf $c --kind CTRL_$k --eps 4; done; done
  run s3_adv s3_score.py sigma ALL_ADV --bs 32
  for c in $CLFS; do run s5_$c s5_feature_baselines.py --clf $c; done
  touch $L/DONE_main; exit 0
fi

if [ "$STAGE" = extra ]; then          # attack family, CIFAR-10, adaptive attacks
  for c in $CLF4; do
    for k in FGSM BIM RFGSM MIFGSM DIFGSM TIFGSM NIFGSM APGD; do run s2_${c}_$k s2_attack.py --clf $c --kind $k --eps 4; done
    run s2_${c}_CW s2_attack.py --clf $c --kind CW --eps 4 --n 500
    run s2_${c}_Square s2_attack.py --clf $c --kind Square --eps 4 --n 300
    run s2_${c}_AutoAttack s2_attack.py --clf $c --kind AutoAttack --eps 4 --n 300
  done
  run s3_adv_extra s3_score.py ALL_ADV --bs 32
  for c in $CLF4; do rm -f $L/ok_s5_$c; run s5_$c s5_feature_baselines.py --clf $c; done   # picks up the new attack files
  for c in resnet20 resnet56 vgg16_bn mobilenetv2 repvgg_a2 shufflenetv2; do
    run s7_$c s7_cifar.py --clf $c; run s5_cifar_$c s5_feature_baselines.py --dataset cifar --clf $c; done
  for c in resnet50 vit_b_16; do
    for lam in 0 20 100; do run s6_${c}_sum_$lam s6_adaptive.py --clf $c --obj sum --lam $lam --steps 12 --eot 2 --n 100 --alpha_div 4 --bs 1; done
    for lam in 10; do run s6_${c}_hinge_$lam s6_adaptive.py --clf $c --obj hinge --lam $lam --steps 100 --eot 4 --n 100 --bs 1; done   # fp32 ADM backward: bs 2 peaks at 21 GB
  done
  touch $L/DONE_extra; exit 0
fi

if [ "$STAGE" = post ]; then           # probe configuration, probe transfer, alignment, analysis tables
  run s3_tgrid s3_score.py tgrid --bs 32
  run s8 s8_probe_k.py probe k --bs 16
  run s9 s9_alignment.py --n 300
  run s10 s10_norm_dependence.py --n 50
  for a in a1_analysis a2_cifar a3_controls a4_probe; do rm -f $L/ok_$a; run $a $a.py; done
  touch $L/DONE_post; exit 0
fi
