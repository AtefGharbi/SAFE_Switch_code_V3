#!/bin/bash
# V3: regenerate every learned component and every result with the audited code.
cd "$(dirname "$0")"
L=results
run() { echo "[$(date +%H:%M:%S)] $*" >> $L/v3_progress.txt; "$@" >> $L/log_v3_$(echo "$*" | tr ' /=.' '____' | cut -c1-40).txt 2>&1; echo "[$(date +%H:%M:%S)] done ($?) $*" >> $L/v3_progress.txt; }
run python3 verify_certificate.py dcmotor
run python3 verify_certificate.py tank
run python3 audit_certificate.py
run python3 post_alarm.py
run python3 train.py risk
run python3 -c "import json; from bench_designs import make_design; from train import train_risk; r=train_risk(make_design('tank')); json.dump(r, open('results/risk_model_tank.json','w'), indent=1)"
run python3 tune_threshold.py
run python3 -c "from config import Design; from train import bc_data, load_risk; bc_data(Design(), load_risk())"
run python3 -c "from bench_designs import make_design; from train import bc_data, load_risk; bc_data(make_design('tank'), load_risk('results/risk_model_tank.json'))"
( run python3 train.py ppo full 0,1,2,3 30; run python3 train2.py gru 0,1,2; run python3 train2.py cvar 0,1; run python3 train.py ppo norisk 0,1 30 ) &
( run python3 train.py ppo full 4,5,6,7 30; run python3 train2.py gru 3,4,5; run python3 train2.py cvar 2; run python3 train.py ppo nofilter 0,1 30; run python3 tank_train.py 0,1,2 ) &
wait
run python3 repro_hd.py
run python3 experiments.py main 20 2 withnf
run python3 experiments.py ablation 20 2
run python3 experiments.py stress 20 2
run python3 experiments_v2.py e12 20 2
run python3 experiments_v2.py e13 15 2
run python3 experiments_v2.py e15 20 2
run python3 experiments_v2.py tank 20 2
run python3 chain_decentral.py
run python3 scal.py
run python3 curve_eval.py
run python3 adv_search.py dcmotor 40
run env ADV_START=0.0 python3 adv_search.py dcmotor 25
run python3 adv_search.py tank 30
run env ADV_START=0.0 python3 adv_search.py tank 20
run env ADV_LARGE=2.0 python3 adv_search.py dcmotor 25
run env ADV_LARGE=2.0 python3 adv_search.py tank 20
run python3 e5_infeas_check.py
run python3 analysis.py
run python3 analysis_v2.py
run python3 figures.py arch repro risk
run python3 figs_results.py traces mech sens fstress learn tank
echo done > $L/v3_pipeline.done
