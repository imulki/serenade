#!/usr/bin/env bash

# Copyright 2024 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# basic settings
stage=0        # stage to start
stop_stage=99  # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=4       # number of gpus in training
n_jobs=2       # do NOT change unless you understand the code, number of parallel jobs in feature extraction

conf=conf/serenade.yaml
cyclic_conf=conf/serenade_cyclic.yaml
f0_path=conf/f0.yaml
ref_dict=conf/refstyles.json

# dataset configuration
db_root=downloads/GTSinger   # path to the GTSinger dataset
dumpdir=dump                # directory to dump full features
train_set=train-gtsinger
dev_set=dev-gtsinger
test_set=test-gtsinger
skip_extract_train=False
# training related setting
tag="baseline"     # tag for directory to save model

# pretrained model related
pretrain=""           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)
cyclic_pretrain=""   # (e.g. <path>/<to>/checkpoint-10000steps.pkl)
resume=""           # path to the checkpoint to resume from

# decoding related setting
checkpoint=""               # checkpoint path to be used for decoding
                            # if not provided, the latest one will be used
                            # (e.g. <path>/<to>/checkpoint-400000steps.pkl)
                                       
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    ./local/download_gtsinger.sh "${db_root}"
    python3 local/create_wav_scp.py \
        --input_dir "${db_root}" \
        --output_file "data/full/wav.scp"

    python3 local/create_gtsinger_splits.py \
        --wav-scp "data/full/wav.scp" \
        --train-set "data/${train_set}/wav.scp" \
        --dev-set "data/${dev_set}/wav.scp" \
        --test-set "data/${test_set}/wav.scp"
    echo "Created ${train_set}, ${dev_set}, and ${test_set} sets."
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Feature extraction"

    # extract raw features
    pids=()
    for name in "${train_set}" "${dev_set}" "${test_set}"; do
        if [ "${name}" = "${train_set}" ] && [ "${skip_extract_train}" = "True" ]; then
            continue
        fi
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            serenade-preprocess \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --midi-path "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --f0-path "${f0_path}" \
                --verbose "${verbose}"
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction."
fi


if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Statistics computation"
    # calculate statistics for normalization
    echo "Statistics computation start. See the progress via ${dumpdir}/${train_set}/compute_statistics.log."
    ${train_cmd} "${dumpdir}/${train_set}/compute_statistics.log" \
        serenade-compute_stats \
            --config "${conf}" \
            --rootdir "${dumpdir}/${train_set}/raw" \
            --dumpdir "${dumpdir}/${train_set}" \
            --verbose "${verbose}"
fi

if [ -z ${tag} ]; then
    expname=${train_set}_$(basename ${conf%.*})
else
    expname=${train_set}_${tag}
fi
expdir=exp/${expname}
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        train="torchrun --nnodes=1 --nproc_per_node=${n_gpus} serenade-train"
    else
        train="serenade-train"
    fi

    cp "${dumpdir}/${train_set}/stats.joblib" "${expdir}/stats.joblib"
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --train-dumpdir "${dumpdir}/${train_set}/raw" \
            --dev-dumpdir "${dumpdir}/${dev_set}/raw" \
            --stats "${expdir}/stats.joblib" \
            --outdir "${expdir}" \
            --init-checkpoint "${pretrain}" \
            --resume "${resume}" \
            --verbose "${verbose}"

    echo "Successfully finished training."
fi


if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    pids=()
    for name in "${dev_set}" "${test_set}"; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.*.log."
        ${cuda_cmd} JOB=1:${n_jobs} --gpu 1 "${outdir}/${name}/decode.JOB.log" \
            serenade-decode \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --checkpoint "${checkpoint}" \
                --stats "${expdir}/stats.joblib" \
                --ref-dict "${ref_dict}" \
                --outdir "${outdir}/${name}/out.JOB" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Successfully finished decoding."
fi


if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    echo "Stage 5: Network decoding (train set for cyclic fine-tuning)"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    pids=()
    for name in "${train_set}"; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.*.log."
        ${cuda_cmd} JOB=1:${n_jobs} --gpu 1 "${outdir}/${name}/decode.JOB.log" \
            serenade-decode \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --checkpoint "${checkpoint}" \
                --stats "${expdir}/stats.joblib" \
                --outdir "${outdir}/${name}/out.JOB" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Successfully finished decoding."
fi

if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 6 ]; then
    echo "Stage 6: Feature extraction (cyclic fine-tuning)"
    # create new wav.scp for cyclic fine-tuning
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    python3 local/create_wav_scp.py \
        --input_dir "${outdir}/${train_set}" \
        --output_file "data/${train_set}_cyclic/wav.scp"

    # extract features from converted samples
    pids=()
    for name in "${train_set}_cyclic"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            serenade-preprocess \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --midi-path "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --skip-gtmidi True \
                --f0-path "${f0_path}" \
                --verbose "${verbose}"
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction."

    # add original source logmel as targets
    python3 local/create_cyclic_dump.py \
        --outdir "${dumpdir}/${train_set}_cyclic/raw" \
        --dumpdir "${dumpdir}" \
        --train_set "${train_set}"
fi

if [ -z ${cyclic_pretrain} ]; then
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    cyclic_pretrain=${checkpoint}
fi
expdir=${expdir}_cyclic

if [ "${stage}" -le 7 ] && [ "${stop_stage}" -ge 7 ]; then
    echo "Stage 7: Network training (Cyclic Training)"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        train="torchrun --nnodes=1 --nproc_per_node=${n_gpus} serenade-train"
    else
        train="serenade-train"
    fi

    cp "${dumpdir}/${train_set}/stats.joblib" "${expdir}/stats.joblib"
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${cyclic_conf}" \
            --train-dumpdir "${dumpdir}/${train_set}_cyclic/raw" \
            --dev-dumpdir "${dumpdir}/${dev_set}/raw" \
            --stats "${expdir}/stats.joblib" \
            --outdir "${expdir}" \
            --init-checkpoint "${cyclic_pretrain}" \
            --resume "${resume}" \
            --verbose "${verbose}"

    echo "Successfully finished training."
fi

if [ "${stage}" -le 8 ] && [ "${stop_stage}" -ge 8 ]; then
    echo "Stage 8: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    if [ -z "${checkpoint}" ]; then
        outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    else
        outdir="$(dirname "${checkpoint}")/results/$(basename "${checkpoint}" .pkl)"
    fi
    pids=()
    for name in "${dev_set}" "${test_set}"; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.*.log."
        ${cuda_cmd} JOB=1:${n_jobs} --gpu 1 "${outdir}/${name}/decode.JOB.log" \
            serenade-decode \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --checkpoint "${checkpoint}" \
                --stats "${expdir}/stats.joblib" \
                --ref-dict "${ref_dict}" \
                --outdir "${outdir}/${name}/out.JOB" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Successfully finished decoding."
fi


if [ "${stage}" -le 9 ] && [ "${stop_stage}" -ge 9 ]; then
    echo "Stage 9: SiFiGAN post-processing"
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    if [ -z "${checkpoint}" ]; then
        outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    else
        outdir="$(dirname "${checkpoint}")/results/$(basename "${checkpoint}" .pkl)"
    fi
    serenade-postprocessing \
        generator=sifigan \
        in_dir="${outdir}/${test_set}" \
        stats="pt_models/postprocessing_sifigan/stats.joblib" \
        checkpoint_path="pt_models/postprocessing_sifigan/model.pkl"
fi