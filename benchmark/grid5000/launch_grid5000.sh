#!/usr/bin/env bash
# =============================================================================
# Grid5000 multi-node launch script for the MergeablePCA network-transfer
# benchmark.
#
# PURPOSE
#   Measure the inter-bridge transfer that the paper reports. Every measurement so
#   far is same-node, so it yields byte VOLUMES and compute costs but cannot
#   support any claim about real network time. This script runs the identical
#   workflow across two nodes so the transfer actually crosses a network.
#
# WHAT IT RUNS
#   One MPI rank per Dask worker, bridges as MPI ranks, exactly as the paper's
#   system model describes. Site is chosen by the user; nothing is hardcoded.
#
# VERIFIED Grid5000 CONSTRAINTS (checked 2026-10-05 against the Grid5000 wiki)
#   * Inter-site backbone is 10 Gb/s shared Ethernet (RENATER). Cross-site MPI
#     therefore runs over TCP. There is NO documented RoCE service.
#   * High-performance interconnects (100 Gb/s) are INTRA-site only. To measure
#     Ethernet at line rate you must stay inside ONE site.
#   * There is NO `network=dedicated` OAR resource. Whole-node reservation plus
#     switch pinning is the substitute.
#   * `/home` is per-site NFS. Benchmark payloads must NOT be staged there.
#   * Single-stream iperf3 is documented as CPU-bound near 1.2 Gbit/s. Use
#     multi-stream iperf3 or OSU, never single-stream, to calibrate bandwidth.
#   * MTU 9000 is supported; the default is 1500.
#
# SUGGESTED SITES (user selects one; see SITE TABLE below)
#   spirou      Louvain      8 nodes    2x100 Gb/s Ethernet
#   fleckenstein Strasbourg   10 nodes   25 Gb/s + 100 Gb/s SR-IOV Ethernet
#   gros        Nancy        123 nodes  2x25 Gb/s Ethernet
#   paradoxe    Rennes       64 nodes   25 Gb/s Ethernet
#   AVOID dahu  Grenoble     100 Gb/s fabric is Omni-Path, not Ethernet.
#
# USAGE
#   SITE=spirou NODES=2 WORKERS=2 ./launch_grid5000.sh
#   DRY_RUN=1 SITE=spirou ./launch_grid5000.sh     # print, reserve nothing
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------- configuration
SITE="${SITE:?set SITE, e.g. SITE=spirou}"
NODES="${NODES:-2}"
WORKERS="${WORKERS:-2}"              # Dask workers == MPI ranks per node
WALLTIME="${WALLTIME:-02:00:00}"
JOB_NAME="${JOB_NAME:-mergeable-pca-net}"
PROJECT="${PROJECT:-REQUIRED}"           # OAR project, required by the site
QOS="${QOS:-default}"
PARTITION="${PARTITION:-}"
DRY_RUN="${DRY_RUN:-0}"

REPO="${REPO:-${HOME}/deisa-dask}"
# Payloads go to LOCAL scratch. /home is NFS and would measure the filesystem.
SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}/mergeable-pca-$$}"
RESULT_DIR="${RESULT_DIR:-${HOME}/mergeable-pca-results}"

OAR_RESOURCES="${OAR_RESOURCES:-}"
if [ -z "${OAR_RESOURCES}" ]; then
  # Pin to Ethernet fabric and keep whole nodes. `kavlan-topo` constrains the
  # network topology; `network=dedicated` does NOT exist on Grid5000.
  OAR_RESOURCES="nodes=${NODES},kavlan-topo:pack=1"
  [ -n "${PARTITION}" ] && OAR_RESOURCES="${OAR_RESOURCES},partition=${PARTITION}"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------------------------------- preflight
preflight() {
  echo "== preflight =="
  local missing=0
  for cmd in oar; do
    command -v "${cmd}" >/dev/null 2>&1 || { echo "MISSING: ${cmd}"; missing=1; }
  done
  if [ "${missing}" -ne 0 ]; then
    echo "Grid5000 tools absent. Are you on a frontend/login node?"
    return 1
  fi
  echo "site:        ${SITE}"
  echo "nodes:       ${NODES}"
  echo "workers:     ${WORKERS}"
  echo "resources:   ${OAR_RESOURCES}"
  echo "scratch:     ${SCRATCH}  [local, NOT /home]"
  echo "result dir:  ${RESULT_DIR}"
  return 0
}

# --------------------------------------------------------- network calibration
# Calibrate BEFORE the application run. Single-stream iperf3 is CPU-bound on
# Grid5000, so a single stream reports the CPU, not the link. 8 parallel streams
# is the documented workaround.
calibrate() {
  local host="$1"
  local port_base="${2:-5201}"
  echo "== network calibration [multi-stream, 8 streams] =="

  # Server on the far node, background.
  ssh -o StrictHostKeyChecking=no "${host}" \
    "iperf3 -s -p ${port_base} -D --logfile /tmp/iperf3-server.log" 2>/dev/null || {
      echo "iperf3 server failed on ${host}"; return 1; }

  # Client here. -P 8 avoids the documented CPU-bound single-stream result.
  # The python reader is written without f-strings so no shell quoting conflicts.
  for i in $(seq 1 3); do
    iperf3 -c "${host}" -p "${port_base}" -P 8 -t 10 \
      --json 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
end = d["end"]
sent = end["sum_sent"]
print("  trial: %.2f Gbit/s sent, retransmits=%d"
      % (sent["bits_per_second"] / 1e9, sent["retransmits"]))
' || echo "  iperf3 trial ${i} failed"
  done

  ssh -o StrictHostKeyChecking=no "${host}" "pkill -f 'iperf3 -s' || true" 2>/dev/null || true
}

# --------------------------------------------------------------- the benchmark
# Runs the two workflows the paper compares:
#   A) bridge-side PCA + merge, then ship the summary   (our contribution)
#   B) ship the full chunk, then PCA                    (the control)
# Both write provenance-bearing JSON so every paper number traces to a file.
run_benchmark() {
  echo "== benchmark =="
  mkdir -p "${SCRATCH}" "${RESULT_DIR}"

  export DEISA_DASK_SCHEDULER_ADDRESS="${DEISA_DASK_SCHEDULER_ADDRESS:-tcp://127.0.0.1:8787}"
  export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
  export TMPDIR="${SCRATCH}"
  export DEISA_GRID5000_SITE="${SITE}"

  # The benchmark scripts take the grid from argv and write JSON with provenance.
  local b1="benchmark/mergeable_pca/b1_network_transfer.py"
  local b6="benchmark/mergeable_pca/b6_tradeoff_cost.py"

  echo "-- workflow A/B volume measurement [b1] --"
  mpirun -np "${WORKERS}" -host "${hostfile}" \
    .venv/bin/python "${REPO}/${b1}" \
      --output "${RESULT_DIR}/b1_network_transfer_grid5000.json" \
      2>&1 | tee "${RESULT_DIR}/b1_network_transfer_grid5000.log"

  if [ -f "${REPO}/${b6}" ]; then
    echo "-- bridge-side PCA + merge vs full transfer + PCA [b6] --"
    mpirun -np "${WORKERS}" -host "${hostfile}" \
      .venv/bin/python "${REPO}/${b6}" \
        --output "${RESULT_DIR}/b6_tradeoff_cost_grid5000.json" \
        2>&1 | tee "${RESULT_DIR}/b6_tradeoff_cost_grid5000.log"
  fi

  cp "${RESULT_DIR}"/*grid5000.json "${RESULT_DIR}/" 2>/dev/null || true
  echo "results in ${RESULT_DIR}"
}

# ------------------------------------------------------------------ main flow
main() {
  preflight

  if [ "${DRY_RUN}" = "1" ]; then
    self="$(basename "${BASH_SOURCE[0]}")"
    {
      echo
      echo "--- DRY RUN, nothing reserved ---"
      echo
      echo "oarsub -S -O -l /bin/bash \\"
      echo "      -n \"${JOB_NAME}\" \\"
      echo "      -l walltime=\"${WALLTIME}\" \\"
      echo "      -l qos=\"${QOS}\" \\"
      echo "      -p \"${PROJECT}\" \\"
      echo "      -t deploy \\"
      echo "      -r \"${OAR_RESOURCES}\" \\"
      echo "      ${REPO}/${self}"
      echo
    }
    exit 0
  fi

  # Inside the OAR job: discover our own allocation.
  if [ -f "${OAR_JOB_FILE:-/etc/oar/job_file}" ]; then
    echo "running inside OAR allocation"
    # OAR_JOB_FILE lists assigned hosts, one per line, as "hostname slots=N"
    mapfile -t hosts < <(awk '{print $1}' "${OAR_JOB_FILE:-/etc/oar/job_file}")
    hostfile="$(mktemp)"
    for h in "${hosts[@]:0:${NODES}}"; do
      echo "${h} slots=${WORKERS}" >> "${hostfile}"
    done
    trap 'rm -f "${hostfile}"' EXIT
  else
    echo "not inside an OAR allocation; using SITE-local hostfile"
    hostfile="${SCRATCH}/hosts"
    mkdir -p "${SCRATCH}"
    # Fill this in from 'oarnodes' on the frontend.
    local domain
    domain="$(hostname -d 2>/dev/null || true)"
    {
      echo "${SITE}${domain:+.${domain}} slots=${WORKERS}"
    } > "${hostfile}"
    echo "WARNING: ${hostfile} is a placeholder. Replace it with real hosts."
  fi

  echo "hostfile ${hostfile}:"
  cat "${hostfile}"

  # Enable jumbo frames where permitted. Default MTU is 1500; 9000 is supported.
  if [ "${SET_MTU:-0}" = "1" ]; then
    echo "== setting MTU 9000 =="
    sudo ip link set dev eth0 mtu 9000 2>/dev/null \
      || echo "could not set MTU; continuing at default [measure and report it]"
  fi

  calibrate "$(head -1 "${hostfile}" | awk '{print $1}')" || \
    echo "calibration FAILED: do NOT report a network speed from this run"

  run_benchmark

  echo "== done =="
}

main "$@"