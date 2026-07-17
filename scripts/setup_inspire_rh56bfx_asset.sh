#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
simtoolreal_root="${SIMTOOLREAL_ROOT:?Set SIMTOOLREAL_ROOT to the simtoolreal repository path}"
source_dir="${simtoolreal_root}/assets/embodiments/franka-inspire-z180"
target_dir="${repo_root}/assets/embodiments/franka-inspire-rh56bfx-mimic"

for name in franka_description meshes; do
    source_path="${source_dir}/${name}"
    target_path="${target_dir}/${name}"
    if [[ ! -e "${source_path}" ]]; then
        echo "Missing Inspire asset dependency: ${source_path}" >&2
        exit 1
    fi
    if [[ -e "${target_path}" && ! -L "${target_path}" ]]; then
        echo "Refusing to replace non-symlink path: ${target_path}" >&2
        exit 1
    fi
    ln -sfn "${source_path}" "${target_path}"
done

echo "RH56BFX mesh links are ready under ${target_dir}"
