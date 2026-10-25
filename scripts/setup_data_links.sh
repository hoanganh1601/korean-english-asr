#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DATASET_ROOT="${ASR_DATASET_ROOT:-/media/icnlab/Data/hoanganh/datasets}"

ZEROTH_ROOT="${DATASET_ROOT}/zeroth_korean"
LIBRISPEECH_ROOT="${DATASET_ROOT}/librispeech/LibriSpeech"
HIKE_ROOT="${DATASET_ROOT}/hike/HiKE-619ebb54-extracted"
LIBRISPEECH_PROCESSED_ROOT="${DATASET_ROOT}/librispeech_processed_v1/dc_corrected"

require_directory() {
    local target="$1"

    if [[ ! -d "${target}" ]]; then
        echo "Missing dataset directory: ${target}" >&2
        exit 1
    fi
}

safe_link() {
    local target="$1"
    local link_path="$2"
    local canonical_target

    canonical_target="$(cd "${target}" && pwd -P)"

    if [[ -L "${link_path}" ]]; then
        if [[ "$(readlink -f "${link_path}")" == "${canonical_target}" ]]; then
            echo "Link already correct: ${link_path}"
            return
        fi

        echo "Existing symlink points elsewhere: ${link_path}" >&2
        exit 1
    fi

    if [[ -d "${link_path}" ]]; then
        echo "Using existing data directory: ${link_path}"
        return
    fi

    if [[ -e "${link_path}" ]]; then
        echo "Refusing to replace existing non-directory path: ${link_path}" >&2
        exit 1
    fi

    ln -s "${canonical_target}" "${link_path}"
    echo "Created link: ${link_path} -> ${canonical_target}"
}

assert_count() {
    local root="$1"
    local pattern="$2"
    local expected="$3"
    local actual

    actual="$(find -L "${root}" -type f -name "${pattern}" | wc -l)"

    if [[ "${actual}" -ne "${expected}" ]]; then
        echo "Unexpected file count at ${root}: ${actual}, expected ${expected}" >&2
        exit 1
    fi

    echo "Verified ${root}: ${actual} files"
}

require_directory "${ZEROTH_ROOT}"
require_directory "${LIBRISPEECH_ROOT}"
require_directory "${HIKE_ROOT}"
require_directory "${LIBRISPEECH_PROCESSED_ROOT}"

mkdir -p \
    "${PROJECT_ROOT}/data/raw" \
    "${PROJECT_ROOT}/data/processed/librispeech"

safe_link "${ZEROTH_ROOT}" "${PROJECT_ROOT}/data/raw/zeroth"
safe_link "${LIBRISPEECH_ROOT}" "${PROJECT_ROOT}/data/raw/librispeech"
safe_link "${HIKE_ROOT}" "${PROJECT_ROOT}/data/raw/hike"
safe_link \
    "${LIBRISPEECH_PROCESSED_ROOT}" \
    "${PROJECT_ROOT}/data/processed/librispeech/dc_corrected"

assert_count "${PROJECT_ROOT}/data/raw/zeroth" "*.flac" 22720
assert_count "${PROJECT_ROOT}/data/raw/librispeech" "*.flac" 33862
assert_count "${PROJECT_ROOT}/data/raw/hike/audio" "*.wav" 1121
assert_count \
    "${PROJECT_ROOT}/data/processed/librispeech/dc_corrected" \
    "*.flac" \
    500

cd "${PROJECT_ROOT}"
sha256sum --quiet -c artifacts/manifests/manifest_v1.sha256

(
    cd "${LIBRISPEECH_PROCESSED_ROOT}"
    sha256sum --quiet -c \
        "${PROJECT_ROOT}/artifacts/manifests/librispeech_dc_corrected_v1.sha256"
)

echo "Data links and handoff checksums: OK"
