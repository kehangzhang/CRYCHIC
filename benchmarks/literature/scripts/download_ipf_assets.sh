#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}"
PROXY="${IPF_HTTP_PROXY:-}"
WITH_022I=0
WITH_PROCESSED_RDS=0
WITH_VM=0

usage() {
    printf '%s\n' \
        "Usage: $0 [--root DIR] [--proxy URL] [--with-022i]" \
        "          [--with-processed-rds] [--with-vm]" \
        "" \
        "The default download is lightweight: manuscript XML, supplement," \
        "commit-pinned gold standards/workflow code, Zenodo metadata, and" \
        "the small GSE136831 gene/metadata tables. Large data are opt-in." \
        "Existing final files are never overwritten. Partial downloads use" \
        "a sibling .part file and are renamed only after validation."
}

while (($#)); do
    case "$1" in
        --root)
            ROOT="$2"
            shift 2
            ;;
        --proxy)
            PROXY="$2"
            shift 2
            ;;
        --with-022i)
            WITH_022I=1
            shift
            ;;
        --with-processed-rds)
            WITH_PROCESSED_RDS=1
            shift
            ;;
        --with-vm)
            WITH_VM=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

mkdir -p "${ROOT}/sources/upstream" "${ROOT}/downloads" "${ROOT}/data"

CURL=(
    curl --fail --location --retry 4 --retry-all-errors
    --connect-timeout 30 --continue-at -
)
if [[ -n "${PROXY}" ]]; then
    CURL+=(--proxy "${PROXY}")
fi

digest_file() {
    local algorithm="$1"
    local path="$2"
    case "${algorithm}" in
        sha256) sha256sum "${path}" | awk '{print $1}' ;;
        md5) md5sum "${path}" | awk '{print $1}' ;;
        *) printf 'Unsupported digest algorithm: %s\n' "${algorithm}" >&2; return 2 ;;
    esac
}

verify_file() {
    local path="$1"
    local expected_size="${2:-}"
    local algorithm="${3:-}"
    local expected_digest="${4:-}"
    if [[ ! -f "${path}" ]]; then
        return 1
    fi
    if [[ -n "${expected_size}" ]]; then
        local actual_size
        actual_size="$(stat -c '%s' "${path}")"
        [[ "${actual_size}" == "${expected_size}" ]] || return 1
    fi
    if [[ -n "${algorithm}" ]]; then
        local actual_digest
        actual_digest="$(digest_file "${algorithm}" "${path}")"
        [[ "${actual_digest}" == "${expected_digest}" ]] || return 1
    fi
}

fetch() {
    local url="$1"
    local destination="$2"
    local expected_size="${3:-}"
    local algorithm="${4:-}"
    local expected_digest="${5:-}"
    mkdir -p "$(dirname "${destination}")"
    if [[ -e "${destination}" ]]; then
        if verify_file "${destination}" "${expected_size}" "${algorithm}" "${expected_digest}"; then
            printf 'verified existing %s\n' "${destination}"
            return 0
        fi
        printf 'Refusing to overwrite an existing but unverified file: %s\n' "${destination}" >&2
        return 1
    fi
    local partial="${destination}.part"
    printf 'downloading %s\n' "${url}"
    "${CURL[@]}" --output "${partial}" "${url}"
    if ! verify_file "${partial}" "${expected_size}" "${algorithm}" "${expected_digest}"; then
        printf 'Downloaded file failed validation: %s\n' "${partial}" >&2
        return 1
    fi
    mv "${partial}" "${destination}"
}

COMMIT="279718d539b47dc890c6f3f2e4c03f5a5df33c3e"
RAW="https://raw.githubusercontent.com/mora-lab/cell-cell-interactions/${COMMIT}/benchmark-workflow"

fetch \
    "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10452151/fullTextXML" \
    "${ROOT}/sources/upstream/crosstalk_tools_2023_PMC10452151.xml"
fetch \
    "https://mdpi-res.com/d_attachment/biomolecules/biomolecules-13-01211/article_deploy/biomolecules-13-01211-s001.zip" \
    "${ROOT}/downloads/biomolecules-13-01211-s001.zip" \
    228993 sha256 f512772199cc524729a50132664cfd7b57140825e14e78f263ff23cccb958665
fetch \
    "${RAW}/data/IPF%20gold%20standard.txt" \
    "${ROOT}/sources/upstream/IPF gold standard.txt" \
    14764 sha256 6f39601728009ea5dc0479d31d5878ad7100bad582961023c303793e654d82ed
fetch \
    "${RAW}/data/IPF%20gold%20standard%20%28with%20ontology%20and%20reference%29.txt" \
    "${ROOT}/sources/upstream/IPF gold standard (with ontology and reference).txt" \
    19308 sha256 a9cf4d040a3da45a77d385ca6b5926f1d4322ad3f356f825ca52f1b2da20e7f6
fetch \
    "${RAW}/Workflow.ipynb" \
    "${ROOT}/sources/upstream/Workflow.ipynb" \
    441660 sha256 5aaafe8e5cc816bd155ee2ed42d4a55df37ad36fc1f08fbd5cf84bbae8747946
fetch \
    "${RAW}/R/SSP.R" \
    "${ROOT}/sources/upstream/SSP.R" \
    758 sha256 857f535994a0ad8e27a0ed8cf6dcbfd6221c8270b08e18072780a9b68d4e10d0
fetch \
    "${RAW}/R/CellPhoneDB_SSP.R" \
    "${ROOT}/sources/upstream/CellPhoneDB_SSP.R" \
    885 sha256 244baf10e74eedcfd46881facbc6b01ad6b31f982136108631bf1f3a55d41a2d
fetch \
    "${RAW}/R/GSE136831%20processing.R" \
    "${ROOT}/sources/upstream/GSE136831 processing.R" \
    1657 sha256 338a1f70a186302c9a637cc049b19887079f27cda207822ce038ba1c9bc53456
fetch \
    "${RAW}/R/run_NATMI.R" \
    "${ROOT}/sources/upstream/run_NATMI.R" \
    866 sha256 4b5188ede9aa80806dac46dd02da45553b66c1c89871cd684cb5ffa7e3554295
fetch \
    "https://zenodo.org/api/records/6497091" \
    "${ROOT}/sources/upstream/zenodo_6497091.json"
fetch \
    "https://zenodo.org/api/records/8020387" \
    "${ROOT}/sources/upstream/zenodo_8020387.json"
fetch \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE136nnn/GSE136831/suppl/GSE136831_AllCells.GeneIDs.txt.gz" \
    "${ROOT}/data/GSE136831_AllCells.GeneIDs.txt.gz" \
    323848 sha256 e43b3008945c87a6875d4fb1a99a791fdaefeeb3c5a72a5d17589f621eb83af8
fetch \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE136nnn/GSE136831/suppl/GSE136831_AllCells.Samples.CellType.MetadataTable.txt.gz" \
    "${ROOT}/data/GSE136831_AllCells.Samples.CellType.MetadataTable.txt.gz" \
    4308752 sha256 147bdd11d720ed30736a459ba62a3d8e2f3f9028a8430e0eb714484f0456049b

if ((WITH_022I)); then
    fetch \
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4058nnn/GSM4058966/suppl/GSM4058966_022I.dgecounts.rds.gz" \
        "${ROOT}/data/GSM4058966_022I.dgecounts.rds.gz" \
        270273041 sha256 341494fde07c7c0f1dcfe15a95fca3d292815f6e8f3a6f3e0276b53a8fce1c95
    gzip -t "${ROOT}/data/GSM4058966_022I.dgecounts.rds.gz"
fi

if ((WITH_PROCESSED_RDS)); then
    fetch \
        "https://zenodo.org/api/records/6497091/files/GSE122960.rds/content" \
        "${ROOT}/data/zenodo_6497091/GSE122960.rds" \
        395992088 md5 daa858e41eac1a01f37df95e6cba3b0d
    fetch \
        "https://zenodo.org/api/records/6497091/files/GSE128033.rds/content" \
        "${ROOT}/data/zenodo_6497091/GSE128033.rds" \
        698288593 md5 edb5bce0b444766dab1287bcb029a1fa
    fetch \
        "https://zenodo.org/api/records/6497091/files/GSE135893.rds/content" \
        "${ROOT}/data/zenodo_6497091/GSE135893.rds" \
        283365154 md5 ad95bd1a89775dabb14cdb889fc4d269
    fetch \
        "https://zenodo.org/api/records/6497091/files/GSE136831.rds/content" \
        "${ROOT}/data/zenodo_6497091/GSE136831.rds" \
        1009357041 md5 3c47e613949a2ce955deb7d04afeabf0
fi

if ((WITH_VM)); then
    fetch \
        "https://zenodo.org/api/records/8020387/files/Benchmark%20study%20of%20CCI%20prediction%20tools.ova/content" \
        "${ROOT}/data/zenodo_8020387/Benchmark study of CCI prediction tools.ova" \
        22804624384 md5 c3a9cde67a82f842072f668a0aba8cd4
fi

printf 'Download stage complete under %s\n' "${ROOT}"
