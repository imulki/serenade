#!/usr/bin/env bash
set -e

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

declare -A data_urls=(
    ["mandarin"]="https://drive.google.com/open?id=1ZuY8Bj0djxK0gQHnX9ur5anUgDusZ8xp"
    ["english"]="https://drive.google.com/open?id=1DK1vnTs_d9JD7Xya4U5CEaXszIWyboFv"
    ["japanese"]="https://drive.google.com/open?id=1wu4ccoht3-Yb_UIGgDGhPG-6T8cwFd70"
    ["korean"]="https://drive.google.com/open?id=1hwROisMOpUEAI5d9RgHi_V5fuW59UB4u"
    ["russian"]="https://drive.google.com/open?id=12_bcpmL4NGWy6bZz-e8nrvV2_MbQCise"
    ["spanish"]="https://drive.google.com/open?id=1MYGtSwSMXjkQ_93ZeQZ5Z2VJ5NX7WNu2"
    ["french"]="https://drive.google.com/open?id=1c-wf0Sx_BYWlW7uDSvUZt6S9-2bdmd2A"
    ["german"]="https://drive.google.com/open?id=18G4ByGGD0vV8dunDZ1NrQKubqPLuD-Jq"
    ["italian"]="https://drive.google.com/open?id=1fxKD95WKbBRtLC0bTF8YcnfwHTRYJRuJ"
)

dir="${download_dir}"
if [ ! -d "${dir}" ]; then
    mkdir -p "${dir}"
fi

for lang_id in "${!data_urls[@]}"; do
    share_url="${data_urls[${lang_id}]}"
    echo "Downloading ${lang_id} data..."
    if [ ! -e "${dir}/${lang_id}/.complete" ]; then
        mkdir -p "${dir}/${lang_id}"
        utils/download_from_google_drive.sh "${share_url}" "${dir}/${lang_id}"
        touch "${dir}/${lang_id}/.complete"
    fi
done
echo "Successfully finished download of GTSinger data."