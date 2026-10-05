#!/bin/bash
# Multi-user test of setup_shared_hf_cache.sh, run as root inside a throwaway container:
#   docker run --rm -e HF_TOKEN -v $PWD/setup_shared_hf_cache.sh:/setup.sh:ro -v $PWD/test_shared_hf_cache.sh:/t.sh:ro --entrypoint bash gr00t-thor /t.sh
set -e
apt-get update -qq >/dev/null && apt-get install -y -qq acl sudo >/dev/null
PY=/opt/gr00t-venv/bin/python
for u in alice bob lkk; do id $u >/dev/null 2>&1 || useradd -m -s /bin/bash $u; done
# lkk has a private cache to migrate
mkdir -p /home/lkk/.cache/huggingface && chown -R lkk:lkk /home/lkk/.cache
su lkk -c "umask 022; HF_HUB_CACHE=/home/lkk/.cache/huggingface/hub $PY -c \"from huggingface_hub import hf_hub_download as d; d('hf-internal-testing/tiny-random-bert','config.json')\""
bash /setup.sh --add-user alice bob --migrate-from lkk | tail -4
echo "== /etc/environment"; grep -E "HF_|OPENPI" /etc/environment
run() { u=$1; shift; su $u -c "umask 022; set -a; . /etc/environment; set +a; $*"; }
echo "== alice downloads a new file into the repo lkk migrated"
run alice "$PY -c \"from huggingface_hub import hf_hub_download as d; print(d('hf-internal-testing/tiny-random-bert','tokenizer_config.json'))\""
echo "== bob downloads the weights of the same repo (needs to write blobs/refs/locks alice+lkk created)"
run bob "$PY -c \"from huggingface_hub import snapshot_download as s; print(s('hf-internal-testing/tiny-random-bert', allow_patterns=['*.json','*.safetensors']))\""
echo "== bob loads everything offline"
run bob "HF_HUB_OFFLINE=1 $PY -c \"from huggingface_hub import snapshot_download as s; import os; p=s('hf-internal-testing/tiny-random-bert', allow_patterns=['*.json','*.safetensors']); print(sorted(os.listdir(p)))\""
echo "== a root-run container writes into the cache, then alice deletes that file"
HF_HUB_CACHE=/srv/hf/hub $PY -c "from huggingface_hub import hf_hub_download as d; d('hf-internal-testing/tiny-random-bert','vocab.txt')"
B=$(readlink -f /srv/hf/hub/models--hf-internal-testing--tiny-random-bert/snapshots/*/vocab.txt)
ls -l "$B" | cut -c1-40; getfacl -p "$B" 2>/dev/null | grep -E "group|mask"
run alice "rm -f $B && echo alice-removed-root-file"
echo "== tokens stay private"
run alice 'python3 -c "import os; print(\"HF_HOME\", os.environ.get(\"HF_HOME\"), \"token path ->\", os.path.expanduser(\"~/.cache/huggingface/token\"))"'
echo "== modes of blobs"
ls -l /srv/hf/hub/models--hf-internal-testing--tiny-random-bert/blobs | head -5 | cut -c1-45
ls -ld /home/lkk/.cache/huggingface/hub; su lkk -c "ls /home/lkk/.cache/huggingface/hub/ | head -3"
echo ALL_OK
