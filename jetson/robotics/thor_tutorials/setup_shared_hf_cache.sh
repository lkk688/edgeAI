#!/bin/bash
# One shared Hugging Face model cache for every user on a multi-user Jetson Thor.
#
#   sudo bash setup_shared_hf_cache.sh                         # create /srv/hf, env, group
#   sudo bash setup_shared_hf_cache.sh --add-user alice bob    # give users access
#   sudo bash setup_shared_hf_cache.sh --migrate-from lkk      # move lkk's existing cache in
#
# Design:
#   * /srv/hf/{hub,xet,datasets} owned by group hfshare, setgid (new files keep the group)
#     plus a DEFAULT ACL g:hfshare:rwX, so files one student downloads stay writable for
#     the next one whatever their umask (huggingface_hub takes lock files and renames
#     blobs in place, so read-only is not enough).
#   * Only the cache locations are shared (HF_HUB_CACHE, HF_XET_CACHE, HF_DATASETS_CACHE).
#     HF_HOME is NOT set, so each user's login token stays in ~/.cache/huggingface/token.
#   * The variables go to /etc/environment (all sessions incl. non-interactive ssh, cron,
#     systemd user units via pam_env) and /etc/profile.d (login shells).
#   * OpenPI's checkpoint cache is shared the same way (OPENPI_DATA_HOME=/srv/openpi).
# Idempotent: safe to re-run.
set -euo pipefail
ROOT=/srv/hf
OPENPI=/srv/openpi
GROUP=hfshare
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
command -v setfacl >/dev/null || apt-get install -y acl

# ---------- group + directories
getent group $GROUP >/dev/null || groupadd $GROUP
for d in $ROOT $ROOT/hub $ROOT/xet $ROOT/datasets $OPENPI; do
  mkdir -p "$d"
done
for d in $ROOT $OPENPI; do
  chgrp -R $GROUP "$d"
  chmod -R g+rwX,o+rX "$d"
  find "$d" -type d -exec chmod g+s {} +
  setfacl -R -m g:$GROUP:rwX "$d"
  find "$d" -type d -exec setfacl -d -m g:$GROUP:rwX -m o::rX {} +
done

# ---------- environment for every session
ENVLINES=(
  "HF_HUB_CACHE=$ROOT/hub"
  "HF_XET_CACHE=$ROOT/xet"
  "HF_DATASETS_CACHE=$ROOT/datasets"
  "OPENPI_DATA_HOME=$OPENPI"
)
for kv in "${ENVLINES[@]}"; do
  k=${kv%%=*}
  grep -q "^$k=" /etc/environment && sed -i "s|^$k=.*|$kv|" /etc/environment || echo "$kv" >> /etc/environment
done
{
  echo "# Shared Hugging Face cache (setup_shared_hf_cache.sh). HF_HOME stays per user (tokens)."
  for kv in "${ENVLINES[@]}"; do echo "export $kv"; done
  echo "umask 002   # group-writable by default too, belt and braces with the ACLs"
} > /etc/profile.d/hf-shared-cache.sh

# ---------- options
while [ $# -gt 0 ]; do
  case "$1" in
    --add-user)
      shift
      while [ $# -gt 0 ] && [[ "$1" != --* ]]; do usermod -aG $GROUP "$1"; echo "added $1 to $GROUP"; shift; done ;;
    --migrate-from)
      u=$2; shift 2
      home=$(getent passwd "$u" | cut -d: -f6)
      usermod -aG $GROUP "$u"
      # Same filesystem -> mv is a rename, instant even for 200 GB. Anything already in the
      # shared cache is left behind in *.migrated for you to inspect and delete.
      move_into() {   # move_into <src dir> <dst dir>
        local src=$1 dst=$2 e
        [ -d "$src" ] && [ ! -L "$src" ] || return 0
        for e in "$src"/* "$src"/.[!.]*; do
          [ -e "$e" ] || continue
          if [ -e "$dst/$(basename "$e")" ]; then echo "keep $(basename "$e") in $src.migrated (already shared)"
          else mv "$e" "$dst/"; fi
        done
        if [ -n "$(ls -A "$src")" ]; then mv "$src" "$src.migrated"; else rmdir "$src"; fi
        # Leave a symlink so absolute paths into the old location (scripts, symlinked
        # snapshots, configs) keep resolving.
        ln -sfn "$dst" "$src" && chown -h "$u:" "$src"
      }
      move_into "$home/.cache/huggingface/hub" "$ROOT/hub"
      move_into "$home/.cache/huggingface/xet" "$ROOT/xet"
      move_into "$home/.cache/openpi" "$OPENPI"
      # re-apply ownership/ACLs to what was moved in
      for d in $ROOT $OPENPI; do
        chgrp -R $GROUP "$d"; chmod -R g+rwX,o+rX "$d"
        find "$d" -type d -exec chmod g+s {} +
        setfacl -R -m g:$GROUP:rwX "$d"
        find "$d" -type d -exec setfacl -d -m g:$GROUP:rwX -m o::rX {} +
      done ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done

echo
echo "Shared cache ready: $ROOT (group $GROUP).  Members: $(getent group $GROUP | cut -d: -f4)"
echo "Users must log out/in once (new group + /etc/environment)."
du -sh $ROOT/hub $OPENPI 2>/dev/null || true
