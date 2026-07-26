#!/bin/bash
# =============================================================================
# sjsujetsontool Installer (Supports v1 and v2)
# Usage:
#   ./install_sjsujetsontool.sh        (Interactive prompt for version choice)
#   ./install_sjsujetsontool.sh v1     (Install v1 - JetPack 6 default)
#   ./install_sjsujetsontool.sh v2     (Install v2 - JetPack 6 & 7 + Thor + vLLM)
# =============================================================================

VERSION_ARG="${1:-}"

if [ -z "$VERSION_ARG" ]; then
  if [ -t 0 ]; then
    echo "=================================================="
    echo " 🚀 Select sjsujetsontool version to install:"
    echo "   1) v1 (JetPack 6 default for Orin Nano)"
    echo "   2) v2 (JetPack 6 & 7 switchable + Jetson Thor + vLLM Cosmos-Reason2)"
    echo "=================================================="
    read -p "Select version [1-2, default: 2]: " choice
    case "$choice" in
      1|v1) VERSION_CHOICE="v1" ;;
      2|v2|"") VERSION_CHOICE="v2" ;;
      *) VERSION_CHOICE="v2" ;;
    esac
  else
    VERSION_CHOICE="v2"
  fi
else
  case "$VERSION_ARG" in
    v1|1) VERSION_CHOICE="v1" ;;
    v2|2|*) VERSION_CHOICE="v2" ;;
  esac
fi

if [ "$VERSION_CHOICE" = "v1" ]; then
  SCRIPT_NAME="sjsujetsontool.sh"
  echo "📦 Selected sjsujetsontool v1."
else
  SCRIPT_NAME="sjsujetsontoolv2.sh"
  echo "📦 Selected sjsujetsontool v2."
fi

SCRIPT_URL="https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/${SCRIPT_NAME}"
INSTALL_PATH="$HOME/.local/bin/sjsujetsontool"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"

TEMP_SCRIPT="$(mktemp)"

# Try local copy first if available in jetson/ directory
if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/$SCRIPT_NAME" ]; then
  echo "📂 Found local file at $SCRIPT_DIR/$SCRIPT_NAME."
  cp "$SCRIPT_DIR/$SCRIPT_NAME" "$TEMP_SCRIPT"
elif [ -f "./jetson/$SCRIPT_NAME" ]; then
  cp "./jetson/$SCRIPT_NAME" "$TEMP_SCRIPT"
else
  echo "⬇️ Downloading $SCRIPT_NAME from GitHub..."
  if ! curl -fsSL "$SCRIPT_URL" -o "$TEMP_SCRIPT"; then
    echo "❌ Failed to download script from $SCRIPT_URL"
    exit 1
  fi
fi

echo "📦 Installing to $INSTALL_PATH"
mkdir -p "$(dirname "$INSTALL_PATH")"
[ -f "$INSTALL_PATH" ] && cp "$INSTALL_PATH" "${INSTALL_PATH}.bak"
mv "$TEMP_SCRIPT" "$INSTALL_PATH"
chmod +x "$INSTALL_PATH"

# Auto-add to PATH if missing
if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
  echo "🛠️  Adding ~/.local/bin to your PATH..."

  SHELL_RC=""
  if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
  elif [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bashrc"
  else
    SHELL_RC="$HOME/.profile"
  fi

  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
  echo "✅ Added to $SHELL_RC"
  echo "👉 Please run: source $SHELL_RC"
fi

echo "✅ Installed sjsujetsontool ($VERSION_CHOICE) successfully. You can now run: sjsujetsontool"