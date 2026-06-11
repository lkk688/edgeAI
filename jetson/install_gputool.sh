#!/bin/bash

SCRIPT_URL="https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/gputool.sh"
INSTALL_PATH="$HOME/.local/bin/gputool"

echo "⬇️ Downloading gputool from GitHub..."
TEMP_SCRIPT="$(mktemp)"

download_file() {
  local url="$1"
  local dest="$2"
  if command -v curl &>/dev/null; then
    curl -fsSL "$url" -o "$dest"
  elif command -v wget &>/dev/null; then
    wget -qO "$dest" "$url"
  elif command -v python3 &>/dev/null; then
    python3 -c "import urllib.request; urllib.request.urlretrieve('$url', '$dest')" &>/dev/null
  else
    return 1
  fi
}

if download_file "$SCRIPT_URL" "$TEMP_SCRIPT"; then
  echo "✅ Downloaded script."

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

  echo "✅ Installed successfully. You can now run: gputool"
else
  echo "❌ Failed to download script from:"
  echo "   $SCRIPT_URL"
  exit 1
fi

#One-liner Installer
#curl -fsSL https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/install_gputool.sh | bash
