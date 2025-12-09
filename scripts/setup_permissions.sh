#!/bin/bash
set -e

# Get the absolute path to the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
CRATE_DIR="$PROJECT_ROOT/nanoforge"

echo "Navigating to crate directory: $CRATE_DIR"
cd "$CRATE_DIR"

# 1. Build the project
echo "Building NanoForge..."
cargo build

# 2. Set Capabilities
BINARY="target/debug/nanoforge"

if [ -f "$BINARY" ]; then
    echo "Setting CAP_PERFMON on $BINARY..."
    # Try CAP_PERFMON first (Linux 5.8+), fallback to CAP_SYS_ADMIN
    sudo setcap cap_perfmon,cap_sys_admin+ep "$BINARY" || echo "Warning: setcap failed. You might need to install libcap2-bin."
    
    # Verify
    getcap "$BINARY"
    
    echo "Done. You can now run '$BINARY' as a normal user."
else
    echo "Error: Binary not found at $BINARY"
    exit 1
fi
