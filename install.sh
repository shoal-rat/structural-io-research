#!/bin/bash
# Install the structural-io-research skill for Claude Code
# Usage: bash install.sh [--global|--project]

SKILL_DIR=""
MODE="${1:---global}"

case "$MODE" in
  --global)
    SKILL_DIR="$HOME/.claude/skills/structural-io-research"
    echo "Installing globally to $SKILL_DIR"
    ;;
  --project)
    SKILL_DIR=".claude/skills/structural-io-research"
    echo "Installing to project directory: $SKILL_DIR"
    ;;
  *)
    echo "Usage: bash install.sh [--global|--project]"
    echo "  --global   Install to ~/.claude/skills/ (available in all projects)"
    echo "  --project  Install to .claude/skills/ (current project only)"
    exit 1
    ;;
esac

mkdir -p "$SKILL_DIR"
cp SKILL.md "$SKILL_DIR/SKILL.md"

echo "Installed successfully."
echo "Use /structural-io-research in Claude Code to invoke the skill."
