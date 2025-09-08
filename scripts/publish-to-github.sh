#!/usr/bin/env bash
set -euo pipefail

OWNER="florin1975"
REPO="aibasedria"
REMOTE="git@github.com:${OWNER}/${REPO}.git"

echo "Checking GitHub CLI..."
if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) not found. In Codespaces: sudo apt-get update && sudo apt-get install gh -y" >&2
  exit 1
fi

echo "Verifying GitHub auth..."
if ! gh auth status -h github.com >/dev/null 2>&1; then
  echo "No auth; starting gh auth login (web-based)."
  gh auth login -h github.com -p https -w
fi

echo "Configuring git remote origin -> ${REMOTE}"
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "${REMOTE}"
else
  git remote add origin "${REMOTE}"
fi

echo "Ensuring repo exists on GitHub..."
if ! gh repo view "${OWNER}/${REPO}" >/dev/null 2>&1; then
  gh repo create "${OWNER}/${REPO}" --public --source . --remote origin --push
fi

echo "Pushing main and PR branches..."
for BR in main track/codex/pr01-bootstrap track/codex/pr02-wizard track/codex/pr03-api track/codex/pr04-pages-docs track/codex/pr05-finishing; do
  if git show-ref --verify --quiet refs/heads/$BR; then
    git push -u origin $BR || true
  fi
done

echo "Opening draft PRs (PR01, PR04)..."
gh pr create --base main --head track/codex/pr01-bootstrap --title "feat(repo): scaffold mono-repo, CI, Pages, devcontainer" --body-file docs/prs/PR01.md --draft || true
gh pr create --base main --head track/codex/pr04-pages-docs --title "chore(ci+docs): Pages deploy + docs polish" --body-file docs/prs/PR04.md --draft || true

echo "Links (use if PR creation failed):"
echo "- https://github.com/${OWNER}/${REPO}/compare/main...track/codex/pr01-bootstrap?quick_pull=1"
echo "- https://github.com/${OWNER}/${REPO}/compare/main...track/codex/pr04-pages-docs?quick_pull=1"

echo "Pages URL: https://${OWNER}.github.io/${REPO}/"
echo "Reminder to set repo settings:"
echo "- Settings → Pages → Source: GitHub Actions"
echo "- Settings → Actions → Workflow permissions: Read and write"

