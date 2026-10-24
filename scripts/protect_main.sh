#!/usr/bin/env bash
# Protect `main` and switch the repository to squash merges only.
#
#   bash scripts/protect_main.sh KrishG7/Aegis-Tensor
#
# Needs `gh` logged in as a repository admin. Run it after the quality-gates PR has merged,
# because the required checks below must already exist on `main`.
#
# Branch protection is free on public repositories. On a private repository it needs
# GitHub Pro (included in the GitHub Student Developer Pack) or a paid organisation plan.
set -euo pipefail

repo="${1:?usage: $0 OWNER/REPO}"

# Exact job names from .github/workflows/ci.yml and pr-checks.yml. Renaming a job there
# means updating it here, or every PR waits forever for a check that never reports.
checks=(
  "rust"
  "lint"
  "test (py3.10)"
  "test (py3.12)"
  "test (no torch)"
  "macos"
  "wheel"
  "pr-title"
  "tests-touched"
)

contexts=$(printf '%s\n' "${checks[@]}" | jq -R '{context: .}' | jq -s .)

jq -n --argjson checks "$contexts" '{
  required_status_checks: {strict: true, checks: $checks},
  enforce_admins: false,
  required_pull_request_reviews: {
    required_approving_review_count: 1,
    require_code_owner_reviews: true,
    dismiss_stale_reviews: true,
    require_last_push_approval: true
  },
  restrictions: null,
  required_conversation_resolution: true,
  required_linear_history: true,
  allow_force_pushes: false,
  allow_deletions: false
}' | gh api --method PUT "repos/${repo}/branches/main/protection" --input - > /dev/null
echo "main is protected: ${#checks[@]} required checks, 1 code-owner approval, linear history"

gh api --method PATCH "repos/${repo}" \
  -F allow_squash_merge=true \
  -F allow_merge_commit=false \
  -F allow_rebase_merge=false \
  -F delete_branch_on_merge=true > /dev/null
echo "merges: squash only; branches are deleted after merge"

gh api "repos/${repo}/branches/main/protection" \
  --jq '"required checks: " + ([.required_status_checks.checks[].context] | join(", "))'
