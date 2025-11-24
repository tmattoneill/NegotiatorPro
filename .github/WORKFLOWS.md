# GitHub Workflows Directory

This directory contains GitHub Actions CI/CD workflows.

## Files

- `test.yml` - Comprehensive test suite workflow (unit tests, Docker tests, security scans)

## Important Note

The workflow file in this directory **cannot be pushed by the Claude Code GitHub App** due to security restrictions. GitHub prevents apps from creating/modifying workflow files without explicit `workflows` permission.

## How to Add the Workflow

After merging this PR, you can add the workflow file manually:

1. Via GitHub UI:
   - Go to your repo → "Add file" → "Create new file"
   - Name: `.github/workflows/test.yml`
   - Copy content from this directory's test.yml
   - Commit to main

2. Via your local machine:
   - Pull the merged changes
   - Create `.github/workflows/test.yml`
   - Commit and push

The workflow file exists in the Claude Code environment but is intentionally not pushed to avoid permission errors.
