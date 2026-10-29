#!/bin/bash
# Helper script to link and push LoanAI to your GitHub repository
set -e

if [ -z "$1" ]; then
  echo "❌ Error: Please provide your GitHub repository URL."
  echo ""
  echo "Usage:"
  echo "  ./push_to_github.sh https://github.com/<your-username>/<repo-name>.git"
  echo ""
  echo "Example:"
  echo "  ./push_to_github.sh https://github.com/kamakshikalpushu/LoanAI.git"
  exit 1
fi

REPO_URL="$1"

echo "🚀 Linking remote origin to: $REPO_URL"
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
git branch -M main

echo "📤 Pushing code to GitHub..."
git push -u origin main

echo ""
echo "✅ Success! Your code is now live on GitHub."
echo "👉 You can now go to https://share.streamlit.io to deploy it."
