#!/usr/bin/env bash
# ─────────────────────────────────────────────────────
# Vertex RAG — One-command setup script
# Usage: bash setup.sh
# ─────────────────────────────────────────────────────

set -e

echo ""
echo "🚀 Vertex RAG Setup"
echo "─────────────────────────────────"

# 1. Check Python
python3 --version || { echo "❌ Python 3 not found"; exit 1; }

# 2. Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo ""
  echo "⚠️  ANTHROPIC_API_KEY not set."
  echo "   Run: export ANTHROPIC_API_KEY=sk-ant-..."
  echo ""
fi

# 3. Install Python deps
echo ""
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt --quiet

# 4. Check Docker / Milvus
echo ""
echo "🐳 Checking Milvus (Docker)..."
if ! docker info > /dev/null 2>&1; then
  echo "⚠️  Docker not running. Start Docker Desktop, then re-run this script."
  echo "   Or start Milvus manually: docker run -d --name milvus-standalone -p 19530:19530 milvusdb/milvus:v2.4.0 milvus run standalone"
else
  # Try to start Milvus if not already running
  if ! docker ps | grep -q milvus; then
    echo "   Starting Milvus standalone..."
    docker run -d --name milvus-standalone \
      -p 19530:19530 \
      -p 9091:9091 \
      milvusdb/milvus:v2.4.0 \
      milvus run standalone 2>/dev/null || \
    docker start milvus-standalone 2>/dev/null || \
    echo "   ℹ️  Couldn't auto-start Milvus. Run manually if needed."
    sleep 5
  else
    echo "   ✅ Milvus already running"
  fi
fi

# 5. Generate sample docs
echo ""
echo "📄 Generating sample documents..."
python3 demo_data.py

# 6. Done
echo ""
echo "─────────────────────────────────"
echo "✅ Setup complete!"
echo ""
echo "▶️  To start the app:"
echo "   streamlit run app.py"
echo ""
