cat > docker-compose.uiurl.yml <<'EOF'
services:
  prefect-server:
    environment:
      PREFECT_UI_API_URL: http://54.161.161.100:4200/api
EOF

docker compose -f docker-compose.prefect.yml -f docker-compose.local.yml -f docker-compose.uiurl.yml up -d prefect-server
