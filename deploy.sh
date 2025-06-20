#!/bin/bash

# 프로젝트 및 서비스 정보
PROJECT_ID="opensource-project-463412"
SERVICE_NAME="streamlit-nutrition-app"
REGION="asia-northeast3"  # 한국 리전 (서울)
IMAGE_URI="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🔧 GCP 프로젝트 설정 중..."
gcloud config set project $PROJECT_ID

echo "🔨 Docker 이미지 빌드 및 업로드 중..."
gcloud builds submit --tag $IMAGE_URI

echo "🚀 Cloud Run에 배포 중..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_URI \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --timeout 300s

echo "✅ 배포 완료!"
gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)'

