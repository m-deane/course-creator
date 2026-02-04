# Deployment Guide: Real-Time Forecasting Dashboard

## Overview

This guide covers production deployment of the complete nowcasting system to AWS, including auto-scaling, monitoring, and CI/CD.

---

## Architecture

```
Users → CloudFront → ALB → [API Servers (ECS)] → RDS
                          ↓
                    [Dashboard (ECS)]
                          ↓
                   ElastiCache (Redis)

Data Pipeline:
EventBridge → Lambda → [Fetch Data] → S3 → [Update Model] → Database
```

---

## Prerequisites

- AWS Account with admin access
- Docker installed locally
- AWS CLI configured
- GitHub repository

---

## Step 1: Containerize Application

**Dockerfile.api:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY backend/ ./backend/

# Run API server
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Dockerfile.dashboard:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ ./frontend/

CMD ["python", "frontend/app.py"]
```

**docker-compose.yml (local development):**
```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/nowcasting
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    ports:
      - "8050:8050"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: nowcasting
      POSTGRES_PASSWORD: password
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7

volumes:
  pgdata:
```

**Test locally:**
```bash
docker-compose up --build
```

---

## Step 2: Setup AWS Infrastructure

### 2.1 VPC and Networking

```bash
# Create VPC
aws ec2 create-vpc --cidr-block 10.0.0.0/16 --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=nowcast-vpc}]'

# Create subnets (public and private)
# ... (full commands in infrastructure/terraform/)
```

**Or use Terraform:**

```hcl
# infrastructure/terraform/main.tf

provider "aws" {
  region = "us-east-1"
}

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "nowcast-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
}

# RDS Database
module "db" {
  source = "terraform-aws-modules/rds/aws"

  identifier = "nowcast-db"
  engine     = "postgres"
  instance_class    = "db.t3.small"
  allocated_storage = 20

  db_name  = "nowcasting"
  username = "admin"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name   = module.vpc.database_subnet_group_name
}

# ... (see full terraform config)
```

### 2.2 Container Registry (ECR)

```bash
# Create ECR repositories
aws ecr create-repository --repository-name nowcast-api
aws ecr create-repository --repository-name nowcast-dashboard

# Build and push images
$(aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com)

docker build -t nowcast-api -f Dockerfile.api .
docker tag nowcast-api:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/nowcast-api:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/nowcast-api:latest
```

### 2.3 ECS Cluster

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name nowcast-cluster

# Create task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

**task-definition.json:**
```json
{
  "family": "nowcast-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/nowcast-api:latest",
      "portMappings": [{"containerPort": 8000}],
      "environment": [
        {"name": "DATABASE_URL", "value": "postgresql://..."}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/nowcast-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "api"
        }
      }
    }
  ]
}
```

### 2.4 Application Load Balancer

```bash
# Create ALB
aws elbv2 create-load-balancer \
  --name nowcast-alb \
  --subnets subnet-12345 subnet-67890 \
  --security-groups sg-12345

# Create target group
aws elbv2 create-target-group \
  --name nowcast-api-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-12345 \
  --target-type ip \
  --health-check-path /health
```

### 2.5 Auto-Scaling

```bash
# Create auto-scaling target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/nowcast-cluster/nowcast-api \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 10

# Create scaling policy (CPU-based)
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/nowcast-cluster/nowcast-api \
  --policy-name cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

---

## Step 3: CI/CD Pipeline

**GitHub Actions (.github/workflows/deploy.yml):**

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/ --cov

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Login to ECR
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

      - name: Build and push
        run: |
          docker build -t nowcast-api -f Dockerfile.api .
          docker tag nowcast-api:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/nowcast-api:${{ github.sha }}
          docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/nowcast-api:${{ github.sha }}

      - name: Update ECS service
        run: |
          aws ecs update-service \
            --cluster nowcast-cluster \
            --service nowcast-api \
            --force-new-deployment
```

---

## Step 4: Monitoring & Alerts

### 4.1 CloudWatch Logs

All container logs automatically go to CloudWatch.

**Query example:**
```
fields @timestamp, @message
| filter @message like /ERROR/
| sort @timestamp desc
| limit 100
```

### 4.2 CloudWatch Alarms

```bash
# High error rate alarm
aws cloudwatch put-metric-alarm \
  --alarm-name api-high-error-rate \
  --metric-name Errors \
  --namespace AWS/ECS \
  --statistic Sum \
  --period 300 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-east-1:123456789012:alerts

# High API latency
aws cloudwatch put-metric-alarm \
  --alarm-name api-high-latency \
  --metric-name TargetResponseTime \
  --namespace AWS/ApplicationELB \
  --statistic Average \
  --period 60 \
  --threshold 0.5 \
  --comparison-operator GreaterThanThreshold \
  --alarm-actions arn:aws:sns:us-east-1:123456789012:alerts
```

### 4.3 Custom Metrics

**In your code:**
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

def log_nowcast_accuracy(rmse):
    """Log custom metric to CloudWatch."""
    cloudwatch.put_metric_data(
        Namespace='Nowcasting',
        MetricData=[
            {
                'MetricName': 'RMSE',
                'Value': rmse,
                'Unit': 'None'
            }
        ]
    )
```

### 4.4 Grafana Dashboard

```yaml
# Install Grafana on EC2 or use managed Grafana

# Connect to CloudWatch data source
# Import dashboard: grafana_dashboard.json

# Panels:
# - API request rate
# - Error rate
# - P50/P95 latency
# - Nowcast RMSE over time
# - Data freshness (minutes since last update)
```

---

## Step 5: Data Pipeline Deployment

**AWS EventBridge rule:**
```bash
# Trigger Lambda daily at 10 AM EST
aws events put-rule \
  --name daily-data-fetch \
  --schedule-expression "cron(0 14 * * ? *)"

aws events put-targets \
  --rule daily-data-fetch \
  --targets "Id"="1","Arn"="arn:aws:lambda:us-east-1:123456789012:function:fetch-data"
```

**Lambda function:**
```python
# lambda/fetch_data.py
import boto3
from fredapi import Fred

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """Fetch latest data and store to S3."""
    fred = Fred(api_key=os.environ['FRED_API_KEY'])

    # Fetch data
    data = fetch_all_indicators(fred)

    # Save to S3
    timestamp = datetime.now().strftime('%Y%m%d')
    s3.put_object(
        Bucket='nowcast-data',
        Key=f'vintages/vintage_{timestamp}.csv',
        Body=data.to_csv()
    )

    # Trigger model update
    # (invoke another Lambda or send to SQS)

    return {'statusCode': 200}
```

---

## Cost Estimate

**Monthly AWS costs (moderate usage):**

| Service | Usage | Cost |
|---------|-------|------|
| ECS Fargate | 2 tasks 24/7 | $50 |
| RDS (db.t3.small) | 1 instance | $30 |
| ALB | 1 load balancer | $20 |
| Lambda | 1000 invocations/month | $1 |
| S3 | 50 GB storage | $1 |
| CloudWatch | Logs + metrics | $10 |
| Data transfer | 100 GB/month | $10 |
| **Total** | | **~$122/month** |

**Cost optimization:**
- Use Spot instances for batch jobs
- Enable S3 lifecycle policies (archive old vintages)
- Use CloudWatch Logs Insights instead of exporting all logs

---

## Security Checklist

- [ ] RDS in private subnet
- [ ] Secrets in AWS Secrets Manager (not environment variables)
- [ ] IAM roles with least privilege
- [ ] API authentication (JWT tokens)
- [ ] HTTPS only (ACM certificate on ALB)
- [ ] Security groups properly configured
- [ ] Enable AWS WAF on ALB
- [ ] VPC Flow Logs enabled
- [ ] Regular security audits (AWS Trusted Advisor)

---

## Backup & Disaster Recovery

**RDS automated backups:**
```bash
aws rds modify-db-instance \
  --db-instance-identifier nowcast-db \
  --backup-retention-period 7 \
  --preferred-backup-window "03:00-04:00"
```

**S3 versioning:**
```bash
aws s3api put-bucket-versioning \
  --bucket nowcast-data \
  --versioning-configuration Status=Enabled
```

**Disaster recovery plan:**
1. Multi-AZ RDS deployment (automatic failover)
2. ECS tasks across multiple AZs
3. S3 cross-region replication (optional)
4. Daily automated backups (7-day retention)

---

## Performance Optimization

1. **API caching:** Use Redis for frequently-accessed nowcasts
2. **Database indexing:** Index on (target_variable, vintage_date)
3. **CDN:** CloudFront in front of dashboard assets
4. **Connection pooling:** Use SQLAlchemy connection pool
5. **Async processing:** Use Celery for long-running tasks

---

## Troubleshooting

**Issue: ECS tasks keep restarting**
```bash
# Check logs
aws logs tail /ecs/nowcast-api --follow

# Common causes:
# - Database connection timeout (check security groups)
# - Missing environment variables
# - OOM (increase task memory)
```

**Issue: High API latency**
```bash
# Check CloudWatch metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ApplicationELB \
  --metric-name TargetResponseTime \
  --dimensions Name=LoadBalancer,Value=app/nowcast-alb/abc123 \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Average

# Optimize:
# - Add caching
# - Scale up ECS tasks
# - Optimize database queries
```

---

## Next Steps

After deployment:

1. **Monitor for 1 week** - Watch metrics, fix issues
2. **Load testing** - Use Locust/JMeter to test capacity
3. **Set up alerts** - Email/Slack for critical issues
4. **Documentation** - API docs (Swagger), runbooks
5. **User training** - Demo dashboard to stakeholders

---

**Congratulations!** You've deployed a production-grade nowcasting system. This is portfolio-worthy material.
