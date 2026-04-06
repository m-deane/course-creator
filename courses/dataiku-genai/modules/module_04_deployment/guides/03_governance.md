# Deployment Governance for Gen AI Applications

> **Reading time:** ~11 min | **Module:** 4 — Deployment | **Prerequisites:** Modules 0-3

## In Brief

Deployment governance ensures Gen AI applications run reliably, securely, and cost-effectively in production. It encompasses deployment pipelines, monitoring, cost management, access control, compliance tracking, and incident response—transforming experimental LLM prototypes into enterprise-grade production systems.

<div class="callout-insight">

<strong>Key Insight:</strong> Production Gen AI requires treating LLM applications with the same rigor as traditional software: version control, automated testing, staged deployments, monitoring, alerting, and rollback capabilities. The difference is that Gen AI governance also includes unique concerns like prompt versioning, token budgets, model drift, and output quality monitoring.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> Deployment governance ensures Gen AI applications run reliably, securely, and cost-effectively in production. It encompasses deployment pipelines, monitoring, cost management, access control, compliance tracking, and incident response—transforming experimental LLM prototypes into enterprise-grade...

</div>

## Formal Definition

**Deployment Governance** consists of:
- **Version Control**: Tracking changes to prompts, code, and model configurations
- **Deployment Pipeline**: Staged rollout from dev → staging → production with gates
- **Access Management**: Role-based permissions for deployment and operation
- **Monitoring**: Real-time tracking of performance, costs, and quality metrics
- **Alerting**: Automated notifications for anomalies, errors, or budget overruns
- **Audit Trail**: Complete history of all changes, deployments, and usage
- **Incident Response**: Procedures for handling failures, rollbacks, and escalations

## Intuitive Explanation

Think of deployment governance like air traffic control for Gen AI applications. Air traffic control ensures planes take off and land safely through strict protocols: pre-flight checks (testing), staged handoffs (deployment pipeline), constant monitoring (observability), immediate alerts for problems (alerting), and emergency procedures (incident response). Similarly, deployment governance ensures your Gen AI applications reach production safely and operate reliably through systematic processes and continuous oversight.

## Visual Representation

```

┌─────────────────────────────────────────────────────────────┐
│           Gen AI Deployment Governance Framework            │
└─────────────────────────────────────────────────────────────┘

                    ┌──────────────┐
                    │ Development  │
                    │ Environment  │
                    └──────┬───────┘
                           │ ✓ Tests Pass
                           │ ✓ Code Review
                           ▼
                    ┌──────────────┐
                    │   Staging    │
                    │ Environment  │ • Identical to prod
                    └──────┬───────┘ • Integration tests
                           │ • Performance tests
                           │ ✓ Approval Required
                           ▼
                    ┌──────────────┐
                    │ Production   │
                    │ Environment  │ • Blue-green deploy
                    └──────┬───────┘ • Gradual rollout
                           │ • Rollback ready
                           ▼
      ┌─────────────────────────────────────────────┐
      │          Continuous Monitoring              │
      ├─────────────────────────────────────────────┤
      │ Performance │ Cost │ Quality │ Security     │
      │ • Latency   │ • $  │ • Output│ • Auth      │
      │ • Throughput│ • Tok│ • Errors│ • Audit     │
      │ • Errors    │ • QPM│ • Drift │ • Compliance│
      └─────────────────────────────────────────────┘
                           │
                           │ Alert on Anomaly
                           ▼
                    ┌──────────────┐
                    │   Incident   │
                    │   Response   │
                    └──────────────┘
```

## Code Implementation

### Deployment Configuration

The `DeploymentConfig` dataclass centralizes all environment-specific settings — token limits, monitoring flags, cost budgets — so the same codebase can run safely in dev, staging, and production without hardcoded values scattered across the application.


<span class="filename">deployment_config.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# deployment_config.py
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"

@dataclass
class DeploymentConfig:
    """Configuration for Gen AI deployment."""

    environment: Environment

    # LLM Configuration
    llm_connection: str
    model: str
    temperature: float
    max_tokens: int

    # Resource Limits
    max_concurrent_requests: int
    request_timeout_sec: int
    daily_token_budget: int
    daily_cost_budget_usd: float

    # Monitoring
    enable_logging: bool
    enable_metrics: bool
    alert_email: str

    # Security
    allowed_user_groups: list[str]
    require_approval: bool

    # Version
    prompt_version: str
    deployment_version: str

# Environment-specific configurations
CONFIGS = {
    Environment.DEVELOPMENT: DeploymentConfig(
        environment=Environment.DEVELOPMENT,
        llm_connection="anthropic-claude-dev",
        model="claude-sonnet-4",
        temperature=0.3,
        max_tokens=1000,
        max_concurrent_requests=5,
        request_timeout_sec=30,
        daily_token_budget=100_000,
        daily_cost_budget_usd=10.0,
        enable_logging=True,
        enable_metrics=False,
        alert_email="dev-team@company.com",
        allowed_user_groups=["developers"],
        require_approval=False,
        prompt_version="v1.0-dev",
        deployment_version="dev"
    ),

    Environment.STAGING: DeploymentConfig(
        environment=Environment.STAGING,
        llm_connection="anthropic-claude-staging",
        model="claude-sonnet-4",
        temperature=0.3,
        max_tokens=1000,
        max_concurrent_requests=20,
        request_timeout_sec=60,
        daily_token_budget=500_000,
        daily_cost_budget_usd=50.0,
        enable_logging=True,
        enable_metrics=True,
        alert_email="staging-alerts@company.com",
        allowed_user_groups=["developers", "qa"],
        require_approval=True,
        prompt_version="v1.0-rc1",
        deployment_version="staging"
    ),

    Environment.PRODUCTION: DeploymentConfig(
        environment=Environment.PRODUCTION,
        llm_connection="anthropic-claude-prod",
        model="claude-sonnet-4",
        temperature=0.3,
        max_tokens=1000,
        max_concurrent_requests=100,
        request_timeout_sec=120,
        daily_token_budget=5_000_000,
        daily_cost_budget_usd=500.0,
        enable_logging=True,
        enable_metrics=True,
        alert_email="prod-alerts@company.com",
        allowed_user_groups=["analysts", "data-scientists"],
        require_approval=True,
        prompt_version="v1.0",
        deployment_version="1.0.0"
    )
}

def get_config(env: Environment) -> DeploymentConfig:
    """Get configuration for environment."""
    return CONFIGS[env]
```

</div>

### Deployment Pipeline


<span class="filename">deployment_pipeline.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# deployment_pipeline.py
import dataiku
from dataiku.llm import LLM
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DeploymentPipeline:
    """
    Automated deployment pipeline for Gen AI applications.
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config

    def run_pre_deployment_checks(self) -> bool:
        """
        Run automated checks before deployment.

        Returns:
            True if all checks pass
        """
        checks = []

        # Check 1: LLM connection available
        try:
            llm = LLM(self.config.llm_connection)
            response = llm.complete("test", max_tokens=10)
            checks.append(("LLM Connection", True, None))
        except Exception as e:
            checks.append(("LLM Connection", False, str(e)))

        # Check 2: Prompt version exists
        try:
            # Verify prompt studio version
            prompt_studio = dataiku.PromptStudio("market-analyzer")
            versions = prompt_studio.list_versions()
            version_exists = self.config.prompt_version in versions
            checks.append(("Prompt Version", version_exists,
                         None if version_exists else "Version not found"))
        except Exception as e:
            checks.append(("Prompt Version", False, str(e)))

        # Check 3: Required datasets exist
        try:
            required_datasets = ["input_data", "output_data"]
            for ds_name in required_datasets:
                ds = dataiku.Dataset(ds_name)
                ds.get_schema()  # Verify exists
            checks.append(("Required Datasets", True, None))
        except Exception as e:
            checks.append(("Required Datasets", False, str(e)))

        # Check 4: User permissions configured
        try:
            # Verify security groups exist
            client = dataiku.api_client()
            groups = client.list_groups()
            group_names = [g['name'] for g in groups]

            all_groups_exist = all(
                g in group_names
                for g in self.config.allowed_user_groups
            )
            checks.append(("User Groups", all_groups_exist,
                         None if all_groups_exist else "Some groups not found"))
        except Exception as e:
            checks.append(("User Groups", False, str(e)))

        # Log results
        logger.info("Pre-deployment checks:")
        all_passed = True
        for check_name, passed, error in checks:
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"  {status} - {check_name}")
            if error:
                logger.error(f"    Error: {error}")
            all_passed = all_passed and passed

        return all_passed

    def deploy(self) -> bool:
        """
        Execute deployment.

        Returns:
            True if deployment successful
        """
        logger.info(f"Starting deployment to {self.config.environment.value}")

        # Pre-deployment checks
        if not self.run_pre_deployment_checks():
            logger.error("Pre-deployment checks failed - aborting")
            return False

        # Require approval for staging/prod
        if self.config.require_approval:
            logger.info("Approval required - waiting for confirmation")
            # In real implementation, this would wait for manual approval
            # For now, we'll assume approval granted

        try:
            # Deploy prompt version
            prompt_studio = dataiku.PromptStudio("market-analyzer")
            prompt_studio.load_version(self.config.prompt_version)

            # Update webapp configuration
            webapp = dataiku.Webapp("market-analyzer-app")
            webapp.set_config({
                'llm_connection': self.config.llm_connection,
                'prompt_version': self.config.prompt_version,
                'max_tokens': self.config.max_tokens
            })

            # Record deployment
            self.record_deployment()

            logger.info("Deployment successful")
            return True

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            # Trigger rollback
            self.rollback()
            return False

    def record_deployment(self):
        """Record deployment in audit log."""
        import pandas as pd

        deployment_record = {
            'timestamp': datetime.now(),
            'environment': self.config.environment.value,
            'prompt_version': self.config.prompt_version,
            'deployment_version': self.config.deployment_version,
            'deployed_by': dataiku.get_current_user(),
            'llm_connection': self.config.llm_connection
        }

        # Append to deployment log dataset
        log_df = pd.DataFrame([deployment_record])

        try:
            existing_df = dataiku.Dataset("deployment_log").get_dataframe()
            combined_df = pd.concat([existing_df, log_df], ignore_index=True)
        except:
            combined_df = log_df

        dataiku.Dataset("deployment_log").write_with_schema(combined_df)

    def rollback(self, to_version: Optional[str] = None):
        """
        Rollback to previous version.

        Args:
            to_version: Specific version to rollback to, or None for previous
        """
        logger.warning(f"Initiating rollback in {self.config.environment.value}")

        if to_version is None:
            # Get previous version from deployment log
            log_df = dataiku.Dataset("deployment_log").get_dataframe()
            env_log = log_df[log_df['environment'] == self.config.environment.value]
            env_log = env_log.sort_values('timestamp', ascending=False)

            if len(env_log) < 2:
                logger.error("No previous version to rollback to")
                return False

            to_version = env_log.iloc[1]['prompt_version']

        logger.info(f"Rolling back to version {to_version}")

        try:
            # Load previous version
            prompt_studio = dataiku.PromptStudio("market-analyzer")
            prompt_studio.load_version(to_version)

            # Update webapp
            webapp = dataiku.Webapp("market-analyzer-app")
            webapp.set_config({
                'prompt_version': to_version
            })

            logger.info("Rollback successful")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

# Usage
config = get_config(Environment.PRODUCTION)
pipeline = DeploymentPipeline(config)

if pipeline.deploy():
    print("Deployment successful")
else:
    print("Deployment failed")
```

</div>

### Production Monitoring


<span class="filename">monitoring.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# monitoring.py
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List

@dataclass
class Alert:
    """Alert configuration."""
    name: str
    condition: callable
    severity: str  # "warning" | "critical"
    message: str

class ProductionMonitor:
    """
    Monitor Gen AI application in production.
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.alerts = self.configure_alerts()

    def configure_alerts(self) -> List[Alert]:
        """Define monitoring alerts."""
        return [
            Alert(
                name="High Error Rate",
                condition=lambda metrics: metrics['error_rate'] > 0.05,
                severity="critical",
                message="Error rate exceeds 5%"
            ),
            Alert(
                name="Budget Exceeded",
                condition=lambda metrics: (
                    metrics['daily_cost'] > self.config.daily_cost_budget_usd * 0.9
                ),
                severity="warning",
                message="Approaching daily budget limit"
            ),
            Alert(
                name="High Latency",
                condition=lambda metrics: metrics['p95_latency'] > 10.0,
                severity="warning",
                message="P95 latency exceeds 10 seconds"
            ),
            Alert(
                name="Token Budget Warning",
                condition=lambda metrics: (
                    metrics['daily_tokens'] > self.config.daily_token_budget * 0.9
                ),
                severity="warning",
                message="Approaching daily token limit"
            ),
            Alert(
                name="Unusual Request Volume",
                condition=lambda metrics: (
                    metrics['requests_per_hour'] >
                    metrics['avg_requests_per_hour'] * 2
                ),
                severity="warning",
                message="Request volume 2x above average"
            )
        ]

    def collect_metrics(self, lookback_hours: int = 1) -> Dict:
        """
        Collect metrics from logs.

        Args:
            lookback_hours: Time window for metrics

        Returns:
            Dictionary of metrics
        """
        # Load usage logs
        logs_df = dataiku.Dataset("llm_usage_logs").get_dataframe()

        # Filter to time window
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
        recent_logs = logs_df[logs_df['timestamp'] > cutoff]

        if len(recent_logs) == 0:
            return self.get_empty_metrics()

        # Calculate metrics
        metrics = {
            # Volume
            'total_requests': len(recent_logs),
            'requests_per_hour': len(recent_logs) / lookback_hours,

            # Success/Error
            'successful_requests': (recent_logs['status'] == 'success').sum(),
            'failed_requests': (recent_logs['status'] == 'error').sum(),
            'error_rate': (recent_logs['status'] == 'error').mean(),

            # Latency
            'avg_latency': recent_logs['latency_sec'].mean(),
            'p50_latency': recent_logs['latency_sec'].quantile(0.5),
            'p95_latency': recent_logs['latency_sec'].quantile(0.95),
            'p99_latency': recent_logs['latency_sec'].quantile(0.99),

            # Tokens & Cost
            'total_tokens': recent_logs['total_tokens'].sum(),
            'avg_tokens_per_request': recent_logs['total_tokens'].mean(),
            'total_cost': recent_logs['cost_usd'].sum(),
            'avg_cost_per_request': recent_logs['cost_usd'].mean(),

            # Daily projections
            'daily_tokens': recent_logs['total_tokens'].sum() * (24 / lookback_hours),
            'daily_cost': recent_logs['cost_usd'].sum() * (24 / lookback_hours),

            # Historical average (for anomaly detection)
            'avg_requests_per_hour': self.get_historical_avg_requests_per_hour()
        }

        return metrics

    def get_historical_avg_requests_per_hour(self) -> float:
        """Calculate historical average requests per hour."""
        logs_df = dataiku.Dataset("llm_usage_logs").get_dataframe()

        # Last 7 days
        cutoff = datetime.now() - timedelta(days=7)
        logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
        historical = logs_df[logs_df['timestamp'] > cutoff]

        if len(historical) == 0:
            return 0.0

        return len(historical) / (7 * 24)

    def get_empty_metrics(self) -> Dict:
        """Return empty metrics when no data."""
        return {
            'total_requests': 0,
            'requests_per_hour': 0,
            'error_rate': 0,
            'avg_latency': 0,
            'p95_latency': 0,
            'total_tokens': 0,
            'total_cost': 0,
            'daily_tokens': 0,
            'daily_cost': 0,
            'avg_requests_per_hour': 0
        }

    def check_alerts(self, metrics: Dict) -> List[Alert]:
        """
        Check if any alerts should fire.

        Args:
            metrics: Current metrics

        Returns:
            List of triggered alerts
        """
        triggered = []

        for alert in self.alerts:
            try:
                if alert.condition(metrics):
                    triggered.append(alert)
            except Exception as e:
                logger.error(f"Alert check failed for {alert.name}: {e}")

        return triggered

    def send_alert(self, alert: Alert, metrics: Dict):
        """
        Send alert notification.

        Args:
            alert: Alert to send
            metrics: Context metrics
        """
        import smtplib
        from email.mime.text import MIMEText

        subject = f"[{alert.severity.upper()}] {alert.name}"

        body = f"""
Alert: {alert.name}
Severity: {alert.severity}
Environment: {self.config.environment.value}

Message: {alert.message}

Current Metrics:
- Requests: {metrics['total_requests']}
- Error Rate: {metrics['error_rate']:.2%}
- Avg Latency: {metrics['avg_latency']:.2f}s
- Daily Cost: ${metrics['daily_cost']:.2f}
- Daily Tokens: {metrics['daily_tokens']:,}

Timestamp: {datetime.now()}
        """

        logger.warning(f"ALERT: {alert.name} - {alert.message}")

        # Send email (simplified - in production use proper email service)
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = 'genai-monitoring@company.com'
            msg['To'] = self.config.alert_email

            # Send email (requires SMTP configuration)
            # smtp = smtplib.SMTP('smtp.company.com')
            # smtp.send_message(msg)
            # smtp.quit()

            logger.info(f"Alert sent to {self.config.alert_email}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def run_monitoring_cycle(self):
        """Run one monitoring cycle."""
        logger.info("Starting monitoring cycle")

        # Collect metrics
        metrics = self.collect_metrics(lookback_hours=1)

        # Check alerts
        triggered_alerts = self.check_alerts(metrics)

        # Send alerts
        for alert in triggered_alerts:
            self.send_alert(alert, metrics)

        # Log metrics
        self.log_metrics(metrics)

        logger.info(f"Monitoring cycle complete - {len(triggered_alerts)} alerts triggered")

    def log_metrics(self, metrics: Dict):
        """Log metrics to monitoring dataset."""
        import pandas as pd

        metrics_record = {
            'timestamp': datetime.now(),
            'environment': self.config.environment.value,
            **metrics
        }

        metrics_df = pd.DataFrame([metrics_record])

        try:
            existing_df = dataiku.Dataset("monitoring_metrics").get_dataframe()
            combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)
        except:
            combined_df = metrics_df

        dataiku.Dataset("monitoring_metrics").write_with_schema(combined_df)

# Usage - run as scheduled job
config = get_config(Environment.PRODUCTION)
monitor = ProductionMonitor(config)
monitor.run_monitoring_cycle()
```

</div>

## Common Pitfalls

**Pitfall 1: No Staged Rollout**
- Deploying directly to production without staging environment increases risk
- Always use dev → staging → production pipeline
- Validate in staging with production-like load

**Pitfall 2: Insufficient Monitoring**
- Deploying without monitoring makes debugging impossible
- Track all critical metrics: latency, errors, costs, quality
- Set up alerts before problems occur

**Pitfall 3: No Rollback Plan**
- Deployments without rollback capability leave you stuck with broken versions
- Maintain previous versions and test rollback procedures
- Document rollback process and automate where possible

**Pitfall 4: Missing Cost Controls**
- Production without budget limits can cause runaway costs
- Implement hard limits and soft alerts
- Review costs daily in early production

**Pitfall 5: Inadequate Access Control**
- Over-permissive access in production creates security and audit risks
- Follow principle of least privilege
- Separate development and production permissions

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>

</div>

**Builds on:**
- LLM Mesh governance (Module 0.3)
- Custom model wrappers (Module 3.2)
- Webapp deployment (Module 4.2)

**Leads to:**
- Advanced monitoring patterns
- Multi-environment orchestration
- Continuous improvement workflows

**Related to:**
- DevOps and CI/CD practices
- Site Reliability Engineering (SRE)
- Production operations

## Practice Problems

1. **Deployment Pipeline**
   - Create a 3-stage deployment pipeline (dev, staging, prod)
   - Implement automated checks at each gate
   - Test deployment and rollback procedures

2. **Monitoring Dashboard**
   - Build a monitoring dashboard showing key production metrics
   - Include: request volume, error rate, latency, costs
   - Add time-series visualizations

3. **Alert Configuration**
   - Define 5 critical alerts for production Gen AI app
   - Set appropriate thresholds based on baseline metrics
   - Test alert triggering with simulated issues

4. **Incident Response**
   - Create an incident response playbook for common Gen AI failures
   - Include: runaway costs, high error rates, slow responses
   - Document step-by-step resolution procedures

5. **Cost Governance**
   - Implement a budget enforcement system
   - Track costs by user, project, and time period
   - Generate monthly cost reports with variance analysis

## Further Reading

- **Dataiku Documentation**: [Deployment and Governance](https://doc.dataiku.com/dss/latest/deployment/index.html) - Official deployment best practices

- **Google SRE Book**: [Site Reliability Engineering](https://sre.google/sre-book/table-of-contents/) - Production operations principles

- **Martin Fowler**: [Continuous Delivery](https://martinfowler.com/bliki/ContinuousDelivery.html) - Deployment pipeline patterns

- **Blog Post**: "Production LLM Governance at Scale" - Real-world governance frameworks (representative of enterprise practices)

- **Research**: "Operational Patterns for Production Machine Learning Systems" - Academic treatment of ML operations (applicable to Gen AI)


## Resources

<a class="link-card" href="../notebooks/01_api_setup.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
