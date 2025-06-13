# Email Service Documentation

The Risk Tool Email Service provides comprehensive email functionality for sending trading signals, predictions, daily summaries, and other automated reports to stakeholders.

## Features

- **Automated Signal Emails**: Generate and send daily trading predictions
- **Professional HTML Templates**: Beautiful, responsive email layouts
- **Flexible Template System**: Easy to expand with new email types
- **Retry Logic**: Robust email delivery with automatic retries
- **Attachment Support**: Include files like CSV reports, charts
- **Multiple Recipients**: Support for CC, BCC, and multiple recipients
- **Test Functionality**: Built-in testing for email configuration
- **Environment-Based Configuration**: Secure credential management

## Quick Start

### 1. Environment Setup

Create a `.env` file or set environment variables:

```bash
# Required
EMAIL_FROM=your-email@gmail.com
EMAIL_PASSWORD=your-gmail-app-password
EMAIL_TO=recipient@example.com

# Optional
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
USE_TLS=true
```

### 2. Send Trading Signals

```bash
# Send signals to default email
python make_signal.py

# Send to specific recipients
python make_signal.py --to "trader1@example.com,trader2@example.com"

# Generate predictions but don't send (dry run)
python make_signal.py --dry-run

# Include attachments
python make_signal.py --attachments
```

### 3. Test Email Service

```bash
# Test SMTP connection
python make_signal.py --test-connection

# Send test email
python make_signal.py --test
```

## Architecture

### Core Components

1. **EmailConfig**: Configuration management with environment variable support
2. **EmailSender**: Core SMTP functionality with retry logic and error handling
3. **SignalEmailService**: Specialized service for trading signal emails
4. **SignalCommand**: Command-line interface for signal generation
5. **Template System**: Jinja2-based HTML templates for professional emails

### Directory Structure

```
src/email_service/
├── __init__.py              # Module exports
├── email_config.py          # Configuration management
├── email_sender.py          # Core email functionality
├── signal_email.py          # Signal email service
├── signal_command.py        # CLI command handler
├── templates/               # HTML email templates
│   ├── base_email.html      # Base template for extension
│   ├── signal_email.html    # Trading signals template
│   ├── daily_summary.html   # Daily summary template
│   └── weekly_report.html   # Weekly report template
└── README.md               # This documentation
```

## Email Templates

### Signal Email Template

The main trading signal template includes:

- **Portfolio Summary**: Total traders, active signals, expected PnL
- **Model Performance**: Hit rate, Sharpe ratio, R² score
- **Individual Predictions**: Per-trader predictions with confidence scores
- **Market Context**: Volatility, trends, risk levels
- **Alerts**: Automated warnings and notifications
- **Professional Styling**: Responsive design with modern UI

### Template Variables

Signal emails support these template variables:

```python
{
    'report_date': 'Monday, June 12, 2025',
    'total_traders': 5,
    'active_signals': 3,
    'expected_pnl': '$1,250.50',
    'traders': [
        {
            'id': 'TRADER001',
            'name': 'Alpha Trader',
            'predicted_pnl': '$150.75',
            'confidence': '82.5',
            'recommendation': '🚀 Strong Buy Signal',
            'recent_performance': '$320.50'
        }
    ],
    'model_performance': {
        'hit_rate': '68.5',
        'sharpe_ratio': '1.25',
        'r2_score': '0.150'
    },
    'alerts': [
        {
            'type': 'success',
            'title': 'High Confidence Signals',
            'message': '2 traders have predictions with >80% confidence.'
        }
    ]
}
```

## API Usage

### Basic Email Sending

```python
from email_service import EmailSender, EmailConfig

# Initialize with environment config
config = EmailConfig.from_env()
sender = EmailSender(config)

# Send simple email
success = sender.send_email(
    to_emails='recipient@example.com',
    subject='Test Email',
    html_content='<h1>Hello World</h1>',
    text_content='Hello World'
)
```

### Signal Email Service

```python
from email_service import SignalEmailService

# Initialize service
signal_service = SignalEmailService()

# Prepare prediction data
predictions = {
    'traders': {
        'TRADER001': {
            'name': 'Alpha Trader',
            'predicted_pnl': 150.75,
            'confidence': 0.82,
            'recent_performance': 320.50
        }
    },
    'model_performance': {
        'hit_rate': 0.685,
        'sharpe_ratio': 1.25,
        'r2_score': 0.15
    }
}

# Send signal email
success = signal_service.send_signal_email(
    predictions=predictions,
    to_emails='trader@example.com',
    include_performance=True,
    attachment_paths=['reports/predictions.csv']
)
```

### Command-Line Interface

```python
from email_service.signal_command import SignalCommand

# Initialize command
signal_cmd = SignalCommand()

# Generate and send signals
success = signal_cmd.make_signal(
    data_path='data/processed/features_latest.csv',
    model_path='results/models_latest',
    to_emails='trader1@example.com,trader2@example.com',
    include_performance=True,
    include_attachments=True,
    dry_run=False
)
```

## Configuration Options

### EmailConfig Parameters

```python
@dataclass
class EmailConfig:
    # SMTP Configuration
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True

    # Authentication
    email_from: str = ""
    email_password: str = ""
    email_to: str = ""

    # Performance
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5

    # Templates
    template_dir: str = "src/email_service/templates"
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `EMAIL_FROM` | Yes | - | Sender email address |
| `EMAIL_PASSWORD` | Yes | - | Email password (app password for Gmail) |
| `EMAIL_TO` | Yes | - | Default recipient email |
| `SMTP_SERVER` | No | smtp.gmail.com | SMTP server hostname |
| `SMTP_PORT` | No | 587 | SMTP server port |
| `USE_TLS` | No | true | Enable TLS encryption |
| `EMAIL_TIMEOUT` | No | 30 | Connection timeout in seconds |
| `EMAIL_MAX_RETRIES` | No | 3 | Maximum retry attempts |
| `EMAIL_RETRY_DELAY` | No | 5 | Delay between retries in seconds |

## Gmail Setup

### App Password Setup

1. Enable 2-Factor Authentication on Gmail
2. Go to Google Account Settings → Security → App passwords
3. Generate an app password for "Mail"
4. Use the generated password (not your regular password) in `EMAIL_PASSWORD`

### Example Gmail Configuration

```bash
EMAIL_FROM=your-email@gmail.com
EMAIL_PASSWORD=abcd-efgh-ijkl-mnop  # 16-character app password
EMAIL_TO=recipient@example.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
USE_TLS=true
```

## Template Development

### Creating New Templates

1. Create HTML file in `src/email_service/templates/`
2. Use Jinja2 template syntax for variables
3. Follow responsive design patterns from existing templates
4. Test with different email clients

### Template Guidelines

- **Responsive Design**: Templates should work on mobile and desktop
- **Email Client Compatibility**: Test with Gmail, Outlook, Apple Mail
- **Inline CSS**: Many email clients strip external stylesheets
- **Alt Text**: Provide alt text for images
- **Fallback Content**: Provide plain text versions

### Extending Templates

```html
<!-- Extend base template -->
{% extends "base_email.html" %}

{% block content %}
<div class="custom-content">
    <h2>{{ custom_title }}</h2>
    <p>{{ custom_message }}</p>
</div>
{% endblock %}
```

## Error Handling

### Common Issues

1. **Authentication Errors**: Check app password and 2FA setup
2. **Connection Timeouts**: Verify SMTP server and port
3. **Template Errors**: Check Jinja2 syntax and template variables
4. **Attachment Issues**: Verify file paths and permissions

### Debugging

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test connection
from email_service import EmailSender, EmailConfig
config = EmailConfig.from_env()
sender = EmailSender(config)
sender.test_connection()

# Send test email
sender.send_test_email()
```

## Expansion Points

The email service is designed for easy expansion:

### 1. New Email Types

- Daily performance summaries
- Weekly/monthly reports
- Risk alerts and notifications
- Model performance updates
- System health reports

### 2. Enhanced Features

- **Charts and Visualizations**: Embed performance charts
- **Interactive Elements**: Add buttons and calls-to-action
- **Personalization**: Trader-specific customization
- **Scheduling**: Automated daily/weekly sending
- **Analytics**: Email open and click tracking

### 3. Integration Points

- **Dashboard Links**: Direct links to web dashboard
- **API Integration**: Connect with external trading systems
- **Database Storage**: Log email history and delivery status
- **Notification Webhooks**: Slack/Teams integration

### 4. Template Expansion

```python
# Add new template types
class ReportEmailService(SignalEmailService):
    def send_daily_summary(self, summary_data):
        template_data = self._prepare_summary_template(summary_data)
        html_content = self._render_template('daily_summary.html', template_data)
        return self.send_email(...)

    def send_weekly_report(self, report_data):
        template_data = self._prepare_report_template(report_data)
        html_content = self._render_template('weekly_report.html', template_data)
        return self.send_email(...)
```

## Security Considerations

- **Credential Management**: Never commit passwords to version control
- **App Passwords**: Use Gmail app passwords instead of regular passwords
- **TLS Encryption**: Always use encrypted connections
- **Sensitive Data**: Be careful with PnL and trading data in emails
- **Rate Limiting**: Respect email provider rate limits
- **Audit Trail**: Log email sending for compliance

## Performance Tips

- **Batch Operations**: Send multiple emails efficiently
- **Template Caching**: Cache compiled templates
- **Connection Pooling**: Reuse SMTP connections
- **Async Sending**: Use async libraries for high volume
- **Attachment Optimization**: Compress large attachments

## Testing

```bash
# Test all functionality
python -m pytest tests/test_email_service.py

# Manual testing
python make_signal.py --test-connection  # Test SMTP
python make_signal.py --test             # Test email sending
python make_signal.py --dry-run          # Test signal generation
```

## Support

For issues and questions:

1. Check the logs for detailed error messages
2. Verify environment configuration
3. Test SMTP connection independently
4. Review Gmail security settings
5. Contact system administrator for access issues
