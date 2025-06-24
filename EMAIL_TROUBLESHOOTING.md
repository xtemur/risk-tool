# Email Service Troubleshooting

## Issue: Email Timeout (Operation timed out)

The email service is experiencing timeout errors when trying to connect to Gmail's SMTP servers.

### Quick Solutions

#### Option 1: Use Local File Generation (Recommended)
```bash
# Generate HTML files instead of sending emails
python send_daily_signals_local.py

# Open in browser automatically
python send_daily_signals_local.py --open-browser
```

This creates:
- HTML file: `inference/outputs/risk_signals_[DATE]_[TIME].html`
- Text summary: `inference/outputs/risk_signals_[DATE]_[TIME]_summary.txt`

#### Option 2: Test Email Connection
```bash
# Diagnose connection issues
python test_email_connection.py
```

### Common Causes & Solutions

1. **Corporate/ISP Firewall**
   - SMTP ports (587, 465) may be blocked
   - Try from personal network/hotspot
   - Contact IT to whitelist Gmail SMTP

2. **VPN Interference**
   - Disconnect VPN and try again
   - Some VPNs block SMTP traffic

3. **Gmail Configuration**
   - Enable 2-factor authentication
   - Generate app password: https://myaccount.google.com/apppasswords
   - Use app password as `EMAIL_PASSWORD`

4. **Alternative SMTP Settings**
   Add to your `.env` file:
   ```
   # Try Gmail SSL instead of TLS
   SMTP_PORT=465
   SMTP_USE_SSL=true
   ```

### Alternative Email Providers

If Gmail doesn't work, try these in your `.env`:

**Office 365:**
```
SMTP_SERVER=smtp.office365.com
SMTP_PORT=587
SMTP_USE_SSL=false
```

**Yahoo Mail:**
```
SMTP_SERVER=smtp.mail.yahoo.com
SMTP_PORT=587
SMTP_USE_SSL=false
```

### Workarounds

1. **Manual Email Distribution**
   - Use `send_daily_signals_local.py` to generate HTML
   - Copy HTML content to your email client
   - Send manually to recipients

2. **Cloud Upload**
   - Generate HTML files
   - Upload to Google Drive/Dropbox
   - Share link with team

3. **Scheduling with File Generation**
   ```bash
   # Add to crontab for daily generation
   0 6 * * 1-5 cd /path/to/risk-tool && python send_daily_signals_local.py
   ```

### Network Testing

Test if SMTP ports are accessible:
```bash
# Test port 587
telnet smtp.gmail.com 587

# Test port 465
telnet smtp.gmail.com 465
```

If these fail, SMTP is blocked by your network.

### Production Deployment

For production servers:
1. Use dedicated email service (SendGrid, AWS SES)
2. Configure firewall to allow SMTP
3. Use internal SMTP relay
4. Set up email queue for reliability
