# Utility Modules

Shared utility functions and services.

## Files

- **`logger.py`** - Dual logging system (local files + S3 for dashboard)
- **`user_manager.py`** - User ID generation and API key management

## Features

### Logging
- Local file logging for debugging (`logs/logs.txt`)
- S3 structured logging for dashboard analytics

### User Management
- Sequential user IDs (`user-1`, `user-2`, etc.)
- API key generation and validation
