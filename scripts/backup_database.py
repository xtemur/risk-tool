#!/usr/bin/env python3
"""
Database Backup Automation Script

This script creates automated backups of the SQLite database with:
- Daily and weekly backup retention
- Compression to save space
- Backup verification
- Old backup cleanup
- Optional remote backup sync

Usage:
    python scripts/backup_database.py [--remote-sync] [--verify-only]

Options:
    --remote-sync: Upload backups to remote server (requires rsync config)
    --verify-only: Only verify existing backups without creating new ones
    --keep-days: Number of daily backups to keep (default: 7)
    --keep-weeks: Number of weekly backups to keep (default: 4)
"""

import os
import sys
import sqlite3
import shutil
import gzip
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
os.chdir(project_root)

from src.utils import load_config


class DatabaseBackup:
    """Automated database backup system."""

    def __init__(self, config_path='configs/main_config.yaml'):
        """Initialize backup system."""
        self.config = load_config(config_path)
        self.setup_logging()

        # Backup configuration
        self.db_path = self.config['paths']['db_path']
        self.backup_dir = Path('data/backups')
        self.backup_dir.mkdir(exist_ok=True)

        # Retention policy
        self.keep_daily = 7  # Keep 7 daily backups
        self.keep_weekly = 4  # Keep 4 weekly backups
        self.keep_monthly = 12  # Keep 12 monthly backups

    def setup_logging(self):
        """Setup logging for backup operations."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        log_filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = log_dir / log_filename

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Database backup started - Log: {log_path}")

    def calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def verify_database_integrity(self):
        """Verify database integrity before backup."""
        try:
            self.logger.info("Verifying database integrity...")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()

            if result[0] != 'ok':
                raise Exception(f"Database integrity check failed: {result[0]}")

            # Check if database is not empty
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]

            if table_count == 0:
                raise Exception("Database appears to be empty")

            # Get database stats
            cursor.execute("SELECT COUNT(*) FROM accounts WHERE is_active = 1")
            active_accounts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM trades")
            total_trades = cursor.fetchone()[0]

            conn.close()

            self.logger.info(f"Database integrity verified - {table_count} tables, {active_accounts} active accounts, {total_trades} trades")
            return True

        except Exception as e:
            self.logger.error(f"Database integrity verification failed: {str(e)}")
            return False

    def create_backup(self):
        """Create a new database backup."""
        try:
            # Verify database integrity first
            if not self.verify_database_integrity():
                return False

            # Generate backup filename
            now = datetime.now()
            backup_name = f"risk_tool_{now.strftime('%Y%m%d_%H%M%S')}.db"
            backup_path = self.backup_dir / backup_name

            self.logger.info(f"Creating backup: {backup_path}")

            # Create backup using SQLite backup API (more reliable than file copy)
            source_conn = sqlite3.connect(self.db_path)
            backup_conn = sqlite3.connect(str(backup_path))

            source_conn.backup(backup_conn)

            source_conn.close()
            backup_conn.close()

            # Verify backup was created successfully
            if not backup_path.exists():
                raise Exception("Backup file was not created")

            # Calculate and store checksum
            backup_hash = self.calculate_file_hash(backup_path)
            checksum_file = backup_path.with_suffix('.db.sha256')
            with open(checksum_file, 'w') as f:
                f.write(f"{backup_hash}  {backup_name}\n")

            # Get file sizes
            original_size = os.path.getsize(self.db_path)
            backup_size = os.path.getsize(backup_path)

            self.logger.info(f"Backup created successfully:")
            self.logger.info(f"  Original size: {original_size:,} bytes")
            self.logger.info(f"  Backup size: {backup_size:,} bytes")
            self.logger.info(f"  Checksum: {backup_hash}")

            # Compress backup to save space
            compressed_path = self.compress_backup(backup_path)
            if compressed_path:
                # Remove uncompressed backup
                backup_path.unlink()
                checksum_file.unlink()

                # Create new checksum for compressed file
                compressed_hash = self.calculate_file_hash(compressed_path)
                compressed_checksum = compressed_path.with_suffix('.gz.sha256')
                with open(compressed_checksum, 'w') as f:
                    f.write(f"{compressed_hash}  {compressed_path.name}\n")

                compressed_size = os.path.getsize(compressed_path)
                compression_ratio = (1 - compressed_size / backup_size) * 100

                self.logger.info(f"Backup compressed:")
                self.logger.info(f"  Compressed size: {compressed_size:,} bytes")
                self.logger.info(f"  Compression ratio: {compression_ratio:.1f}%")

            return True

        except Exception as e:
            self.logger.error(f"Backup creation failed: {str(e)}")
            return False

    def compress_backup(self, backup_path):
        """Compress backup file using gzip."""
        try:
            compressed_path = backup_path.with_suffix('.db.gz')

            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            return compressed_path

        except Exception as e:
            self.logger.error(f"Backup compression failed: {str(e)}")
            return None

    def verify_backup(self, backup_path):
        """Verify backup integrity."""
        try:
            # Check if backup file exists
            if not backup_path.exists():
                return False, "Backup file does not exist"

            # Check if it's compressed
            if backup_path.suffix == '.gz':
                # Decompress temporarily for verification
                temp_path = backup_path.with_suffix('')
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(temp_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Verify decompressed file
                conn = sqlite3.connect(str(temp_path))
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                conn.close()

                # Clean up temp file
                temp_path.unlink()

            else:
                # Verify uncompressed backup
                conn = sqlite3.connect(str(backup_path))
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                conn.close()

            if result[0] == 'ok':
                return True, "Backup verification successful"
            else:
                return False, f"Backup verification failed: {result[0]}"

        except Exception as e:
            return False, f"Backup verification error: {str(e)}"

    def cleanup_old_backups(self, keep_daily=None, keep_weekly=None, keep_monthly=None):
        """Clean up old backups based on retention policy."""
        keep_daily = keep_daily or self.keep_daily
        keep_weekly = keep_weekly or self.keep_weekly
        keep_monthly = keep_monthly or self.keep_monthly

        try:
            # Get all backup files
            backup_files = list(self.backup_dir.glob('risk_tool_*.db*'))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            if not backup_files:
                self.logger.info("No backup files found for cleanup")
                return

            now = datetime.now()
            files_to_keep = set()
            files_to_delete = []

            # Categorize backups by age
            for backup_file in backup_files:
                # Extract date from filename
                try:
                    date_str = backup_file.stem.split('_')[2]  # risk_tool_YYYYMMDD_HHMMSS
                    backup_date = datetime.strptime(date_str, '%Y%m%d')
                    age_days = (now - backup_date).days

                    # Keep daily backups for specified days
                    if age_days < keep_daily:
                        files_to_keep.add(backup_file)
                        continue

                    # Keep weekly backups (one per week)
                    if age_days < keep_weekly * 7 and backup_date.weekday() == 6:  # Sunday
                        files_to_keep.add(backup_file)
                        continue

                    # Keep monthly backups (first of month)
                    if age_days < keep_monthly * 30 and backup_date.day == 1:
                        files_to_keep.add(backup_file)
                        continue

                    # Mark for deletion
                    files_to_delete.append(backup_file)

                except (IndexError, ValueError):
                    # Skip files that don't match expected format
                    continue

            # Delete old backups
            deleted_count = 0
            for backup_file in files_to_delete:
                try:
                    # Also delete associated checksum files
                    checksum_file = backup_file.with_suffix(backup_file.suffix + '.sha256')
                    if checksum_file.exists():
                        checksum_file.unlink()

                    backup_file.unlink()
                    deleted_count += 1
                    self.logger.info(f"Deleted old backup: {backup_file.name}")

                except Exception as e:
                    self.logger.error(f"Failed to delete backup {backup_file.name}: {str(e)}")

            kept_count = len(backup_files) - deleted_count
            self.logger.info(f"Backup cleanup completed: {kept_count} kept, {deleted_count} deleted")

        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {str(e)}")

    def sync_to_remote(self, remote_path=None):
        """Sync backups to remote server using rsync."""
        if not remote_path:
            remote_path = os.getenv('BACKUP_REMOTE_PATH')

        if not remote_path:
            self.logger.warning("No remote backup path configured")
            return False

        try:
            # Use rsync to sync backups
            rsync_cmd = [
                'rsync',
                '-av',
                '--delete',
                str(self.backup_dir) + '/',
                remote_path
            ]

            self.logger.info(f"Syncing backups to remote: {remote_path}")
            result = subprocess.run(rsync_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info("Remote backup sync completed successfully")
                return True
            else:
                self.logger.error(f"Remote backup sync failed: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Remote backup sync error: {str(e)}")
            return False

    def verify_all_backups(self):
        """Verify integrity of all existing backups."""
        backup_files = list(self.backup_dir.glob('risk_tool_*.db*'))
        backup_files.sort()

        if not backup_files:
            self.logger.info("No backup files found to verify")
            return True

        self.logger.info(f"Verifying {len(backup_files)} backup files...")

        verified_count = 0
        failed_count = 0

        for backup_file in backup_files:
            success, message = self.verify_backup(backup_file)
            if success:
                verified_count += 1
                self.logger.info(f"✅ {backup_file.name}: {message}")
            else:
                failed_count += 1
                self.logger.error(f"❌ {backup_file.name}: {message}")

        self.logger.info(f"Backup verification completed: {verified_count} successful, {failed_count} failed")
        return failed_count == 0

    def get_backup_status(self):
        """Get current backup status and statistics."""
        backup_files = list(self.backup_dir.glob('risk_tool_*.db*'))

        if not backup_files:
            return {
                'total_backups': 0,
                'total_size': 0,
                'oldest_backup': None,
                'newest_backup': None
            }

        # Calculate total size
        total_size = sum(f.stat().st_size for f in backup_files)

        # Get oldest and newest
        backup_files.sort(key=lambda x: x.stat().st_mtime)
        oldest = backup_files[0]
        newest = backup_files[-1]

        return {
            'total_backups': len(backup_files),
            'total_size': total_size,
            'oldest_backup': oldest.name,
            'newest_backup': newest.name,
            'oldest_date': datetime.fromtimestamp(oldest.stat().st_mtime),
            'newest_date': datetime.fromtimestamp(newest.stat().st_mtime)
        }


def main():
    """Main function for database backup."""
    parser = argparse.ArgumentParser(description='Database Backup Automation')
    parser.add_argument('--remote-sync', action='store_true', help='Sync backups to remote server')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing backups')
    parser.add_argument('--keep-days', type=int, default=7, help='Number of daily backups to keep')
    parser.add_argument('--keep-weeks', type=int, default=4, help='Number of weekly backups to keep')
    parser.add_argument('--keep-months', type=int, default=12, help='Number of monthly backups to keep')
    parser.add_argument('--config', default='configs/main_config.yaml', help='Configuration file path')

    args = parser.parse_args()

    # Initialize backup system
    backup_system = DatabaseBackup(args.config)

    overall_success = True

    if args.verify_only:
        # Only verify existing backups
        success = backup_system.verify_all_backups()
        if not success:
            overall_success = False
    else:
        # Full backup process

        # Create new backup
        success = backup_system.create_backup()
        if not success:
            overall_success = False

        # Clean up old backups
        backup_system.cleanup_old_backups(
            keep_daily=args.keep_days,
            keep_weekly=args.keep_weeks,
            keep_monthly=args.keep_months
        )

        # Sync to remote if requested
        if args.remote_sync:
            success = backup_system.sync_to_remote()
            if not success:
                overall_success = False

    # Print backup status
    status = backup_system.get_backup_status()
    backup_system.logger.info("=" * 50)
    backup_system.logger.info("BACKUP STATUS")
    backup_system.logger.info("=" * 50)
    backup_system.logger.info(f"Total backups: {status['total_backups']}")
    backup_system.logger.info(f"Total size: {status['total_size']:,} bytes ({status['total_size'] / 1024 / 1024:.1f} MB)")
    if status['total_backups'] > 0:
        backup_system.logger.info(f"Oldest backup: {status['oldest_backup']} ({status['oldest_date'].strftime('%Y-%m-%d %H:%M')})")
        backup_system.logger.info(f"Newest backup: {status['newest_backup']} ({status['newest_date'].strftime('%Y-%m-%d %H:%M')})")

    sys.exit(0 if overall_success else 1)


if __name__ == '__main__':
    main()
