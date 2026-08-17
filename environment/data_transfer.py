import os
import paramiko
import logging
from pathlib import Path
import yaml

class SFTPTransfer:
    def __init__(self, config_path='configs/lab_config.yml'):
        # Setup logging
        self.logger = logging.getLogger('data_transfer')
        
        # Load config
        self.config = self._load_config(config_path)
        
        # Initialize SSH client
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
    def _load_config(self, config_path):
        """Load SFTP configuration"""
        if not os.path.exists(config_path):
            config = {
                'sftp': {
                    'hostname': os.environ.get('PRAXIS_GPU_HOST', 'your.gpu.server'),
                    'username': os.environ.get('PRAXIS_GPU_USER', 'user'),
                    'remote_path': os.environ.get('PRAXIS_GPU_REMOTE_PATH', '/path/to/agent/data/plate_data'),
                    'key_filename': os.environ.get('PRAXIS_SSH_KEY', '~/.ssh/id_rsa'),
                    'port': int(os.environ.get('PRAXIS_GPU_PORT', 22))
                }
            }
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            raise FileNotFoundError(f"Created default config at {config_path}. Please edit before running.")
            
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def connect(self):
        """Establish SFTP connection"""
        try:
            cfg = self.config['sftp']
            self.ssh.connect(
                hostname=cfg['hostname'],
                username=cfg['username'],
                key_filename=os.path.expanduser(cfg['key_filename']),
                port=cfg['port']
            )
            self.sftp = self.ssh.open_sftp()
            self.logger.info(f"Connected to {cfg['hostname']}")
            return True
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
        
    def transfer_file(self, local_path):
        """Transfer file to remote server"""
        try:
            if not self.sftp:
                self.logger.error("Not connected to SFTP server")
                return False
            
            # Use posix-style paths (forward slashes) for remote server
            remote_path = self.config['sftp']['remote_path'] + '/' + os.path.basename(local_path)
            # Ensure all slashes are forward slashes
            remote_path = remote_path.replace('\\', '/')
            
            self.logger.info(f"Attempting transfer: {local_path} to {remote_path}")
            
            self.sftp.put(local_path, remote_path)
            self.logger.info(f"Transferred {local_path} to {remote_path}")
            return True
        except Exception as e:
            self.logger.error(f"Transfer failed: {e}")
            return False

            
    def close(self):
        """Close SFTP connection"""
        if hasattr(self, 'sftp'):
            self.sftp.close()
        self.ssh.close()