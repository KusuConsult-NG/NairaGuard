#!/usr/bin/env python3
"""
Cloud Storage Integration for Naira Note Dataset
Uploads and manages dataset in cloud storage (S3, Azure, GCP)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import hashlib
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage
import requests
from botocore.exceptions import ClientError, NoCredentialsError

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudStorageManager:
    """Manages dataset storage in cloud platforms"""
    
    def __init__(self, provider: str = "aws", bucket_name: str = None, 
                 region: str = "us-east-1", credentials_path: str = None):
        self.provider = provider.lower()
        self.bucket_name = bucket_name
        self.region = region
        self.credentials_path = credentials_path
        
        # Initialize cloud client
        self.client = self._initialize_client()
        
        # Dataset structure
        self.dataset_structure = {
            "raw": "datasets/raw/",
            "processed": "datasets/processed/",
            "metadata": "datasets/metadata/",
            "models": "models/",
            "reports": "reports/"
        }
    
    def _initialize_client(self):
        """Initialize cloud storage client based on provider"""
        try:
            if self.provider == "aws" or self.provider == "s3":
                if self.credentials_path:
                    session = boto3.Session(profile_name=self.credentials_path)
                    return session.client('s3', region_name=self.region)
                else:
                    return boto3.client('s3', region_name=self.region)
            
            elif self.provider == "azure":
                if self.credentials_path:
                    return BlobServiceClient.from_connection_string(
                        self._load_azure_credentials()
                    )
                else:
                    return BlobServiceClient.from_connection_string(
                        os.getenv('AZURE_STORAGE_CONNECTION_STRING')
                    )
            
            elif self.provider == "gcp" or self.provider == "google":
                if self.credentials_path:
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
                return storage.Client()
            
            else:
                raise ValueError(f"Unsupported cloud provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} client: {str(e)}")
            raise
    
    def _load_azure_credentials(self) -> str:
        """Load Azure credentials from file"""
        try:
            with open(self.credentials_path, 'r') as f:
                credentials = json.load(f)
                return credentials.get('connection_string', '')
        except Exception as e:
            logger.error(f"Failed to load Azure credentials: {str(e)}")
            raise
    
    def create_bucket(self, bucket_name: str = None) -> bool:
        """Create cloud storage bucket"""
        bucket_name = bucket_name or self.bucket_name
        if not bucket_name:
            raise ValueError("Bucket name must be provided")
        
        try:
            if self.provider in ["aws", "s3"]:
                self.client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
                logger.info(f"Created S3 bucket: {bucket_name}")
            
            elif self.provider == "azure":
                # Azure containers are created automatically when first blob is uploaded
                logger.info(f"Azure container will be created automatically: {bucket_name}")
            
            elif self.provider in ["gcp", "google"]:
                bucket = self.client.bucket(bucket_name)
                bucket.location = self.region
                bucket.create()
                logger.info(f"Created GCP bucket: {bucket_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create bucket {bucket_name}: {str(e)}")
            return False
    
    def upload_dataset(self, local_path: str, remote_prefix: str = "datasets/") -> Dict[str, int]:
        """Upload entire dataset to cloud storage"""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local path not found: {local_path}")
        
        logger.info(f"Uploading dataset from {local_path} to {self.provider}://{self.bucket_name}/{remote_prefix}")
        
        upload_stats = {
            "total_files": 0,
            "successful_uploads": 0,
            "failed_uploads": 0,
            "total_size": 0
        }
        
        # Find all files to upload
        files_to_upload = []
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                files_to_upload.append(file_path)
        
        upload_stats["total_files"] = len(files_to_upload)
        
        # Upload files
        for file_path in files_to_upload:
            try:
                # Calculate relative path
                relative_path = file_path.relative_to(local_path)
                remote_key = f"{remote_prefix}{relative_path}".replace("\\", "/")
                
                # Upload file
                if self._upload_file(file_path, remote_key):
                    upload_stats["successful_uploads"] += 1
                    upload_stats["total_size"] += file_path.stat().st_size
                    logger.info(f"✓ Uploaded: {remote_key}")
                else:
                    upload_stats["failed_uploads"] += 1
                    logger.error(f"✗ Failed: {remote_key}")
                
            except Exception as e:
                upload_stats["failed_uploads"] += 1
                logger.error(f"✗ Error uploading {file_path}: {str(e)}")
        
        logger.info(f"Upload complete: {upload_stats['successful_uploads']}/{upload_stats['total_files']} files")
        return upload_stats
    
    def _upload_file(self, local_path: Path, remote_key: str) -> bool:
        """Upload a single file to cloud storage"""
        try:
            if self.provider in ["aws", "s3"]:
                self.client.upload_file(
                    str(local_path),
                    self.bucket_name,
                    remote_key,
                    ExtraArgs={'ServerSideEncryption': 'AES256'}
                )
            
            elif self.provider == "azure":
                blob_client = self.client.get_blob_client(
                    container=self.bucket_name,
                    blob=remote_key
                )
                with open(local_path, 'rb') as data:
                    blob_client.upload_blob(data, overwrite=True)
            
            elif self.provider in ["gcp", "google"]:
                bucket = self.client.bucket(self.bucket_name)
                blob = bucket.blob(remote_key)
                blob.upload_from_filename(str(local_path))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to {remote_key}: {str(e)}")
            return False
    
    def download_dataset(self, remote_prefix: str = "datasets/", 
                        local_path: str = "datasets/downloaded") -> Dict[str, int]:
        """Download entire dataset from cloud storage"""
        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading dataset from {self.provider}://{self.bucket_name}/{remote_prefix}")
        
        download_stats = {
            "total_files": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "total_size": 0
        }
        
        # List files in remote prefix
        remote_files = self._list_remote_files(remote_prefix)
        download_stats["total_files"] = len(remote_files)
        
        # Download files
        for remote_key in remote_files:
            try:
                # Calculate local path
                relative_path = remote_key.replace(remote_prefix, "").lstrip("/")
                local_file_path = local_path / relative_path
                
                # Create local directory if needed
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                if self._download_file(remote_key, local_file_path):
                    download_stats["successful_downloads"] += 1
                    download_stats["total_size"] += local_file_path.stat().st_size
                    logger.info(f"✓ Downloaded: {remote_key}")
                else:
                    download_stats["failed_downloads"] += 1
                    logger.error(f"✗ Failed: {remote_key}")
                
            except Exception as e:
                download_stats["failed_downloads"] += 1
                logger.error(f"✗ Error downloading {remote_key}: {str(e)}")
        
        logger.info(f"Download complete: {download_stats['successful_downloads']}/{download_stats['total_files']} files")
        return download_stats
    
    def _list_remote_files(self, prefix: str) -> List[str]:
        """List files in cloud storage with given prefix"""
        files = []
        
        try:
            if self.provider in ["aws", "s3"]:
                paginator = self.client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
                
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            files.append(obj['Key'])
            
            elif self.provider == "azure":
                container_client = self.client.get_container_client(self.bucket_name)
                blobs = container_client.list_blobs(name_starts_with=prefix)
                files = [blob.name for blob in blobs]
            
            elif self.provider in ["gcp", "google"]:
                bucket = self.client.bucket(self.bucket_name)
                blobs = bucket.list_blobs(prefix=prefix)
                files = [blob.name for blob in blobs]
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files with prefix {prefix}: {str(e)}")
            return []
    
    def _download_file(self, remote_key: str, local_path: Path) -> bool:
        """Download a single file from cloud storage"""
        try:
            if self.provider in ["aws", "s3"]:
                self.client.download_file(self.bucket_name, remote_key, str(local_path))
            
            elif self.provider == "azure":
                blob_client = self.client.get_blob_client(
                    container=self.bucket_name,
                    blob=remote_key
                )
                with open(local_path, 'wb') as data:
                    data.write(blob_client.download_blob().readall())
            
            elif self.provider in ["gcp", "google"]:
                bucket = self.client.bucket(self.bucket_name)
                blob = bucket.blob(remote_key)
                blob.download_to_filename(str(local_path))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {remote_key} to {local_path}: {str(e)}")
            return False
    
    def sync_dataset(self, local_path: str, remote_prefix: str = "datasets/") -> Dict[str, int]:
        """Sync local dataset with cloud storage (upload only new/changed files)"""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local path not found: {local_path}")
        
        logger.info(f"Syncing dataset from {local_path} to {self.provider}://{self.bucket_name}/{remote_prefix}")
        
        sync_stats = {
            "total_files": 0,
            "uploaded_files": 0,
            "skipped_files": 0,
            "failed_files": 0
        }
        
        # Get remote file hashes
        remote_hashes = self._get_remote_file_hashes(remote_prefix)
        
        # Process local files
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                sync_stats["total_files"] += 1
                
                # Calculate relative path and remote key
                relative_path = file_path.relative_to(local_path)
                remote_key = f"{remote_prefix}{relative_path}".replace("\\", "/")
                
                # Calculate local file hash
                local_hash = self._calculate_file_hash(file_path)
                
                # Check if file needs to be uploaded
                if remote_key not in remote_hashes or remote_hashes[remote_key] != local_hash:
                    if self._upload_file(file_path, remote_key):
                        sync_stats["uploaded_files"] += 1
                        logger.info(f"✓ Synced: {remote_key}")
                    else:
                        sync_stats["failed_files"] += 1
                        logger.error(f"✗ Failed: {remote_key}")
                else:
                    sync_stats["skipped_files"] += 1
                    logger.debug(f"- Skipped: {remote_key}")
        
        logger.info(f"Sync complete: {sync_stats['uploaded_files']} uploaded, {sync_stats['skipped_files']} skipped")
        return sync_stats
    
    def _get_remote_file_hashes(self, prefix: str) -> Dict[str, str]:
        """Get MD5 hashes of remote files"""
        hashes = {}
        
        try:
            if self.provider in ["aws", "s3"]:
                paginator = self.client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
                
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            # S3 provides ETag which is MD5 hash
                            hashes[obj['Key']] = obj['ETag'].strip('"')
            
            elif self.provider == "azure":
                container_client = self.client.get_container_client(self.bucket_name)
                blobs = container_client.list_blobs(name_starts_with=prefix)
                
                for blob in blobs:
                    hashes[blob.name] = blob.properties.content_md5.hex() if blob.properties.content_md5 else ""
            
            elif self.provider in ["gcp", "google"]:
                bucket = self.client.bucket(self.bucket_name)
                blobs = bucket.list_blobs(prefix=prefix)
                
                for blob in blobs:
                    hashes[blob.name] = blob.md5_hash if blob.md5_hash else ""
            
            return hashes
            
        except Exception as e:
            logger.error(f"Failed to get remote file hashes: {str(e)}")
            return {}
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of local file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def generate_dataset_urls(self, remote_prefix: str = "datasets/") -> Dict[str, str]:
        """Generate public URLs for dataset files"""
        urls = {}
        
        try:
            if self.provider in ["aws", "s3"]:
                # Generate presigned URLs (valid for 7 days)
                remote_files = self._list_remote_files(remote_prefix)
                
                for remote_key in remote_files:
                    url = self.client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': self.bucket_name, 'Key': remote_key},
                        ExpiresIn=604800  # 7 days
                    )
                    urls[remote_key] = url
            
            elif self.provider == "azure":
                # Azure public URLs
                base_url = f"https://{self.bucket_name}.blob.core.windows.net"
                remote_files = self._list_remote_files(remote_prefix)
                
                for remote_key in remote_files:
                    urls[remote_key] = f"{base_url}/{remote_key}"
            
            elif self.provider in ["gcp", "google"]:
                # GCP public URLs
                base_url = f"https://storage.googleapis.com/{self.bucket_name}"
                remote_files = self._list_remote_files(remote_prefix)
                
                for remote_key in remote_files:
                    urls[remote_key] = f"{base_url}/{remote_key}"
            
            return urls
            
        except Exception as e:
            logger.error(f"Failed to generate dataset URLs: {str(e)}")
            return {}

def main():
    parser = argparse.ArgumentParser(description='Manage naira dataset in cloud storage')
    parser.add_argument('--provider', type=str, choices=['aws', 's3', 'azure', 'gcp', 'google'],
                       default='aws', help='Cloud storage provider')
    parser.add_argument('--bucket', type=str, required=True, help='Bucket/container name')
    parser.add_argument('--region', type=str, default='us-east-1', help='Region for bucket')
    parser.add_argument('--credentials', type=str, help='Path to credentials file')
    parser.add_argument('--action', type=str, choices=['upload', 'download', 'sync', 'create-bucket'],
                       required=True, help='Action to perform')
    parser.add_argument('--local-path', type=str, help='Local dataset path')
    parser.add_argument('--remote-prefix', type=str, default='datasets/', 
                       help='Remote prefix for dataset')
    
    args = parser.parse_args()
    
    # Initialize cloud storage manager
    try:
        manager = CloudStorageManager(
            provider=args.provider,
            bucket_name=args.bucket,
            region=args.region,
            credentials_path=args.credentials
        )
    except Exception as e:
        logger.error(f"Failed to initialize cloud storage manager: {str(e)}")
        sys.exit(1)
    
    # Perform action
    try:
        if args.action == 'create-bucket':
            success = manager.create_bucket()
            if success:
                logger.info("Bucket created successfully")
            else:
                logger.error("Failed to create bucket")
                sys.exit(1)
        
        elif args.action == 'upload':
            if not args.local_path:
                logger.error("Local path required for upload action")
                sys.exit(1)
            stats = manager.upload_dataset(args.local_path, args.remote_prefix)
            logger.info(f"Upload stats: {stats}")
        
        elif args.action == 'download':
            stats = manager.download_dataset(args.remote_prefix, args.local_path)
            logger.info(f"Download stats: {stats}")
        
        elif args.action == 'sync':
            if not args.local_path:
                logger.error("Local path required for sync action")
                sys.exit(1)
            stats = manager.sync_dataset(args.local_path, args.remote_prefix)
            logger.info(f"Sync stats: {stats}")
    
    except Exception as e:
        logger.error(f"Action failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
