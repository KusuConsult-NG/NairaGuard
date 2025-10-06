#!/usr/bin/env python3
"""
Dataset Export and Import Script
Exports dataset in various formats and imports from external sources
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import csv
import pickle
import zipfile
import tarfile
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
import cv2

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetExporter:
    """Exports naira note dataset in various formats"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.export_results = {
            "export_date": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_path),
            "exports_created": [],
            "total_files_exported": 0,
            "export_size_mb": 0
        }
    
    def export_dataset(self, output_path: str, format: str = "zip", 
                      include_metadata: bool = True, compress: bool = True) -> Dict:
        """Export dataset in specified format"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting dataset to {output_path} in {format} format")
        
        if format.lower() == "zip":
            self._export_to_zip(output_path, include_metadata, compress)
        elif format.lower() == "tar":
            self._export_to_tar(output_path, include_metadata, compress)
        elif format.lower() == "csv":
            self._export_to_csv(output_path, include_metadata)
        elif format.lower() == "json":
            self._export_to_json(output_path, include_metadata)
        elif format.lower() == "pickle":
            self._export_to_pickle(output_path, include_metadata)
        elif format.lower() == "directory":
            self._export_to_directory(output_path, include_metadata)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        # Generate export report
        self._generate_export_report()
        
        logger.info("Dataset export complete")
        return self.export_results
    
    def _export_to_zip(self, output_path: Path, include_metadata: bool, compress: bool):
        """Export dataset to ZIP archive"""
        compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
        
        with zipfile.ZipFile(output_path, 'w', compression) as zipf:
            # Add all image files
            image_files = list(self.dataset_path.rglob("*.jpg"))
            for image_file in image_files:
                arcname = image_file.relative_to(self.dataset_path)
                zipf.write(image_file, arcname)
                self.export_results["total_files_exported"] += 1
            
            # Add metadata files
            if include_metadata:
                metadata_files = list(self.dataset_path.rglob("*.json"))
                for metadata_file in metadata_files:
                    arcname = metadata_file.relative_to(self.dataset_path)
                    zipf.write(metadata_file, arcname)
        
        self.export_results["exports_created"].append(f"zip: {output_path}")
        self.export_results["export_size_mb"] = output_path.stat().st_size / (1024 * 1024)
    
    def _export_to_tar(self, output_path: Path, include_metadata: bool, compress: bool):
        """Export dataset to TAR archive"""
        mode = 'w:gz' if compress else 'w'
        
        with tarfile.open(output_path, mode) as tarf:
            # Add all image files
            image_files = list(self.dataset_path.rglob("*.jpg"))
            for image_file in image_files:
                arcname = image_file.relative_to(self.dataset_path)
                tarf.add(image_file, arcname)
                self.export_results["total_files_exported"] += 1
            
            # Add metadata files
            if include_metadata:
                metadata_files = list(self.dataset_path.rglob("*.json"))
                for metadata_file in metadata_files:
                    arcname = metadata_file.relative_to(self.dataset_path)
                    tarf.add(metadata_file, arcname)
        
        self.export_results["exports_created"].append(f"tar: {output_path}")
        self.export_results["export_size_mb"] = output_path.stat().st_size / (1024 * 1024)
    
    def _export_to_csv(self, output_path: Path, include_metadata: bool):
        """Export dataset metadata to CSV"""
        data = []
        
        # Collect image information
        image_files = list(self.dataset_path.rglob("*.jpg"))
        for image_file in image_files:
            try:
                # Extract information from filename
                filename = image_file.stem
                parts = filename.split('_')
                
                if len(parts) >= 3:
                    authenticity = parts[0]
                    denomination = parts[1]
                    condition = parts[2]
                    identifier = '_'.join(parts[3:]) if len(parts) > 3 else ""
                else:
                    authenticity = "unknown"
                    denomination = "unknown"
                    condition = "unknown"
                    identifier = filename
                
                # Get file information
                file_size = image_file.stat().st_size
                
                # Get image information
                image = cv2.imread(str(image_file))
                if image is not None:
                    height, width = image.shape[:2]
                    quality_score = cv2.Laplacian(image, cv2.CV_64F).var()
                else:
                    height, width, quality_score = 0, 0, 0
                
                data.append({
                    "filename": image_file.name,
                    "path": str(image_file.relative_to(self.dataset_path)),
                    "authenticity": authenticity,
                    "denomination": denomination,
                    "condition": condition,
                    "identifier": identifier,
                    "file_size_bytes": file_size,
                    "width": width,
                    "height": height,
                    "quality_score": quality_score,
                    "modified_date": datetime.fromtimestamp(image_file.stat().st_mtime).isoformat()
                })
                
            except Exception as e:
                logger.warning(f"Error processing {image_file}: {str(e)}")
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        self.export_results["exports_created"].append(f"csv: {output_path}")
        self.export_results["total_files_exported"] = len(data)
    
    def _export_to_json(self, output_path: Path, include_metadata: bool):
        """Export dataset metadata to JSON"""
        data = {
            "export_info": {
                "export_date": datetime.now().isoformat(),
                "dataset_path": str(self.dataset_path),
                "total_images": 0,
                "by_authenticity": {},
                "by_denomination": {},
                "by_condition": {}
            },
            "images": []
        }
        
        # Collect image information
        image_files = list(self.dataset_path.rglob("*.jpg"))
        data["export_info"]["total_images"] = len(image_files)
        
        for image_file in image_files:
            try:
                # Extract information from filename
                filename = image_file.stem
                parts = filename.split('_')
                
                if len(parts) >= 3:
                    authenticity = parts[0]
                    denomination = parts[1]
                    condition = parts[2]
                    identifier = '_'.join(parts[3:]) if len(parts) > 3 else ""
                else:
                    authenticity = "unknown"
                    denomination = "unknown"
                    condition = "unknown"
                    identifier = filename
                
                # Update counters
                data["export_info"]["by_authenticity"][authenticity] = data["export_info"]["by_authenticity"].get(authenticity, 0) + 1
                data["export_info"]["by_denomination"][denomination] = data["export_info"]["by_denomination"].get(denomination, 0) + 1
                data["export_info"]["by_condition"][condition] = data["export_info"]["by_condition"].get(condition, 0) + 1
                
                # Get file information
                file_size = image_file.stat().st_size
                
                # Get image information
                image = cv2.imread(str(image_file))
                if image is not None:
                    height, width = image.shape[:2]
                    quality_score = cv2.Laplacian(image, cv2.CV_64F).var()
                else:
                    height, width, quality_score = 0, 0, 0
                
                data["images"].append({
                    "filename": image_file.name,
                    "path": str(image_file.relative_to(self.dataset_path)),
                    "authenticity": authenticity,
                    "denomination": denomination,
                    "condition": condition,
                    "identifier": identifier,
                    "file_size_bytes": file_size,
                    "width": width,
                    "height": height,
                    "quality_score": quality_score,
                    "modified_date": datetime.fromtimestamp(image_file.stat().st_mtime).isoformat()
                })
                
            except Exception as e:
                logger.warning(f"Error processing {image_file}: {str(e)}")
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.export_results["exports_created"].append(f"json: {output_path}")
        self.export_results["total_files_exported"] = len(data["images"])
    
    def _export_to_pickle(self, output_path: Path, include_metadata: bool):
        """Export dataset to pickle format"""
        data = {
            "export_info": {
                "export_date": datetime.now().isoformat(),
                "dataset_path": str(self.dataset_path),
                "total_images": 0
            },
            "images": []
        }
        
        # Collect image information
        image_files = list(self.dataset_path.rglob("*.jpg"))
        data["export_info"]["total_images"] = len(image_files)
        
        for image_file in image_files:
            try:
                # Extract information from filename
                filename = image_file.stem
                parts = filename.split('_')
                
                if len(parts) >= 3:
                    authenticity = parts[0]
                    denomination = parts[1]
                    condition = parts[2]
                    identifier = '_'.join(parts[3:]) if len(parts) > 3 else ""
                else:
                    authenticity = "unknown"
                    denomination = "unknown"
                    condition = "unknown"
                    identifier = filename
                
                # Get file information
                file_size = image_file.stat().st_size
                
                # Get image information
                image = cv2.imread(str(image_file))
                if image is not None:
                    height, width = image.shape[:2]
                    quality_score = cv2.Laplacian(image, cv2.CV_64F).var()
                else:
                    height, width, quality_score = 0, 0, 0
                
                data["images"].append({
                    "filename": image_file.name,
                    "path": str(image_file.relative_to(self.dataset_path)),
                    "authenticity": authenticity,
                    "denomination": denomination,
                    "condition": condition,
                    "identifier": identifier,
                    "file_size_bytes": file_size,
                    "width": width,
                    "height": height,
                    "quality_score": quality_score,
                    "modified_date": datetime.fromtimestamp(image_file.stat().st_mtime).isoformat()
                })
                
            except Exception as e:
                logger.warning(f"Error processing {image_file}: {str(e)}")
        
        # Save to pickle
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        self.export_results["exports_created"].append(f"pickle: {output_path}")
        self.export_results["total_files_exported"] = len(data["images"])
    
    def _export_to_directory(self, output_path: Path, include_metadata: bool):
        """Export dataset to directory structure"""
        # Copy entire dataset
        shutil.copytree(self.dataset_path, output_path, dirs_exist_ok=True)
        
        # Count files
        image_files = list(output_path.rglob("*.jpg"))
        self.export_results["total_files_exported"] = len(image_files)
        self.export_results["export_size_mb"] = sum(f.stat().st_size for f in image_files) / (1024 * 1024)
        
        self.export_results["exports_created"].append(f"directory: {output_path}")
    
    def _generate_export_report(self):
        """Generate export report"""
        report_path = self.dataset_path / "metadata" / "export_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.export_results, f, indent=2)
        
        logger.info(f"Export report saved to {report_path}")

class DatasetImporter:
    """Imports naira note dataset from various sources"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.import_results = {
            "import_date": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_path),
            "imports_created": [],
            "total_files_imported": 0,
            "import_size_mb": 0
        }
    
    def import_dataset(self, source_path: str, format: str = "auto") -> Dict:
        """Import dataset from specified source"""
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")
        
        logger.info(f"Importing dataset from {source_path}")
        
        # Auto-detect format if not specified
        if format == "auto":
            format = self._detect_format(source_path)
        
        if format.lower() == "zip":
            self._import_from_zip(source_path)
        elif format.lower() == "tar":
            self._import_from_tar(source_path)
        elif format.lower() == "csv":
            self._import_from_csv(source_path)
        elif format.lower() == "json":
            self._import_from_json(source_path)
        elif format.lower() == "pickle":
            self._import_from_pickle(source_path)
        elif format.lower() == "directory":
            self._import_from_directory(source_path)
        else:
            raise ValueError(f"Unsupported import format: {format}")
        
        # Generate import report
        self._generate_import_report()
        
        logger.info("Dataset import complete")
        return self.import_results
    
    def _detect_format(self, source_path: Path) -> str:
        """Auto-detect file format"""
        if source_path.is_file():
            if source_path.suffix.lower() == '.zip':
                return 'zip'
            elif source_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz']:
                return 'tar'
            elif source_path.suffix.lower() == '.csv':
                return 'csv'
            elif source_path.suffix.lower() == '.json':
                return 'json'
            elif source_path.suffix.lower() == '.pkl':
                return 'pickle'
        elif source_path.is_dir():
            return 'directory'
        
        raise ValueError(f"Could not detect format for: {source_path}")
    
    def _import_from_zip(self, source_path: Path):
        """Import dataset from ZIP archive"""
        with zipfile.ZipFile(source_path, 'r') as zipf:
            zipf.extractall(self.dataset_path)
            
            # Count extracted files
            extracted_files = list(self.dataset_path.rglob("*"))
            self.import_results["total_files_imported"] = len(extracted_files)
            self.import_results["import_size_mb"] = sum(f.stat().st_size for f in extracted_files if f.is_file()) / (1024 * 1024)
        
        self.import_results["imports_created"].append(f"zip: {source_path}")
    
    def _import_from_tar(self, source_path: Path):
        """Import dataset from TAR archive"""
        with tarfile.open(source_path, 'r:*') as tarf:
            tarf.extractall(self.dataset_path)
            
            # Count extracted files
            extracted_files = list(self.dataset_path.rglob("*"))
            self.import_results["total_files_imported"] = len(extracted_files)
            self.import_results["import_size_mb"] = sum(f.stat().st_size for f in extracted_files if f.is_file()) / (1024 * 1024)
        
        self.import_results["imports_created"].append(f"tar: {source_path}")
    
    def _import_from_csv(self, source_path: Path):
        """Import dataset metadata from CSV"""
        df = pd.read_csv(source_path)
        
        # Create directory structure
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Process each row
        for _, row in df.iterrows():
            # Create directory structure
            file_path = self.dataset_path / row['path']
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create placeholder file (since CSV doesn't contain actual images)
            placeholder_path = file_path.with_suffix('.txt')
            with open(placeholder_path, 'w') as f:
                f.write(f"Placeholder for {row['filename']}\n")
                f.write(f"Authenticity: {row['authenticity']}\n")
                f.write(f"Denomination: {row['denomination']}\n")
                f.write(f"Condition: {row['condition']}\n")
                f.write(f"Quality Score: {row['quality_score']}\n")
        
        self.import_results["total_files_imported"] = len(df)
        self.import_results["imports_created"].append(f"csv: {source_path}")
    
    def _import_from_json(self, source_path: Path):
        """Import dataset metadata from JSON"""
        with open(source_path, 'r') as f:
            data = json.load(f)
        
        # Create directory structure
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Process each image entry
        for image_info in data['images']:
            # Create directory structure
            file_path = self.dataset_path / image_info['path']
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create placeholder file
            placeholder_path = file_path.with_suffix('.txt')
            with open(placeholder_path, 'w') as f:
                f.write(f"Placeholder for {image_info['filename']}\n")
                f.write(f"Authenticity: {image_info['authenticity']}\n")
                f.write(f"Denomination: {image_info['denomination']}\n")
                f.write(f"Condition: {image_info['condition']}\n")
                f.write(f"Quality Score: {image_info['quality_score']}\n")
        
        self.import_results["total_files_imported"] = len(data['images'])
        self.import_results["imports_created"].append(f"json: {source_path}")
    
    def _import_from_pickle(self, source_path: Path):
        """Import dataset metadata from pickle"""
        with open(source_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create directory structure
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Process each image entry
        for image_info in data['images']:
            # Create directory structure
            file_path = self.dataset_path / image_info['path']
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create placeholder file
            placeholder_path = file_path.with_suffix('.txt')
            with open(placeholder_path, 'w') as f:
                f.write(f"Placeholder for {image_info['filename']}\n")
                f.write(f"Authenticity: {image_info['authenticity']}\n")
                f.write(f"Denomination: {image_info['denomination']}\n")
                f.write(f"Condition: {image_info['condition']}\n")
                f.write(f"Quality Score: {image_info['quality_score']}\n")
        
        self.import_results["total_files_imported"] = len(data['images'])
        self.import_results["imports_created"].append(f"pickle: {source_path}")
    
    def _import_from_directory(self, source_path: Path):
        """Import dataset from directory"""
        # Copy entire directory
        shutil.copytree(source_path, self.dataset_path, dirs_exist_ok=True)
        
        # Count files
        imported_files = list(self.dataset_path.rglob("*"))
        self.import_results["total_files_imported"] = len(imported_files)
        self.import_results["import_size_mb"] = sum(f.stat().st_size for f in imported_files if f.is_file()) / (1024 * 1024)
        
        self.import_results["imports_created"].append(f"directory: {source_path}")
    
    def _generate_import_report(self):
        """Generate import report"""
        report_path = self.dataset_path / "metadata" / "import_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.import_results, f, indent=2)
        
        logger.info(f"Import report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Export/Import naira note dataset')
    parser.add_argument('--action', type=str, choices=['export', 'import'], required=True,
                       help='Action to perform')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output-path', type=str,
                       help='Output path for export')
    parser.add_argument('--source-path', type=str,
                       help='Source path for import')
    parser.add_argument('--format', type=str,
                       choices=['zip', 'tar', 'csv', 'json', 'pickle', 'directory', 'auto'],
                       default='auto', help='Export/import format')
    parser.add_argument('--include-metadata', action='store_true', default=True,
                       help='Include metadata files')
    parser.add_argument('--compress', action='store_true', default=True,
                       help='Compress archives')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'export':
            if not args.output_path:
                logger.error("Output path required for export")
                sys.exit(1)
            
            exporter = DatasetExporter(args.dataset_path)
            results = exporter.export_dataset(
                args.output_path, 
                args.format, 
                args.include_metadata, 
                args.compress
            )
            
            print("\n" + "="*60)
            print("DATASET EXPORT SUMMARY")
            print("="*60)
            print(f"Files Exported: {results['total_files_exported']}")
            print(f"Export Size: {results['export_size_mb']:.1f} MB")
            print(f"Exports Created: {len(results['exports_created'])}")
            for export in results['exports_created']:
                print(f"- {export}")
            print("="*60)
        
        elif args.action == 'import':
            if not args.source_path:
                logger.error("Source path required for import")
                sys.exit(1)
            
            importer = DatasetImporter(args.dataset_path)
            results = importer.import_dataset(args.source_path, args.format)
            
            print("\n" + "="*60)
            print("DATASET IMPORT SUMMARY")
            print("="*60)
            print(f"Files Imported: {results['total_files_imported']}")
            print(f"Import Size: {results['import_size_mb']:.1f} MB")
            print(f"Imports Created: {len(results['imports_created'])}")
            for import_item in results['imports_created']:
                print(f"- {import_item}")
            print("="*60)
    
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
