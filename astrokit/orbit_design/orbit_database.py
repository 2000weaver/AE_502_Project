import os
import json
import numpy as np
from pathlib import Path
from .reference_trajectory import ReferenceTrajectory


class OrbitDatabase:
    """
    Database for storing and managing periodic orbits.
    Supports multi-family storage with efficient retrieval.
    """

    def __init__(self, database_path="./orbit_database"):
        self.db_path = Path(database_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.db_path / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """Load metadata about stored orbits."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'families': {}}

    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _create_family_dir(self, family_name, family_type):
        """Create directory for a family if it doesn't exist."""
        family_dir = self.db_path / family_name
        family_dir.mkdir(parents=True, exist_ok=True)
        return family_dir

    def store_family(self, family_name, family_type, orbits, metadata=None):
        """
        Store an entire orbit family.

        Parameters:
        family_name: Unique name for the family (e.g., 'halo_earth_moon_l2')
        family_type: Type of family ('halo', 'nrho', 'butterfly')
        orbits: List of ReferenceTrajectory objects
        metadata: Optional dict with family metadata
        """
        family_dir = self._create_family_dir(family_name, family_type)

        if family_name not in self.metadata['families']:
            self.metadata['families'][family_name] = {
                'type': family_type,
                'count': 0,
                'metadata': metadata or {}
            }

        # Store each orbit
        for i, orbit in enumerate(orbits):
            orbit_file = family_dir / f"orbit_{i:04d}.npz"
            orbit.save(str(orbit_file))

            self.metadata['families'][family_name]['count'] = len(orbits)

        self._save_metadata()
        print(f"Stored {len(orbits)} orbits in family '{family_name}'")

    def load_family(self, family_name):
        """
        Load all orbits from a family.

        Returns:
        List of ReferenceTrajectory objects
        """
        family_dir = self.db_path / family_name
        if not family_dir.exists():
            raise FileNotFoundError(f"Family '{family_name}' not found")

        orbits = []
        orbit_files = sorted(family_dir.glob("orbit_*.npz"))

        for orbit_file in orbit_files:
            orbit = ReferenceTrajectory.load(str(orbit_file))
            orbits.append(orbit)

        return orbits

    def get_orbit(self, family_name, index):
        """Get a specific orbit from a family by index."""
        family_dir = self.db_path / family_name
        orbit_file = family_dir / f"orbit_{index:04d}.npz"

        if not orbit_file.exists():
            raise FileNotFoundError(f"Orbit {index} not found in family '{family_name}'")

        return ReferenceTrajectory.load(str(orbit_file))

    def list_families(self):
        """List all stored families."""
        return list(self.metadata['families'].keys())

    def get_family_info(self, family_name):
        """Get information about a family."""
        if family_name not in self.metadata['families']:
            raise ValueError(f"Family '{family_name}' not found")

        info = self.metadata['families'][family_name].copy()
        family_dir = self.db_path / family_name
        orbit_count = len(list(family_dir.glob("orbit_*.npz")))
        info['stored_count'] = orbit_count

        return info

    def get_family_statistics(self, family_name):
        """Compute statistics for a family (periods, jacobi constants, etc)."""
        orbits = self.load_family(family_name)

        if not orbits:
            return {}

        periods = np.array([o.period for o in orbits])
        jacobis = np.array([o.jacobi_constant for o in orbits])

        stats = {
            'count': len(orbits),
            'period_min': float(np.min(periods)),
            'period_max': float(np.max(periods)),
            'period_mean': float(np.mean(periods)),
            'period_std': float(np.std(periods)),
            'jacobi_min': float(np.min(jacobis)),
            'jacobi_max': float(np.max(jacobis)),
            'jacobi_mean': float(np.mean(jacobis)),
            'jacobi_std': float(np.std(jacobis)),
        }

        return stats

    def export_family_csv(self, family_name, output_file=None):
        """Export family data to CSV for analysis."""
        orbits = self.load_family(family_name)

        if output_file is None:
            output_file = self.db_path / f"{family_name}.csv"

        import csv

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Index', 'Period', 'Jacobi Constant', 'x0', 'y0', 'z0', 'vx0', 'vy0', 'vz0'])

            for i, orbit in enumerate(orbits):
                state = orbit.initial_state
                writer.writerow([
                    i,
                    orbit.period,
                    orbit.jacobi_constant,
                    *state
                ])

        print(f"Exported family '{family_name}' to {output_file}")

    def delete_family(self, family_name):
        """Delete a family from the database."""
        family_dir = self.db_path / family_name
        if family_dir.exists():
            import shutil
            shutil.rmtree(family_dir)

        if family_name in self.metadata['families']:
            del self.metadata['families'][family_name]
            self._save_metadata()

        print(f"Deleted family '{family_name}'")

    def merge_families(self, source_family, target_family):
        """Merge source family into target family."""
        source_orbits = self.load_family(source_family)
        target_orbits = self.load_family(target_family)

        merged = target_orbits + source_orbits

        self.store_family(target_family,
                         self.metadata['families'][target_family]['type'],
                         merged,
                         self.metadata['families'][target_family]['metadata'])

        self.delete_family(source_family)
        print(f"Merged {len(source_orbits)} orbits from '{source_family}' into '{target_family}'")

    def summary(self):
        """Print a summary of the database."""
        print("\n" + "=" * 60)
        print("ORBIT DATABASE SUMMARY")
        print("=" * 60)
        print(f"Database location: {self.db_path.absolute()}")
        print(f"Total families: {len(self.metadata['families'])}\n")

        for family_name in self.list_families():
            info = self.get_family_info(family_name)
            stats = self.get_family_statistics(family_name)

            print(f"Family: {family_name}")
            print(f"  Type: {info['type']}")
            print(f"  Orbits: {info['stored_count']}")

            if stats:
                print(f"  Period range: [{stats['period_min']:.4f}, {stats['period_max']:.4f}]")
                print(f"  Jacobi range: [{stats['jacobi_min']:.6f}, {stats['jacobi_max']:.6f}]")

            print()

        print("=" * 60)
