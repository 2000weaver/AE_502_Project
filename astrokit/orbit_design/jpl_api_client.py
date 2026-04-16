"""
jpl_api_client.py
-----------------
Client for querying the NASA/JPL Three-Body Periodic Orbits API.

This module provides a high-level interface to query periodic orbits from the
JPL Poincare Catalog database. The API returns orbits in the CR3BP for various
three-body systems (Earth-Moon, Jupiter-Europa, Saturn-Titan, etc.).

API Documentation:
    https://ssd-api.jpl.nasa.gov/doc/periodic_orbits.html

Example usage:
    >>> client = JPLPeriodicOrbitsAPIClient()
    >>> results = client.query(
    ...     system="earth-moon",
    ...     family="halo",
    ...     libration_point=1,
    ...     branch="N"
    ... )
    >>> orbits = results.to_reference_trajectories()
"""

from __future__ import annotations

import numpy as np
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any

from .reference_trajectory import ReferenceTrajectory

__all__ = [
    "JPLPeriodicOrbitsAPIClient",
    "APIResponse",
    "SystemInfo",
    "QueryFilter",
    "FamilyType",
    "PeriodUnits",
    "BranchType",
    "query_earth_moon_halo",
    "query_jupiter_europa_dro",
    "query_saturn_titan_butterfly",
]


# ===================================================================
# Type definitions and constants
# ===================================================================

FamilyType = Literal[
    "halo", "vertical", "axial", "lyapunov", "longp", "short",
    "butterfly", "dragonfly", "resonant", "dro", "dpo", "lpo"
]

PeriodUnits = Literal["s", "h", "d", "TU"]

BranchType = Literal["N", "S", "E", "W"]


@dataclass
class SystemInfo:
    """Information about a three-body system."""
    name: str
    mass_ratio: float
    radius_secondary: float
    L1: np.ndarray
    L2: np.ndarray
    L3: np.ndarray
    L4: np.ndarray
    L5: np.ndarray
    lunit: float  # Length unit in km
    tunit: float  # Time unit in seconds


@dataclass
class QueryFilter:
    """Filtering options for orbit queries."""
    periodmin: Optional[float] = None
    periodmax: Optional[float] = None
    periodunits: PeriodUnits = "TU"
    jacobimin: Optional[float] = None
    jacobimax: Optional[float] = None
    stabmin: Optional[float] = None
    stabmax: Optional[float] = None


@dataclass
class APIResponse:
    """Parsed response from JPL API."""
    signature: Dict[str, str]
    system: SystemInfo
    family: str
    libration_point: Optional[int]
    branch: Optional[str]
    filter: QueryFilter
    count: int
    limits: Dict[str, List[float]]
    fields: List[str]
    data: np.ndarray
    raw_json: Dict[str, Any]

    def to_reference_trajectories(self) -> List[ReferenceTrajectory]:
        """
        Convert API response data to ReferenceTrajectory objects.

        Returns a list of ReferenceTrajectory objects, one for each orbit
        in the response data.
        """
        trajectories = []

        # Find indices of each column in the data
        try:
            x_idx = self.fields.index("x")
            y_idx = self.fields.index("y")
            z_idx = self.fields.index("z")
            vx_idx = self.fields.index("vx")
            vy_idx = self.fields.index("vy")
            vz_idx = self.fields.index("vz")
            jacobi_idx = self.fields.index("jacobi")
            period_idx = self.fields.index("period")
        except ValueError as e:
            raise ValueError(f"Required field not found in API response: {e}")

        for row in self.data:
            initial_state = np.array([
                float(row[x_idx]),
                float(row[y_idx]),
                float(row[z_idx]),
                float(row[vx_idx]),
                float(row[vy_idx]),
                float(row[vz_idx]),
            ])

            period = float(row[period_idx])
            jacobi = float(row[jacobi_idx])

            traj = ReferenceTrajectory(
                initial_state=initial_state,
                period=period,
                t=np.array([0.0]),  # API doesn't provide trajectory time series
                states=initial_state.reshape(6, 1),  # Single point (initial state)
                family_type=self.family,
                jacobi_constant=jacobi,
                metadata={
                    "system": self.system.name,
                    "branch": self.branch,
                    "libration_point": self.libration_point,
                    "source": "JPL API",
                }
            )
            trajectories.append(traj)

        return trajectories

    def get_orbit_by_index(self, index: int) -> ReferenceTrajectory:
        """
        Get a single orbit from the response by index.

        Parameters
        ----------
        index : int
            Index of the orbit (0-based).

        Returns
        -------
        ReferenceTrajectory
            The orbit at the specified index.
        """
        if index < 0 or index >= len(self.data):
            raise IndexError(f"Index {index} out of range [0, {len(self.data)})")

        trajectories = self.to_reference_trajectories()
        return trajectories[index]


# ===================================================================
# JPL API Client
# ===================================================================

class JPLPeriodicOrbitsAPIClient:
    """
    Client for the NASA/JPL Three-Body Periodic Orbits API.

    This client handles queries to the JPL periodic orbits database,
    which contains orbits computed for various three-body systems
    using the CR3BP.

    Attributes
    ----------
    base_url : str
        Base URL of the JPL API.
    timeout : float
        Timeout for API requests in seconds.
    """

    BASE_URL = "https://ssd-api.jpl.nasa.gov/periodic_orbits.api"

    def __init__(self, timeout: float = 30.0):
        """
        Initialize the JPL API client.

        Parameters
        ----------
        timeout : float, optional
            Timeout for HTTP requests (default: 30 seconds).
        """
        self.timeout = timeout
        self.session = requests.Session()

    def query(
        self,
        system: str,
        family: FamilyType,
        libration_point: Optional[int] = None,
        branch: Optional[str] = None,
        periodmin: Optional[float] = None,
        periodmax: Optional[float] = None,
        periodunits: PeriodUnits = "TU",
        jacobimin: Optional[float] = None,
        jacobimax: Optional[float] = None,
        stabmin: Optional[float] = None,
        stabmax: Optional[float] = None,
    ) -> APIResponse:
        """
        Query the JPL periodic orbits database.

        Parameters
        ----------
        system : str
            Three-body system (e.g., "earth-moon", "jupiter-europa").
        family : FamilyType
            Orbit family (e.g., "halo", "vertical", "dro").
        libration_point : int, optional
            Libration point (1-5). Required for some families.
        branch : str, optional
            Branch name ("N"/"S" for halo, etc.). May be required.
        periodmin : float, optional
            Minimum period (inclusive).
        periodmax : float, optional
            Maximum period (inclusive).
        periodunits : PeriodUnits, optional
            Units for period: "s", "h", "d", or "TU" (default).
        jacobimin : float, optional
            Minimum Jacobi constant (inclusive).
        jacobimax : float, optional
            Maximum Jacobi constant (inclusive).
        stabmin : float, optional
            Minimum stability index (inclusive).
        stabmax : float, optional
            Maximum stability index (inclusive).

        Returns
        -------
        APIResponse
            Parsed API response containing orbits and metadata.

        Raises
        ------
        requests.RequestException
            If the HTTP request fails.
        ValueError
            If the API returns an error or warning.
        """
        params = self._build_params(
            system=system,
            family=family,
            libration_point=libration_point,
            branch=branch,
            periodmin=periodmin,
            periodmax=periodmax,
            periodunits=periodunits,
            jacobimin=jacobimin,
            jacobimax=jacobimax,
            stabmin=stabmin,
            stabmax=stabmax,
        )

        response = self.session.get(
            self.BASE_URL,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()

        json_data = response.json()

        # Check for API errors or warnings
        if "error" in json_data:
            raise ValueError(f"API Error: {json_data['error']}")
        if "warning" in json_data:
            print(f"API Warning: {json_data['warning']}")

        return self._parse_response(json_data)

    def _build_params(
        self,
        system: str,
        family: FamilyType,
        libration_point: Optional[int] = None,
        branch: Optional[str] = None,
        periodmin: Optional[float] = None,
        periodmax: Optional[float] = None,
        periodunits: PeriodUnits = "TU",
        jacobimin: Optional[float] = None,
        jacobimax: Optional[float] = None,
        stabmin: Optional[float] = None,
        stabmax: Optional[float] = None,
    ) -> Dict[str, str]:
        """Build query parameters for the API request."""
        params = {
            "sys": system.lower(),
            "family": family.lower(),
        }

        if libration_point is not None:
            params["libr"] = str(libration_point)

        if branch is not None:
            params["branch"] = str(branch)

        if periodmin is not None:
            params["periodmin"] = str(periodmin)

        if periodmax is not None:
            params["periodmax"] = str(periodmax)

        if periodunits != "TU":
            params["periodunits"] = periodunits

        if jacobimin is not None:
            params["jacobimin"] = str(jacobimin)

        if jacobimax is not None:
            params["jacobimax"] = str(jacobimax)

        if stabmin is not None:
            params["stabmin"] = str(stabmin)

        if stabmax is not None:
            params["stabmax"] = str(stabmax)

        return params

    def _parse_response(self, json_data: Dict[str, Any]) -> APIResponse:
        """Parse JSON response from the API."""
        # System information
        sys_info_raw = json_data["system"]
        system_info = SystemInfo(
            name=sys_info_raw["name"],
            mass_ratio=float(sys_info_raw["mass_ratio"]),
            radius_secondary=float(sys_info_raw["radius_secondary"]),
            L1=np.array([float(x) for x in sys_info_raw["L1"]]),
            L2=np.array([float(x) for x in sys_info_raw["L2"]]),
            L3=np.array([float(x) for x in sys_info_raw["L3"]]),
            L4=np.array([float(x) for x in sys_info_raw["L4"]]),
            L5=np.array([float(x) for x in sys_info_raw["L5"]]),
            lunit=float(sys_info_raw["lunit"]),
            tunit=float(sys_info_raw["tunit"]),
        )

        # Filter information
        filter_raw = json_data.get("filter", {})
        query_filter = QueryFilter(
            periodmin=float(filter_raw.get("periodmin")) if "periodmin" in filter_raw else None,
            periodmax=float(filter_raw.get("periodmax")) if "periodmax" in filter_raw else None,
            periodunits=filter_raw.get("periodunits", "TU"),
            jacobimin=float(filter_raw.get("jacobimin")) if "jacobimin" in filter_raw else None,
            jacobimax=float(filter_raw.get("jacobimax")) if "jacobimax" in filter_raw else None,
            stabmin=float(filter_raw.get("stabmin")) if "stabmin" in filter_raw else None,
            stabmax=float(filter_raw.get("stabmax")) if "stabmax" in filter_raw else None,
        )

        # Convert data to numpy array
        data_array = np.array(json_data["data"], dtype=float)

        response = APIResponse(
            signature=json_data["signature"],
            system=system_info,
            family=json_data["family"],
            libration_point=json_data.get("libration_point"),
            branch=json_data.get("branch"),
            filter=query_filter,
            count=json_data["count"],
            limits=json_data["limits"],
            fields=json_data["fields"],
            data=data_array,
            raw_json=json_data,
        )

        return response

    def query_and_save(
        self,
        output_path: str | Path,
        **query_kwargs
    ) -> APIResponse:
        """
        Query the API and save results to a NumPy file.

        Parameters
        ----------
        output_path : str or Path
            Path where to save the .npz file.
        **query_kwargs
            Keyword arguments passed to query().

        Returns
        -------
        APIResponse
            The parsed API response.
        """
        response = self.query(**query_kwargs)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as .npz for easy loading
        np.savez_compressed(
            output_path,
            system=response.system.name,
            family=response.family,
            branch=response.branch,
            libration_point=response.libration_point,
            fields=response.fields,
            data=response.data,
            signature=str(response.signature),
        )

        print(f"Saved {response.count} orbits to {output_path}")
        return response


# ===================================================================
# Convenience functions
# ===================================================================

def query_earth_moon_halo(
    libration_point: int = 1,
    branch: str = "N",
    **kwargs
) -> APIResponse:
    """
    Query Earth-Moon halo orbits.

    Parameters
    ----------
    libration_point : int, optional
        L1, L2, or L3 (default: L1).
    branch : str, optional
        "N" for northern or "S" for southern (default: "N").
    **kwargs
        Additional query parameters.

    Returns
    -------
    APIResponse
        API response with matching orbits.
    """
    client = JPLPeriodicOrbitsAPIClient()
    return client.query(
        system="earth-moon",
        family="halo",
        libration_point=libration_point,
        branch=branch,
        **kwargs
    )


def query_jupiter_europa_dro(**kwargs) -> APIResponse:
    """
    Query Jupiter-Europa distant retrograde orbits (DROs).

    Parameters
    ----------
    **kwargs
        Query parameters (e.g., period limits, Jacobi constant filters).

    Returns
    -------
    APIResponse
        API response with matching orbits.
    """
    client = JPLPeriodicOrbitsAPIClient()
    return client.query(
        system="jupiter-europa",
        family="dro",
        **kwargs
    )


def query_saturn_titan_butterfly(
    jacobimin: float = 3.0,
    stabmax: float = 1.0,
    **kwargs
) -> APIResponse:
    """
    Query Saturn-Titan butterfly orbits.

    Parameters
    ----------
    jacobimin : float, optional
        Minimum Jacobi constant (default: 3.0).
    stabmax : float, optional
        Maximum stability index (default: 1.0 for stable orbits).
    **kwargs
        Additional query parameters.

    Returns
    -------
    APIResponse
        API response with matching orbits.
    """
    client = JPLPeriodicOrbitsAPIClient()
    return client.query(
        system="saturn-titan",
        family="butterfly",
        jacobimin=jacobimin,
        stabmax=stabmax,
        **kwargs
    )
