"""
ASTRAEUS Observatory Intelligence - Facility & Instrument Registry (WS7.5).
Provides versioned models of ground-based and space-based astronomical observatories,
instrument profiles, filter passbands, and operational access classes.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field


class FacilityAccessType(str, Enum):
    """Operational access classification governing target scheduling dispatch."""
    STREAMING_SURVEY_BROKER = "STREAMING_SURVEY_BROKER"  # e.g. ZTF, Rubin alert streams
    RAPID_TOO_TARGETABLE = "RAPID_TOO_TARGETABLE"        # Fast Target-of-Opportunity (Swift ToO, VLT RRM)
    QUEUE_SCHEDULED = "QUEUE_SCHEDULED"                  # Queue scheduled facilities (Gemini, VLT, MeerKAT, VLA)
    PROPOSAL_GATED = "PROPOSAL_GATED"                    # Proposal-gated high-barrier facilities (JWST, Chandra)
    ARCHIVAL_RETROSPECTIVE_ONLY = "ARCHIVAL_RETROSPECTIVE_ONLY"  # Data archives without pointing dispatch


class WavelengthRegime(str, Enum):
    OPTICAL = "OPTICAL"
    NIR = "NIR"
    UV = "UV"
    XRAY = "XRAY"
    RADIO = "RADIO"


class ObservatoryLocation(BaseModel):
    """Geodetic or orbital coordinates for an astronomical facility."""
    name: str
    latitude_deg: float = Field(..., ge=-90.0, le=90.0)
    longitude_deg: float = Field(..., ge=-180.0, le=360.0, description="East positive longitude")
    elevation_m: float = Field(default=0.0)
    is_space_based: bool = False
    orbit_description: Optional[str] = None


class InstrumentFilter(BaseModel):
    """Photometric or spectroscopic filter passband parameters."""
    filter_id: str
    filter_name: str
    regime: WavelengthRegime
    effective_wavelength_angstrom: float
    bandwidth_angstrom: float
    zero_point_ab_mag: Optional[float] = None
    sky_brightness_mag_arcsec2: Optional[float] = None
    extinction_coeff: float = 0.12


class InstrumentProfile(BaseModel):
    """Physical and detector characteristics of an astronomical instrument."""
    instrument_id: str
    instrument_name: str
    regime: WavelengthRegime
    aperture_m: float = Field(..., gt=0.0)
    pixel_scale_arcsec: float = 0.2
    field_of_view_arcmin: float = 10.0
    read_noise_electrons: float = 5.0
    dark_current_e_per_s: float = 0.01
    slew_speed_deg_sec: float = 2.0
    settle_time_sec: float = 10.0
    readout_overhead_sec: float = 15.0
    min_exposure_sec: float = 1.0
    max_exposure_sec: float = 3600.0
    filters: Dict[str, InstrumentFilter] = Field(default_factory=dict)

    def get_filter(self, filter_id: str) -> Optional[InstrumentFilter]:
        return self.filters.get(filter_id)


class ObservatoryFacility(BaseModel):
    """Full architectural model of an astronomical facility."""
    facility_id: str
    full_name: str
    location: ObservatoryLocation
    access_type: FacilityAccessType
    instruments: Dict[str, InstrumentProfile] = Field(default_factory=dict)
    horizon_min_alt_deg: float = 20.0
    max_airmass: float = 2.0
    proposal_id_required: bool = False
    is_active: bool = True
    operational_schedule_notes: Optional[str] = None

    def get_instrument(self, inst_id: str) -> Optional[InstrumentProfile]:
        return self.instruments.get(inst_id)


class FacilityRegistry:
    """Central repository of supported astronomical facilities across wavebands."""

    def __init__(self):
        self._facilities: Dict[str, ObservatoryFacility] = {}
        self._initialize_default_facilities()

    def register_facility(self, facility: ObservatoryFacility) -> None:
        self._facilities[facility.facility_id.upper()] = facility

    def get_facility(self, facility_id: str) -> Optional[ObservatoryFacility]:
        return self._facilities.get(facility_id.strip().upper())

    def list_all_facilities(self) -> List[ObservatoryFacility]:
        return list(self._facilities.values())

    def list_facilities_by_regime(self, regime: WavelengthRegime) -> List[ObservatoryFacility]:
        matched = []
        for fac in self._facilities.values():
            if any(inst.regime == regime for inst in fac.instruments.values()):
                matched.append(fac)
        return matched

    def list_facilities_by_access_type(self, access_type: FacilityAccessType) -> List[ObservatoryFacility]:
        return [f for f in self._facilities.values() if f.access_type == access_type]

    def _initialize_default_facilities(self) -> None:
        """Configures the standard nine multi-facility telescopes."""
        
        # 1. Zwicky Transient Facility (Palomar Observatory)
        ztf_loc = ObservatoryLocation(
            name="Palomar Observatory",
            latitude_deg=33.3563,
            longitude_deg=-116.8648,
            elevation_m=1706.0,
        )
        ztf_filters = {
            "g": InstrumentFilter(filter_id="g", filter_name="ZTF_g", regime=WavelengthRegime.OPTICAL, effective_wavelength_angstrom=4770.0, bandwidth_angstrom=1400.0, zero_point_ab_mag=26.3, sky_brightness_mag_arcsec2=22.0, extinction_coeff=0.18),
            "r": InstrumentFilter(filter_id="r", filter_name="ZTF_r", regime=WavelengthRegime.OPTICAL, effective_wavelength_angstrom=6440.0, bandwidth_angstrom=1500.0, zero_point_ab_mag=26.2, sky_brightness_mag_arcsec2=21.2, extinction_coeff=0.10),
            "i": InstrumentFilter(filter_id="i", filter_name="ZTF_i", regime=WavelengthRegime.OPTICAL, effective_wavelength_angstrom=7970.0, bandwidth_angstrom=1400.0, zero_point_ab_mag=25.7, sky_brightness_mag_arcsec2=20.5, extinction_coeff=0.07),
        }
        ztf_inst = InstrumentProfile(
            instrument_id="ZTF_CAM",
            instrument_name="ZTF 48-inch Mosaic Camera",
            regime=WavelengthRegime.OPTICAL,
            aperture_m=1.22,
            pixel_scale_arcsec=1.01,
            field_of_view_arcmin=2800.0,
            slew_speed_deg_sec=3.0,
            settle_time_sec=8.0,
            readout_overhead_sec=10.0,
            filters=ztf_filters,
        )
        self.register_facility(ObservatoryFacility(
            facility_id="ZTF",
            full_name="Zwicky Transient Facility",
            location=ztf_loc,
            access_type=FacilityAccessType.RAPID_TOO_TARGETABLE,
            instruments={"ZTF_CAM": ztf_inst},
            horizon_min_alt_deg=20.0,
            max_airmass=2.2,
        ))

        # 2. Vera C. Rubin Observatory (Cerro Pachón)
        rubin_loc = ObservatoryLocation(
            name="Cerro Pachón",
            latitude_deg=-30.2407,
            longitude_deg=-70.7494,
            elevation_m=2663.0,
        )
        rubin_filters = {
            "u": InstrumentFilter(filter_id="u", filter_name="Rubin_u", regime=WavelengthRegime.OPTICAL, effective_wavelength_angstrom=3670.0, bandwidth_angstrom=550.0, zero_point_ab_mag=27.0, sky_brightness_mag_arcsec2=22.8, extinction_coeff=0.45),
            "g": InstrumentFilter(filter_id="g", filter_name="Rubin_g", regime=WavelengthRegime.OPTICAL, effective_wavelength_angstrom=4825.0, bandwidth_angstrom=1380.0, zero_point_ab_mag=28.3, sky_brightness_mag_arcsec2=22.2, extinction_coeff=0.19),
            "r": InstrumentFilter(filter_id="r", filter_name="Rubin_r", regime=WavelengthRegime.OPTICAL, effective_wavelength_angstrom=6222.0, bandwidth_angstrom=1360.0, zero_point_ab_mag=28.1, sky_brightness_mag_arcsec2=21.3, extinction_coeff=0.11),
            "i": InstrumentFilter(filter_id="i", filter_name="Rubin_i", regime=WavelengthRegime.OPTICAL, effective_wavelength_angstrom=7545.0, bandwidth_angstrom=1300.0, zero_point_ab_mag=27.8, sky_brightness_mag_arcsec2=20.6, extinction_coeff=0.08),
            "z": InstrumentFilter(filter_id="z", filter_name="Rubin_z", regime=WavelengthRegime.OPTICAL, effective_wavelength_angstrom=8691.0, bandwidth_angstrom=1200.0, zero_point_ab_mag=27.4, sky_brightness_mag_arcsec2=20.0, extinction_coeff=0.06),
            "y": InstrumentFilter(filter_id="y", filter_name="Rubin_y", regime=WavelengthRegime.NIR, effective_wavelength_angstrom=9710.0, bandwidth_angstrom=950.0, zero_point_ab_mag=26.6, sky_brightness_mag_arcsec2=19.3, extinction_coeff=0.04),
        }
        rubin_inst = InstrumentProfile(
            instrument_id="LSST_CAM",
            instrument_name="LSST Camera 3.2 Gpix",
            regime=WavelengthRegime.OPTICAL,
            aperture_m=8.4,
            pixel_scale_arcsec=0.2,
            field_of_view_arcmin=210.0,
            slew_speed_deg_sec=7.0,
            settle_time_sec=4.0,
            readout_overhead_sec=2.0,
            filters=rubin_filters,
        )
        self.register_facility(ObservatoryFacility(
            facility_id="RUBIN",
            full_name="Vera C. Rubin Observatory",
            location=rubin_loc,
            access_type=FacilityAccessType.STREAMING_SURVEY_BROKER,
            instruments={"LSST_CAM": rubin_inst},
            horizon_min_alt_deg=25.0,
            max_airmass=2.0,
        ))

        # 3. Neil Gehrels Swift Observatory (Space-based Low Earth Orbit)
        swift_loc = ObservatoryLocation(
            name="Swift Low Earth Orbit",
            latitude_deg=0.0,
            longitude_deg=0.0,
            elevation_m=550000.0,
            is_space_based=True,
            orbit_description="LEO 550 km, 96 min period, 20.6 deg inclination",
        )
        swift_filters = {
            "uvw2": InstrumentFilter(filter_id="uvw2", filter_name="Swift_UVW2", regime=WavelengthRegime.UV, effective_wavelength_angstrom=1928.0, bandwidth_angstrom=657.0, zero_point_ab_mag=24.5, extinction_coeff=0.0),
            "uvm2": InstrumentFilter(filter_id="uvm2", filter_name="Swift_UVM2", regime=WavelengthRegime.UV, effective_wavelength_angstrom=2246.0, bandwidth_angstrom=498.0, zero_point_ab_mag=24.2, extinction_coeff=0.0),
            "uvw1": InstrumentFilter(filter_id="uvw1", filter_name="Swift_UVW1", regime=WavelengthRegime.UV, effective_wavelength_angstrom=2600.0, bandwidth_angstrom=693.0, zero_point_ab_mag=24.8, extinction_coeff=0.0),
            "u": InstrumentFilter(filter_id="u", filter_name="Swift_U", regime=WavelengthRegime.UV, effective_wavelength_angstrom=3465.0, bandwidth_angstrom=785.0, zero_point_ab_mag=25.3, extinction_coeff=0.0),
        }
        swift_uvot = InstrumentProfile(
            instrument_id="UVOT",
            instrument_name="Ultra-Violet/Optical Telescope",
            regime=WavelengthRegime.UV,
            aperture_m=0.30,
            pixel_scale_arcsec=0.5,
            field_of_view_arcmin=17.0,
            slew_speed_deg_sec=5.0,
            settle_time_sec=20.0,
            readout_overhead_sec=5.0,
            filters=swift_filters,
        )
        swift_xrt = InstrumentProfile(
            instrument_id="XRT",
            instrument_name="X-ray Telescope (0.3-10 keV)",
            regime=WavelengthRegime.XRAY,
            aperture_m=0.51,
            pixel_scale_arcsec=2.36,
            field_of_view_arcmin=23.6,
            slew_speed_deg_sec=5.0,
            settle_time_sec=20.0,
            readout_overhead_sec=5.0,
            filters={"xray_03_10": InstrumentFilter(filter_id="xray_03_10", filter_name="XRT 0.3-10 keV", regime=WavelengthRegime.XRAY, effective_wavelength_angstrom=12.4, bandwidth_angstrom=40.0)},
        )
        self.register_facility(ObservatoryFacility(
            facility_id="SWIFT",
            full_name="Neil Gehrels Swift Observatory",
            location=swift_loc,
            access_type=FacilityAccessType.RAPID_TOO_TARGETABLE,
            instruments={"UVOT": swift_uvot, "XRT": swift_xrt},
            horizon_min_alt_deg=0.0,
            max_airmass=1.0,
        ))

        # 4. AstroSat (ISRO Space Observatory)
        astrosat_loc = ObservatoryLocation(
            name="AstroSat LEO Orbit",
            latitude_deg=0.0,
            longitude_deg=0.0,
            elevation_m=650000.0,
            is_space_based=True,
            orbit_description="LEO 650 km, 98 min period, 6 deg equatorial inclination",
        )
        astrosat_uvit = InstrumentProfile(
            instrument_id="UVIT",
            instrument_name="Ultra Violet Imaging Telescope (FUV/NUV)",
            regime=WavelengthRegime.UV,
            aperture_m=0.38,
            pixel_scale_arcsec=0.42,
            field_of_view_arcmin=28.0,
            slew_speed_deg_sec=1.5,
            settle_time_sec=60.0,
            readout_overhead_sec=10.0,
            filters={
                "FUV_F1": InstrumentFilter(filter_id="FUV_F1", filter_name="UVIT BaF2 (154nm)", regime=WavelengthRegime.UV, effective_wavelength_angstrom=1540.0, bandwidth_angstrom=380.0, zero_point_ab_mag=24.0),
                "NUV_N1": InstrumentFilter(filter_id="NUV_N1", filter_name="UVIT NUV-Silica (241nm)", regime=WavelengthRegime.UV, effective_wavelength_angstrom=2418.0, bandwidth_angstrom=300.0, zero_point_ab_mag=24.4),
            },
        )
        self.register_facility(ObservatoryFacility(
            facility_id="ASTROSAT",
            full_name="AstroSat Observatory (ISRO)",
            location=astrosat_loc,
            access_type=FacilityAccessType.QUEUE_SCHEDULED,
            instruments={"UVIT": astrosat_uvit},
            horizon_min_alt_deg=0.0,
            max_airmass=1.0,
        ))

        # 5. Very Large Telescope (ESO Cerro Paranal)
        vlt_loc = ObservatoryLocation(
            name="Cerro Paranal",
            latitude_deg=-24.6272,
            longitude_deg=-70.4042,
            elevation_m=2635.0,
        )
        vlt_xshooter = InstrumentProfile(
            instrument_id="XSHOOTER",
            instrument_name="X-shooter Echelle Spectrograph (UVB-VIS-NIR)",
            regime=WavelengthRegime.OPTICAL,
            aperture_m=8.2,
            pixel_scale_arcsec=0.16,
            field_of_view_arcmin=1.5,
            slew_speed_deg_sec=2.0,
            settle_time_sec=15.0,
            readout_overhead_sec=20.0,
            filters={
                "VIS_SPEC": InstrumentFilter(filter_id="VIS_SPEC", filter_name="X-shooter VIS (550-1000nm)", regime=WavelengthRegime.OPTICAL, effective_wavelength_angstrom=7500.0, bandwidth_angstrom=4500.0, zero_point_ab_mag=28.5),
                "NIR_SPEC": InstrumentFilter(filter_id="NIR_SPEC", filter_name="X-shooter NIR (1000-2500nm)", regime=WavelengthRegime.NIR, effective_wavelength_angstrom=16000.0, bandwidth_angstrom=15000.0, zero_point_ab_mag=27.8),
            },
        )
        self.register_facility(ObservatoryFacility(
            facility_id="VLT",
            full_name="ESO Very Large Telescope (Paranal)",
            location=vlt_loc,
            access_type=FacilityAccessType.RAPID_TOO_TARGETABLE,
            instruments={"XSHOOTER": vlt_xshooter},
            horizon_min_alt_deg=20.0,
            max_airmass=2.0,
        ))

        # 6. Gemini Observatory (Gemini South & North)
        gemini_loc = ObservatoryLocation(
            name="Cerro Pachón (Gemini South)",
            latitude_deg=-30.2407,
            longitude_deg=-70.7367,
            elevation_m=2722.0,
        )
        gemini_gmos = InstrumentProfile(
            instrument_id="GMOS",
            instrument_name="Gemini Multi-Object Spectrograph",
            regime=WavelengthRegime.OPTICAL,
            aperture_m=8.1,
            pixel_scale_arcsec=0.08,
            field_of_view_arcmin=5.5,
            slew_speed_deg_sec=1.5,
            settle_time_sec=15.0,
            readout_overhead_sec=18.0,
            filters={
                "g": InstrumentFilter(filter_id="g", filter_name="GMOS_g", regime=WavelengthRegime.OPTICAL, effective_wavelength_angstrom=4750.0, bandwidth_angstrom=1500.0, zero_point_ab_mag=28.0),
                "r": InstrumentFilter(filter_id="r", filter_name="GMOS_r", regime=WavelengthRegime.OPTICAL, effective_wavelength_angstrom=6300.0, bandwidth_angstrom=1400.0, zero_point_ab_mag=27.9),
                "i": InstrumentFilter(filter_id="i", filter_name="GMOS_i", regime=WavelengthRegime.OPTICAL, effective_wavelength_angstrom=7800.0, bandwidth_angstrom=1400.0, zero_point_ab_mag=27.5),
            },
        )
        self.register_facility(ObservatoryFacility(
            facility_id="GEMINI",
            full_name="Gemini Observatory",
            location=gemini_loc,
            access_type=FacilityAccessType.QUEUE_SCHEDULED,
            instruments={"GMOS": gemini_gmos},
            horizon_min_alt_deg=20.0,
            max_airmass=2.0,
        ))

        # 7. James Webb Space Telescope (Sun-Earth L2 Orbit)
        jwst_loc = ObservatoryLocation(
            name="Sun-Earth L2 Halo Orbit",
            latitude_deg=0.0,
            longitude_deg=0.0,
            elevation_m=1500000000.0,
            is_space_based=True,
            orbit_description="Sun-Earth L2 halo orbit, 85-135 deg solar elongation field of regard",
        )
        jwst_nircam = InstrumentProfile(
            instrument_id="NIRCAM",
            instrument_name="Near-Infrared Camera (0.6-5.0um)",
            regime=WavelengthRegime.NIR,
            aperture_m=6.5,
            pixel_scale_arcsec=0.031,
            field_of_view_arcmin=9.7,
            slew_speed_deg_sec=0.1,
            settle_time_sec=300.0,
            readout_overhead_sec=60.0,
            min_exposure_sec=10.0,
            max_exposure_sec=7200.0,
            filters={
                "F150W": InstrumentFilter(filter_id="F150W", filter_name="NIRCam F150W (1.5um)", regime=WavelengthRegime.NIR, effective_wavelength_angstrom=15007.0, bandwidth_angstrom=3160.0, zero_point_ab_mag=29.2, sky_brightness_mag_arcsec2=22.5),
                "F200W": InstrumentFilter(filter_id="F200W", filter_name="NIRCam F200W (2.0um)", regime=WavelengthRegime.NIR, effective_wavelength_angstrom=19900.0, bandwidth_angstrom=4600.0, zero_point_ab_mag=29.4, sky_brightness_mag_arcsec2=22.4),
                "F356W": InstrumentFilter(filter_id="F356W", filter_name="NIRCam F356W (3.6um)", regime=WavelengthRegime.NIR, effective_wavelength_angstrom=35630.0, bandwidth_angstrom=7870.0, zero_point_ab_mag=28.8, sky_brightness_mag_arcsec2=21.8),
                "F444W": InstrumentFilter(filter_id="F444W", filter_name="NIRCam F444W (4.4um)", regime=WavelengthRegime.NIR, effective_wavelength_angstrom=44210.0, bandwidth_angstrom=10240.0, zero_point_ab_mag=28.2, sky_brightness_mag_arcsec2=21.0),
            },
        )
        self.register_facility(ObservatoryFacility(
            facility_id="JWST",
            full_name="James Webb Space Telescope (NASA/ESA/CSA)",
            location=jwst_loc,
            access_type=FacilityAccessType.PROPOSAL_GATED,
            instruments={"NIRCAM": jwst_nircam},
            proposal_id_required=True,
            horizon_min_alt_deg=0.0,
            max_airmass=1.0,
        ))

        # 8. MeerKAT Radio Telescope (SARAO, South Africa)
        meerkat_loc = ObservatoryLocation(
            name="Karoo Semi-Desert",
            latitude_deg=-30.7132,
            longitude_deg=21.4431,
            elevation_m=1038.0,
        )
        meerkat_receiver = InstrumentProfile(
            instrument_id="MEERKAT_L",
            instrument_name="MeerKAT L-Band Receiver (856-1712 MHz)",
            regime=WavelengthRegime.RADIO,
            aperture_m=64.0 * 13.5,  # 64 dishes of 13.5m effective
            pixel_scale_arcsec=5.0,
            field_of_view_arcmin=60.0,
            slew_speed_deg_sec=1.0,
            settle_time_sec=10.0,
            readout_overhead_sec=5.0,
            filters={
                "L_BAND": InstrumentFilter(filter_id="L_BAND", filter_name="MeerKAT L-band 1.4 GHz", regime=WavelengthRegime.RADIO, effective_wavelength_angstrom=2.14e9, bandwidth_angstrom=1.0e9),
            },
        )
        self.register_facility(ObservatoryFacility(
            facility_id="MEERKAT",
            full_name="MeerKAT Radio Telescope (SARAO)",
            location=meerkat_loc,
            access_type=FacilityAccessType.QUEUE_SCHEDULED,
            instruments={"MEERKAT_L": meerkat_receiver},
            horizon_min_alt_deg=15.0,
            max_airmass=3.0,
        ))

        # 9. Karl G. Jansky Very Large Array (NRAO Socorro)
        vla_loc = ObservatoryLocation(
            name="Plains of San Agustin",
            latitude_deg=34.0784,
            longitude_deg=-107.6184,
            elevation_m=2124.0,
        )
        vla_receiver = InstrumentProfile(
            instrument_id="VLA_CX",
            instrument_name="VLA C/X Band Synthesis Array",
            regime=WavelengthRegime.RADIO,
            aperture_m=27.0 * 25.0,
            pixel_scale_arcsec=0.5,
            field_of_view_arcmin=10.0,
            slew_speed_deg_sec=0.6,
            settle_time_sec=8.0,
            readout_overhead_sec=3.0,
            filters={
                "C_BAND": InstrumentFilter(filter_id="C_BAND", filter_name="VLA C-band (4-8 GHz)", regime=WavelengthRegime.RADIO, effective_wavelength_angstrom=5.0e8, bandwidth_angstrom=4.0e8),
                "X_BAND": InstrumentFilter(filter_id="X_BAND", filter_name="VLA X-band (8-12 GHz)", regime=WavelengthRegime.RADIO, effective_wavelength_angstrom=3.0e8, bandwidth_angstrom=4.0e8),
            },
        )
        self.register_facility(ObservatoryFacility(
            facility_id="VLA",
            full_name="Karl G. Jansky Very Large Array (NRAO)",
            location=vla_loc,
            access_type=FacilityAccessType.QUEUE_SCHEDULED,
            instruments={"VLA_CX": vla_receiver},
            horizon_min_alt_deg=12.0,
            max_airmass=3.5,
        ))
