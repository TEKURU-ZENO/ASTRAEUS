"""
ASTRAEUS Experimental Intelligence: Telescope Scheduling Environment (Track C).
Gymnasium-compatible environment modeling astronomical telescope operations,
spherical coordinate slewing dynamics, airmass constraints, stochastic weather,
and transient Target-of-Opportunity (ToO) preemption.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from engines.experimental_design.eig_objective import (
    TransientTarget,
    ExpectedInformationGain,
)


def angular_distance_deg(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    """Calculates spherical angular separation in degrees using the haversine formula."""
    r1, d1 = math.radians(ra1_deg), math.radians(dec1_deg)
    r2, d2 = math.radians(ra2_deg), math.radians(dec2_deg)
    d_ra = r2 - r1
    d_dec = d2 - d1

    a = math.sin(d_dec / 2.0) ** 2 + math.cos(d1) * math.cos(d2) * (math.sin(d_ra / 2.0) ** 2)
    a = min(1.0, max(0.0, a))
    dist_rad = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return math.degrees(dist_rad)


class TelescopeSchedulingEnv:
    """
    Observational telescope scheduling reinforcement learning environment.
    Observation space: LST, current pointing, weather, and candidate targets.
    Action space: select target k in [0, n_targets - 1] to observe, or n_targets to idle.
    """

    def __init__(
        self,
        n_targets: int = 8,
        observatory_lat_deg: float = 31.95,
        night_duration_hours: float = 8.0,
        slew_speed_deg_sec: float = 2.5,
        slew_settle_sec: float = 10.0,
        exposure_time_sec: float = 300.0,
        max_airmass: float = 2.0,
    ):
        self.n_targets = n_targets
        self.lat_rad = math.radians(observatory_lat_deg)
        self.night_duration_sec = night_duration_hours * 3600.0
        self.slew_speed = slew_speed_deg_sec
        self.slew_settle = slew_settle_sec
        self.exposure_time_sec = exposure_time_sec
        self.max_airmass = max_airmass

        self.eig_calc = ExpectedInformationGain()
        self.targets: List[TransientTarget] = []
        self.current_time_sec: float = 0.0
        self.lst_hours: float = 0.0
        self.tel_ra: float = 0.0
        self.tel_dec: float = observatory_lat_deg  # Zenith pointing initially
        self.weather_state: float = 1.0            # 1.0 = clear, 0.5 = degraded, 0.0 = closed
        self.rng = np.random.RandomState(42)

    def calculate_airmass(self, ra_deg: float, dec_deg: float, lst_hours: float) -> float:
        """
        Computes plane-parallel airmass X(z) = 1 / cos(z).
        Returns 99.0 if target is below horizon.
        """
        dec_rad = math.radians(dec_deg)
        ha_hours = (lst_hours - (ra_deg / 15.0)) % 24.0
        if ha_hours > 12.0:
            ha_hours -= 24.0
        ha_rad = math.radians(ha_hours * 15.0)

        # cos(z) = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(HA)
        cos_z = (
            math.sin(self.lat_rad) * math.sin(dec_rad)
            + math.cos(self.lat_rad) * math.cos(dec_rad) * math.cos(ha_rad)
        )

        if cos_z <= 0.10:  # Horizon threshold ~ 84 degrees zenith angle
            return 99.0
        return float(1.0 / cos_z)

    def calculate_slew_time_sec(self, target_ra: float, target_dec: float) -> float:
        """Computes slewing time from current pointing to target."""
        dist = angular_distance_deg(self.tel_ra, self.tel_dec, target_ra, target_dec)
        return float(self.slew_settle + dist / self.slew_speed)

    def _generate_targets(self, seed: Optional[int] = None) -> List[TransientTarget]:
        """Generates realistic heterogeneous transient target population."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        targets = []
        # Target types: SN Ia, Kilonova, GRB afterglow, TDE
        types = [
            ("SN_Ia", 19.5, 48.0, 1.0, False),
            ("KN_GW", 21.0, 12.0, 4.0, True),     # High priority rapid decay ToO
            ("GRB_aft", 18.0, 6.0, 3.5, True),     # Ultra rapid decay ToO
            ("TDE_flare", 20.0, 96.0, 1.5, False),
            ("SN_II", 18.5, 72.0, 1.0, False),
            ("FastBlue", 21.5, 18.0, 2.5, False),
            ("AGN_flare", 17.5, 120.0, 0.8, False),
            ("Nova", 16.0, 36.0, 1.2, False),
        ]

        for idx in range(self.n_targets):
            t_type, base_m, tau, prio, is_too = types[idx % len(types)]
            # Spread RA across visible sky
            ra = float((self.lst_hours * 15.0 + self.rng.uniform(-60.0, 60.0)) % 360.0)
            dec = float(np.clip(math.degrees(self.lat_rad) + self.rng.uniform(-30.0, 30.0), -60.0, 80.0))
            time_since_trigger = float(self.rng.uniform(1.0, 12.0))

            targets.append(
                TransientTarget(
                    target_id=f"TGT_{idx:02d}_{t_type}",
                    ra_deg=ra,
                    dec_deg=dec,
                    base_magnitude=base_m + float(self.rng.uniform(-0.5, 0.5)),
                    decay_timescale_hours=tau,
                    time_since_trigger_hours=time_since_trigger,
                    priority_weight=prio,
                    is_too=is_too,
                    prior_variance=1.0,
                    times_observed=0,
                )
            )

        return targets

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets environment to start of observing night."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.current_time_sec = 0.0
        self.lst_hours = float(self.rng.uniform(0.0, 24.0))
        self.tel_ra = float(self.lst_hours * 15.0)
        self.tel_dec = float(math.degrees(self.lat_rad))
        self.weather_state = 1.0
        self.targets = self._generate_targets(seed)

        obs = self._get_observation()
        info = {"lst_hours": self.lst_hours, "num_targets": len(self.targets)}
        return obs, info

    def _get_observation(self) -> np.ndarray:
        """
        Constructs normalized flat observation vector:
        - 5 observatory features: [sin(LST), cos(LST), tel_ra/360, tel_dec/90, weather]
        - 8 features per target * n_targets:
          [ra/360, dec/90, mag/30, tau/120, time/48, airmass/3, variance, is_too]
        Total dimension: 5 + 8 * n_targets.
        """
        lst_rad = 2.0 * math.pi * (self.lst_hours / 24.0)
        obs_features = [
            math.sin(lst_rad),
            math.cos(lst_rad),
            self.tel_ra / 360.0,
            self.tel_dec / 90.0,
            self.weather_state,
        ]

        target_features = []
        for tgt in self.targets:
            airmass = self.calculate_airmass(tgt.ra_deg, tgt.dec_deg, self.lst_hours)
            cur_mag = self.eig_calc.calculate_apparent_magnitude(tgt)
            target_features.extend([
                tgt.ra_deg / 360.0,
                tgt.dec_deg / 90.0,
                min(cur_mag / 30.0, 1.0),
                min(tgt.decay_timescale_hours / 120.0, 1.0),
                min(tgt.time_since_trigger_hours / 48.0, 1.0),
                min(airmass / 3.0, 1.5),
                tgt.prior_variance,
                1.0 if tgt.is_too else 0.0,
            ])

        return np.array(obs_features + target_features, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one telescope scheduling decision step:
        action in [0, n_targets - 1]: Slew and expose target
        action == n_targets: Idle / wait 15 minutes
        """
        terminated = False
        truncated = False
        reward = 0.0
        info: Dict[str, Any] = {
            "action": action,
            "constraint_violation": False,
            "eig": 0.0,
            "slew_sec": 0.0,
        }

        # Weather transition: 95% keep clear, 5% degrade/recover
        if self.rng.uniform(0, 1) < 0.05:
            self.weather_state = 0.5 if self.weather_state == 1.0 else 1.0

        if action < self.n_targets:
            target = self.targets[action]
            slew_sec = self.calculate_slew_time_sec(target.ra_deg, target.dec_deg)
            elapsed_step_sec = slew_sec + self.exposure_time_sec

            # Advance pointing
            self.tel_ra = target.ra_deg
            self.tel_dec = target.dec_deg
            airmass = self.calculate_airmass(target.ra_deg, target.dec_deg, self.lst_hours)

            info["slew_sec"] = slew_sec
            info["airmass"] = airmass

            # Check constraint violations
            if airmass > self.max_airmass:
                # Severe constraint violation: observing below horizon / unacceptable airmass
                reward = -5.0
                info["constraint_violation"] = True
            elif self.weather_state < 0.2:
                # Dome closed
                reward = -5.0
                info["constraint_violation"] = True
            else:
                # Calculate Information Gain
                eig = self.eig_calc.evaluate_target_eig(
                    target, airmass, exposure_time_sec=self.exposure_time_sec
                )
                info["eig"] = eig

                # Slew penalty: small cost per minute of slewing to favor efficient geodesics
                slew_penalty = 0.05 * (slew_sec / 60.0)
                reward = eig - slew_penalty

                # Bayesian update on target
                cur_mag = self.eig_calc.calculate_apparent_magnitude(target)
                snr = self.eig_calc.compute_snr(cur_mag, airmass, self.exposure_time_sec)
                self.eig_calc.posterior_update(target, snr)

            # Advance time
            self.current_time_sec += elapsed_step_sec
            delta_hours = elapsed_step_sec / 3600.0
            self.lst_hours = (self.lst_hours + delta_hours) % 24.0

            # Advance all targets age
            for tgt in self.targets:
                tgt.time_since_trigger_hours += delta_hours

        else:
            # Idle action
            idle_sec = 900.0  # 15 minutes
            self.current_time_sec += idle_sec
            delta_hours = idle_sec / 3600.0
            self.lst_hours = (self.lst_hours + delta_hours) % 24.0
            for tgt in self.targets:
                tgt.time_since_trigger_hours += delta_hours
            reward = -0.1  # Minimal idle penalty to prevent inaction

        if self.current_time_sec >= self.night_duration_sec:
            terminated = True

        obs = self._get_observation()
        return obs, float(reward), terminated, truncated, info
