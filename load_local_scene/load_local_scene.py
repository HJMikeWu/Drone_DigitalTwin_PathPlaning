"""
=============================================================================
 COPYRIGHT NOTICE
=============================================================================

 @Author       : Hsien-Ju Wu
 @Affiliation  : Robotics Lab, National Kaohsiung University of Science and Technology (NKUST)

 This software is the proprietary work of Hsien-Ju Wu and is explicitly
 restricted. Open usage, modification, and distribution are exclusively
 granted to:

 1. Prof. Tien-Szu Pang and his research team
     (Robotics Lab, National Kaohsiung University of Science and Technology)
 2. Prof. Jeng-Shyang Pan and his research team
     (Nanjing University of Information Science and Technology)
 3. Prof. Shu-Chuan Chu and her research team
     (Nanjing University of Information Science and Technology)

 Any other usage without explicit authorization is strictly prohibited.
=============================================================================

Demo: Load a local USD file as the simulation scene in Isaac Sim.

Usage:
    $ISAACSIM_PYTHON load_local_scene.py

To change the scene, modify the USD_PATH variable below to point to your
local .usd file.
"""

import time
from pathlib import Path

# ── Step 1: Initialize SimulationApp BEFORE any omni/Isaac imports ──────────
from isaacsim import SimulationApp

simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": False})

# ── Step 2: Import omni modules (must be after SimulationApp) ─────────────
import omni.usd
import omni.timeline
from omni.isaac.core.world import World

# ── Step 3: Set the path to your local USD file ───────────────────────────
#  Use an absolute path, e.g.:
#    USD_PATH = "/home/user/assets/my_scene.usd"
#  Or a path relative to this script's location:
#    USD_PATH = str(Path(__file__).parent / "assets" / "my_scene.usd")
#
#  Isaac Sim also accepts omniverse:// nucleus paths and https:// URLs.

BASE_DIR = Path(__file__).resolve().parent
USD_PATH = str(BASE_DIR / "20250403.usd")


# ── Step 4: Wait for Isaac Sim to finish initializing ────────────────────
def wait_for_app_ready(timeout=60):
    print("[INFO] Waiting for Isaac Sim to initialize...")
    start = time.time()
    while time.time() - start < timeout:
        simulation_app.update()
        if simulation_app.is_running():
            print(f"[INFO] Isaac Sim ready. ({time.time()-start:.1f}s)")
            return
        time.sleep(0.3)
    print("[WARN] Timeout reached, continuing anyway.")


# ── Step 5: Wait for the USD stage to fully stream/load ───────────────────
def wait_for_stage_loaded(timeout=120):
    print("[INFO] Waiting for USD stage to load...")
    start = time.time()
    while time.time() - start < timeout:
        simulation_app.update()
        loading, progress, _ = omni.usd.get_context().get_stage_loading_status()
        if not loading:
            print(f"[INFO] Stage loaded. ({time.time()-start:.1f}s)")
            return
        time.sleep(0.1)
    print("[WARN] Stage load timeout, continuing anyway.")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    wait_for_app_ready()

    # Open the local USD file as the current stage
    # Relative-path example:
    # USD_PATH = str(BASE_DIR / "assets" / "warehouse.usd")
    print(f"[INFO] Opening USD: {USD_PATH}")
    omni.usd.get_context().open_stage(USD_PATH)

    # Wait for the stage to finish loading all assets
    wait_for_stage_loaded()

    # Create a World to manage physics and rendering
    world = World(stage_units_in_meters=1.0)
    world.reset()

    # Settle a few frames so the scene renders properly
    for _ in range(60):
        simulation_app.update()

    print("[INFO] Scene is live. Running for 30 seconds...")
    end_time = time.time() + 30
    while time.time() < end_time:
        simulation_app.update()
        world.step(render=True)

    print("[INFO] Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
