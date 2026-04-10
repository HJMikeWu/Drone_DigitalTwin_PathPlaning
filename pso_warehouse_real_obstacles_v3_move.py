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
"""

import os
import shutil
import signal
import subprocess
from datetime import datetime
import carb
import omni
from isaacsim import SimulationApp
import time

# 1. Initialize SimulationApp before importing extensions.
# Ensures that all required modules (e.g., Pegasus, Debug Draw) are loaded correctly.
CONFIG = {"renderer": "RayTracedLighting", "headless": False}
simulation_app = SimulationApp(CONFIG)

# Await simulation environment initialization
print("[INFO] Awaiting Isaac Sim application initialization.")
start_time = time.time()
while time.time() - start_time < 60:
    simulation_app.update()
    try:
        if hasattr(simulation_app, 'is_app_ready') and simulation_app.is_app_ready():
            print("[INFO] Isaac Sim application is fully initialized.")
            break
        elif hasattr(simulation_app, 'is_running') and simulation_app.is_running():
            print("[INFO] Isaac Sim application is running.")
            break
    except:
        pass
    time.sleep(0.5)

print("[INFO] Application startup phase completed. Proceeding with execution.")

# --- Import Core Modules After Simulation Initialization ---
import omni.timeline
from omni.isaac.core.world import World
import omni.usd
from pxr import Gf, UsdGeom

# Acquire the timeline that will be used to start/stop the simulation
timeline = omni.timeline.get_timeline_interface()

# Pegasus simulator imports
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

# Auxiliary scipy and numpy modules
import numpy as np
from scipy.spatial.transform import Rotation

# Enable debug draw
import omni.kit.app
import omni.kit.commands
import omni.ui as ui
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.util.debug_draw", True)
from isaacsim.util.debug_draw import _debug_draw
draw_interface = _debug_draw.acquire_debug_draw_interface()
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import create_viewport_window, get_active_viewport_window


class PSOWarehouseRealObstacles:
    """
    Particle Swarm Optimization (PSO) Path Planning Simulation
    Integrates dynamic obstacle detection using Isaac Sim's USD context.

    Architecture Capabilities:
    - Automated geometric primitive extraction from the USD Stage.
    - Identification and segmentation of obstacles via precise Bounding Box evaluations.
    - Spatial filtering algorithms relying on volumetric dimensions and elevation constraints.
    - Procedurally robust detection invariant to hardcoded spatial limits.
    """

    def __init__(self):
        try:
            print(" Pegasus ...")
            from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
            from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
            from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

            self.timeline = timeline

            print(" Pegasus Interface...")
            self.pg = PegasusInterface()

            print(" World...")
            self.pg._world = World(**self.pg._world_settings)
            self.world = self.pg.world
            print("World ")

            print("Initializing...")
            self.pg.load_environment(SIMULATION_ENVIRONMENTS["Warehouse with Shelves"])
            print("")

            #  ——  USD Stage 
            print("Initializing...")
            for _ in range(30):
                simulation_app.update()
            print("")

            # Use the global draw interface
            self.draw = draw_interface
            print("Debug draw ")

            print("=" * 60)
            print(" USD Stage ...")
            print("=" * 60)
            self.obstacle_penalty = 500.0
            self.safety_margin = 1.0
            self.drone_body_radius = 0.275
            self.drone_height = 0.30
            self.goal_radius = self.drone_body_radius
            self.flight_z = 1.2
            self.planning_z_min = 0.8
            self.planning_z_max = 2.2
            self.obstacles = self.detect_real_obstacles()

            if not self.obstacles:
                print("WARNING: No obstacles detected. PSO will run in obstacle-free space.")
            else:
                print(f"Detected {len(self.obstacles)} obstacles.")

            print("Initializing...")
            self.start_pos, self.goal_pos = self.generate_random_positions()
            print(f"Generated start position: {self.start_pos}")
            print(f"Generated goal position: {self.goal_pos}")

            print(" Iris ...")
            import sys, os
            sys.path.insert(0, '/home/mirdc_ju/PegasusSimulator/examples/utils')
            from nonlinear_controller import NonlinearController
            
            config_multirotor = MultirotorConfig()
            
            #  Pegasus  NonlinearController
            # Core PSO optimization parameters
            controller = NonlinearController(
                trajectory_file=None,
                Kp=[15.0, 15.0, 15.0],  # Proportional gains (Kp)
                Kd=[10.0, 10.0, 10.0]
            )
            config_multirotor.backends = [controller]
            config_multirotor.init_pos = self.start_pos.tolist()

            self.drone = Multirotor(
                "/World/Iris_PSO",
                ROBOTS['Iris'],
                0,
                self.start_pos.tolist(),
                Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
                config=config_multirotor,
            )
            print("Iris ")

            print("Initializing...")
            for _ in range(10):
                self.world.step(render=False)
            print("")

            self.world.reset()
            print("")

            self.top_camera_path = "/World/TopViewCamera"
            self.follow_camera_path = "/World/DroneFollowCamera"
            self.top_view_window = None
            self.follow_view_window = None
            self.layout_applied = False
            self.top_view_locked = False
            self.viewport_retry_count = 0
            self.visualize_frame_count = 0
            self.setup_dual_viewports()

        except Exception as e:
            print(f"Error during initialization: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Core PSO optimization parameters
        self.num_particles = 60
        self.goal_tolerance = 0.05
        self.obs_bounds_xyz = np.empty((0, 6), dtype=float)
        self.refresh_obstacle_cache()

        # num_waypoints will be set by compute_dynamic_waypoints()
        # fitness_function uses this for particle dimension
        self.num_waypoints = 3
        self.particle_dim = self.num_waypoints * 3
        self.particles = np.zeros((self.num_particles, self.particle_dim))
        self.velocities = np.zeros((self.num_particles, self.particle_dim))
        self.personal_best = self.particles.copy()
        self.personal_best_fitness = np.full(self.num_particles, np.inf)

        self.global_best = self.particles[0].copy()
        self.global_best_fitness = self.fitness_function(self.global_best)

        self.global_best_path = [self.global_best.copy()]
        self.goal_reached = False
        self.path_visible = False
        self.pso_iteration = 0
        self.pso_max_iterations = 500

        # Video recording settings (for _move version)
        self.recording_enabled = True
        self.recording_process = None
        self.recording_output_path = ""
        self.recording_log_path = ""
        self.recording_log_file = None
        self.recording_round_index = 0

        self.compute_dynamic_waypoints()

        self.update_fitness()

        print("PSO initialization complete.")

    def ensure_camera_prim(self, camera_path):
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(camera_path).IsValid():
            UsdGeom.Camera.Define(stage, camera_path)
        return stage.GetPrimAtPath(camera_path)

    def setup_dual_viewports(self):
        """Set up dual viewports: top-down main view and third-person chase view."""
        try:
            self.ensure_camera_prim(self.top_camera_path)
            self.ensure_camera_prim(self.follow_camera_path)

            self.top_view_window = get_active_viewport_window()
            if self.top_view_window is None:
                print("WARNING: No active viewport window found.")
                return

            existing_windows = []
            try:
                from omni.kit.viewport.window import get_viewport_window_instances

                existing_windows = list(get_viewport_window_instances(None))
            except Exception:
                existing_windows = []

            self.follow_view_window = None
            for window in existing_windows:
                if getattr(window, "title", "") == "Drone Chase View":
                    self.follow_view_window = window
                    break

            if self.follow_view_window is None:
                self.follow_view_window = create_viewport_window("Drone Chase View")

            self.enforce_embedded_split_layout()

            self.set_top_view_camera()
            self.update_follow_camera_view()
            omni.kit.commands.execute(
                "SetViewportCamera",
                camera_path=self.follow_camera_path,
                viewport_api=self.follow_view_window.viewport_api,
            )
            print("✓  viewport: + ")
        except Exception as e:
            print(f"⚠  viewport :{e}")

    def enforce_embedded_split_layout(self):
        """Dock the chase viewport on the right with 50% split ratio."""
        if self.layout_applied:
            return
        if not getattr(self, "top_view_window", None) or not getattr(self, "follow_view_window", None):
            return

        try:
            target_name = getattr(self.top_view_window, "title", "Viewport")
            dock_target = ui.Workspace.get_window(target_name)
            if dock_target is None:
                dock_target = ui.Workspace.get_window("Viewport")
            if dock_target is not None:
                self.follow_view_window.dock_in(dock_target, ui.DockPosition.RIGHT, 0.5)
                self.layout_applied = True
        except Exception:
            pass

    def set_top_view_camera(self, force=False):
        """Set the main viewport camera to /OmniverseKit_Top when available."""
        if self.top_view_locked and not force:
            return
        try:
            stage = omni.usd.get_context().get_stage()
            top_prim_path = "/OmniverseKit_Top"
            if stage and stage.GetPrimAtPath(top_prim_path).IsValid():
                omni.kit.commands.execute(
                    "SetViewportCamera",
                    camera_path=top_prim_path,
                    viewport_api=self.top_view_window.viewport_api,
                )
            else:
                # Fallback when built-in top camera is unavailable.
                focus_xy = (self.start_pos[:2] + self.goal_pos[:2]) / 2.0
                eye = np.array([focus_xy[0], focus_xy[1], 42.0])
                target = np.array([focus_xy[0], focus_xy[1], 0.0])
                set_camera_view(
                    eye=eye,
                    target=target,
                    camera_prim_path=self.top_camera_path,
                    viewport_api=self.top_view_window.viewport_api,
                )
                omni.kit.commands.execute(
                    "SetViewportCamera",
                    camera_path=self.top_camera_path,
                    viewport_api=self.top_view_window.viewport_api,
                )
            self.top_view_locked = True
        except Exception as e:
            print(f"WARNING: Failed to set top view camera: {e}")

    def update_follow_camera_view(self):
        """Update third-person chase camera from current drone pose."""
        if not getattr(self, "follow_view_window", None):
            return

        try:
            current_pose = self.drone.get_world_pose()
            if not current_pose or current_pose[0] is None or current_pose[1] is None:
                return

            drone_pos = np.array(current_pose[0], dtype=float)
            drone_quat = current_pose[1]
            body_rotation = Rotation.from_quat([drone_quat[1], drone_quat[2], drone_quat[3], drone_quat[0]])

            # Keep a steep chase angle (roughly 60-80 degrees downward).
            #  eye->target  dx=1.9, dz=-3.75, arctan(3.75/1.9)=63°
            backward_offset = body_rotation.apply(np.array([-2.0, 0.0, 0.75]))
            target_offset = body_rotation.apply(np.array([3.0, 0.0, -1.0]))
            eye = drone_pos + backward_offset
            target = drone_pos + target_offset

            set_camera_view(
                eye=eye,
                target=target,
                camera_prim_path=self.follow_camera_path,
                viewport_api=self.follow_view_window.viewport_api,
            )
        except Exception:
            pass

    def _find_isaac_window_id(self):
        """Find Isaac Sim X11 window ID, or return None if unavailable."""
        search_cmds = [
            ["xdotool", "search", "--name", "Isaac Sim"],
            ["xdotool", "search", "--name", "isaac"],
            ["xdotool", "search", "--name", "Omniverse"],
            ["xdotool", "search", "--name", "omni"],
        ]

        for cmd in search_cmds:
            bin_path = shutil.which(cmd[0])
            if not bin_path:
                continue
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
                if not out:
                    continue
                wid_dec = out.splitlines()[-1].strip()
                if wid_dec.isdigit():
                    return hex(int(wid_dec))
            except Exception:
                continue
        return None

    def start_video_recording(self, round_index=1):
        """Start ffmpeg recording for Isaac Sim window on Linux/X11."""
        if not self.recording_enabled:
            return False

        if self.recording_process and self.recording_process.poll() is None:
            return True

        ffmpeg_bin = shutil.which("ffmpeg")
        xdotool_bin = shutil.which("xdotool")
        missing_tools = []
        if not ffmpeg_bin:
            missing_tools.append("ffmpeg")
        if not xdotool_bin:
            missing_tools.append("xdotool")
        if missing_tools:
            print(f"WARNING: Missing required tools for recording: {', '.join(missing_tools)}")
            return False

        display = os.environ.get("DISPLAY")
        if not display:
            print("WARNING: DISPLAY environment variable not set.")
            return False

        window_id = self._find_isaac_window_id()
        if not window_id:
            print("WARNING: Could not find Isaac Sim window ID (using xdotool).")
            return False

        movies_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "movies")
        os.makedirs(movies_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_output_path = os.path.join(movies_dir, f"pso_v3_move_round_{int(round_index):03d}_{ts}.mp4")
        self.recording_log_path = os.path.join(movies_dir, f"pso_v3_move_round_{int(round_index):03d}_{ts}.log")

        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "x11grab",
            "-framerate", "30",
            "-window_id", window_id,
            "-i", display,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            self.recording_output_path,
        ]

        try:
            self.recording_log_file = open(self.recording_log_path, "w", encoding="utf-8")
            self.recording_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=self.recording_log_file,
                preexec_fn=os.setsid,
            )
            time.sleep(0.2)
            if self.recording_process.poll() is not None:
                print(f"WARNING: Recording process failed, check log: {self.recording_log_path}")
                self.stop_video_recording()
                return False
            print(f"✓  {int(round_index)} :{self.recording_output_path}")
            return True
        except Exception as e:
            print(f"WARNING: Failed to start video recording: {e}")
            self.recording_process = None
            self.recording_output_path = ""
            if self.recording_log_file:
                try:
                    self.recording_log_file.close()
                except Exception:
                    pass
                self.recording_log_file = None
            return False

    def stop_video_recording(self):
        """Stop ffmpeg recording and finalize output file."""
        if not self.recording_process:
            return

        try:
            if self.recording_process.poll() is None:
                os.killpg(os.getpgid(self.recording_process.pid), signal.SIGINT)
                self.recording_process.wait(timeout=5)
        except Exception:
            try:
                self.recording_process.terminate()
                self.recording_process.wait(timeout=3)
            except Exception:
                try:
                    self.recording_process.kill()
                except Exception:
                    pass
        finally:
            self.recording_process = None
            if self.recording_log_file:
                try:
                    self.recording_log_file.close()
                except Exception:
                    pass
                self.recording_log_file = None

        if self.recording_output_path:
            print(f"✓ :{self.recording_output_path}")

    def handle_manual_close(self):
        """Handle manual Isaac Sim window close by saving active recording."""
        if self.recording_process:
            print("\nINFO: Isaac Sim window closed, saving recording.")
            self.stop_video_recording()

    def generate_random_positions(self):
        """Generate collision-free start/goal positions from detected workspace bounds."""
        fixed_z = 1.2

        # Infer usable bounds from detected obstacle set when available.
        if getattr(self, 'obstacles', None):
            # Estimate interior flyable region from wall obstacle AABBs.
            wall_obs = [
                obs for obs in self.obstacles
                if obs[5] >= fixed_z
            ]
            if wall_obs:
                # X: [wall_x_max_left_side, wall_x_min_right_side]
                # Y: [wall_y_max_bottom_side, wall_y_min_top_side]
                # Fallback to global obstacle envelope if interior estimate fails.
                xs_lo = sorted(obs[0] for obs in wall_obs)  # x_min 
                xs_hi = sorted(obs[3] for obs in wall_obs)  # x_max 
                ys_lo = sorted(obs[1] for obs in wall_obs)  # y_min 
                ys_hi = sorted(obs[4] for obs in wall_obs)  # y_max

                # X-axis: from x_max (20%) to x_min (80%)
                p20_x = xs_hi[int(len(xs_hi) * 0.20)]
                p80_x = xs_lo[int(len(xs_lo) * 0.80)]
                p20_y = ys_hi[int(len(ys_hi) * 0.20)]
                p80_y = ys_lo[int(len(ys_lo) * 0.80)]

                if p80_x > p20_x and p80_y > p20_y:
                    min_x, max_x = p20_x + 0.3, p80_x - 0.3
                    min_y, max_y = p20_y + 0.3, p80_y - 0.3
                    print("Using interior flyable region from wall obstacles.")
                else:
                    # Fallback to global obstacle envelope.
                    all_x_min = min(obs[0] for obs in self.obstacles)
                    all_y_min = min(obs[1] for obs in self.obstacles)
                    all_x_max = max(obs[3] for obs in self.obstacles)
                    all_y_max = max(obs[4] for obs in self.obstacles)
                    min_x = all_x_min + 0.8
                    max_x = all_x_max - 0.8
                    min_y = all_y_min + 0.8
                    max_y = all_y_max - 0.8
                    print("Using global obstacle envelope as fallback.")
            else:
                min_x, max_x = -20.0, 20.0
                min_y, max_y = -20.0, 20.0
                print("Using default envelope for wall obstacles.")
        else:
            # Default envelope when no obstacle scan is available.
            min_x, max_x = -20.0, 20.0
            min_y, max_y = -20.0, 20.0
            print("Using default envelope when no obstacles available.")

        self.space_min_x = min_x
        self.space_max_x = max_x
        self.space_min_y = min_y
        self.space_max_y = max_y

        print(f": X:[{min_x:.1f}, {max_x:.1f}], Y:[{min_y:.1f}, {max_y:.1f}]")

        max_attempts = 5000
        start_pos = None
        goal_pos = None

        for _ in range(max_attempts):
            pos = np.array([
                np.random.uniform(min_x, max_x),
                np.random.uniform(min_y, max_y),
                fixed_z
            ])
            if self.is_valid_position(pos):
                start_pos = pos
                break

        for _ in range(max_attempts):
            pos = np.array([
                np.random.uniform(min_x, max_x),
                np.random.uniform(min_y, max_y),
                fixed_z
            ])
            if not self.is_valid_position(pos):
                continue
            if start_pos is not None and np.linalg.norm(pos[:2] - start_pos[:2]) < 2.0:
                continue
            goal_pos = pos
            break

        if start_pos is None or goal_pos is None:
            print("WARNING: Failed to generate valid positions, using fallback.")
            start_pos = np.array([min_x + 1.0, min_y + 1.0, fixed_z])
            goal_pos = np.array([max_x - 1.0, max_y - 1.0, fixed_z])

        return start_pos, goal_pos

    def is_valid_position(self, pos, buffer=0.2):
        """Validate whether a candidate position is clear of relevant 3D obstacles."""
        drone_z = pos[2]
        h_half = self.drone_height / 2.0
        drone_z_min = drone_z - h_half
        drone_z_max = drone_z + h_half
        for obstacle in self.obstacles:
            obs_x, obs_y, obs_z_min, obs_x_max, obs_y_max, obs_z_max = obstacle
            # Only consider obstacles overlapping drone vertical envelope.
            if obs_z_max < drone_z_min or obs_z_min > drone_z_max:
                continue
            dx = max(obs_x - pos[0], pos[0] - obs_x_max, 0)
            dy = max(obs_y - pos[1], pos[1] - obs_y_max, 0)
            distance = np.sqrt(dx**2 + dy**2)
            if distance < self.safety_margin + buffer:
                return False
        return True

    def compute_dynamic_waypoints(self):
        """Select waypoint count from path complexity and initialize particle state."""
        straight_distance = np.linalg.norm(self.start_pos[:2] - self.goal_pos[:2])

        if straight_distance < 6.0:
            self.num_waypoints = 3
        elif straight_distance < 12.0:
            self.num_waypoints = 4
        else:
            self.num_waypoints = 5

        if len(self.obstacles) > 20 and straight_distance > 8.0:
            self.num_waypoints = min(self.num_waypoints + 1, 6)

        self.particle_dim = self.num_waypoints * 3
        self.particles = np.zeros((self.num_particles, self.particle_dim))
        self.velocities = np.zeros((self.num_particles, self.particle_dim))
        self.personal_best = np.zeros((self.num_particles, self.particle_dim))
        self.personal_best_fitness = np.full(self.num_particles, np.inf)

        z_fixed = float(self.start_pos[2])
        self.particles[:, 0::3] = np.random.uniform(self.space_min_x, self.space_max_x, size=(self.num_particles, self.num_waypoints))
        self.particles[:, 1::3] = np.random.uniform(self.space_min_y, self.space_max_y, size=(self.num_particles, self.num_waypoints))
        self.particles[:, 2::3] = np.random.uniform(self.planning_z_min, self.planning_z_max, size=(self.num_particles, self.num_waypoints))

        # Seed some particles near start-goal line and keep broad global exploration.
        guided_n = int(self.num_particles * 0.2)
        if guided_n > 0:
            alphas = np.linspace(1.0 / (self.num_waypoints + 1), self.num_waypoints / (self.num_waypoints + 1), self.num_waypoints)
            guide_xy = np.stack([
                (1.0 - alphas) * self.start_pos[0] + alphas * self.goal_pos[0],
                (1.0 - alphas) * self.start_pos[1] + alphas * self.goal_pos[1],
            ], axis=1)
            # Larger noise keeps seeded particles sufficiently diverse.
            space_w = max(self.space_max_x - self.space_min_x, self.space_max_y - self.space_min_y)
            noise_std = max(space_w * 0.25, 2.0)
            noise = np.random.normal(0.0, noise_std, size=(guided_n, self.num_waypoints, 2))
            guided_xy = guide_xy[np.newaxis, :, :] + noise
            guided_xy[..., 0] = np.clip(guided_xy[..., 0], self.space_min_x, self.space_max_x)
            guided_xy[..., 1] = np.clip(guided_xy[..., 1], self.space_min_y, self.space_max_y)
            self.particles[:guided_n, 0::3] = guided_xy[..., 0]
            self.particles[:guided_n, 1::3] = guided_xy[..., 1]
            guided_z = z_fixed + np.random.normal(0.0, 0.25, size=(guided_n, self.num_waypoints))
            guided_z = np.clip(guided_z, self.planning_z_min, self.planning_z_max)
            self.particles[:guided_n, 2::3] = guided_z

        self.personal_best = self.particles.copy()
        self.global_best = self.particles[0].copy()
        self.global_best_fitness = self.fitness_function(self.global_best)
        self.global_best_path = [self.start_pos.copy()] + list(self.global_best.reshape(self.num_waypoints, 3)) + [self.goal_pos.copy()]

    def refresh_obstacle_cache(self):
        """Cache inflated 3D obstacle bounds for fast PSO collision evaluation."""
        if not getattr(self, "obstacles", None):
            self.obs_bounds_xyz = np.empty((0, 6), dtype=float)
            return

        obs_arr_all = np.array(self.obstacles, dtype=float)
        z_relevant = (obs_arr_all[:, 5] >= self.planning_z_min) & (obs_arr_all[:, 2] <= self.planning_z_max)
        obs_arr = obs_arr_all[z_relevant]
        if len(obs_arr) == 0:
            self.obs_bounds_xyz = np.empty((0, 6), dtype=float)
            return

        inflate_xy = self.safety_margin + self.drone_body_radius
        inflate_z = self.drone_height / 2.0
        ox = obs_arr[:, 0] - inflate_xy
        oy = obs_arr[:, 1] - inflate_xy
        oz = obs_arr[:, 2] - inflate_z
        oxw = obs_arr[:, 3] + inflate_xy
        oyh = obs_arr[:, 4] + inflate_xy
        ozh = obs_arr[:, 5] + inflate_z
        self.obs_bounds_xyz = np.stack([ox, oy, oz, oxw, oyh, ozh], axis=1)

    def detect_real_obstacles(self):
        """Detect real obstacles recursively from USD Stage.
        Model shelf-like structures as 2D footprints with effectively infinite height
        to enforce aisle routing, while filtering irrelevant high-altitude geometry.
        """
        obstacles = []
        try:
            from pxr import Usd, UsdGeom, Gf
            stage = omni.usd.get_context().get_stage()
            if not stage:
                return obstacles

            all_prims = list(stage.Traverse())
            print(f"\nINFO: Scanning {len(all_prims)} primitives for obstacles...")

            purpose_tokens = [UsdGeom.Tokens.default_]
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purpose_tokens)

            allowed_keywords = {'wall', 'shelf', 'rack', 'cart', 'trolley', 'forklift'}
            shelf_like_keywords = {'shelf', 'rack'}
            reject_keywords = {
                'floor', 'ground', 'ceiling', 'light', 'lamp', 'truss', 'beam',
                'frame', 'roof', 'glass', 'window', 'door', 'camera', 'ceilinglight'
            }
            shelf_group_bbox = {}
            detected_count = 0

            for prim in all_prims:
                if prim.GetTypeName() not in ["Mesh", "Cube", "Cylinder"]:
                    continue

                prim_path_lower = str(prim.GetPath()).lower()
                if any(bad in prim_path_lower for bad in reject_keywords):
                    continue

                is_obstacle = False
                hit_keyword = None
                hit_group_path = None
                curr_p = prim
                while curr_p:
                    p_name_lower = curr_p.GetName().lower()
                    if any(bad in p_name_lower for bad in reject_keywords):
                        break
                    for kw in allowed_keywords:
                        if kw in p_name_lower:
                            is_obstacle = True
                            hit_keyword = kw
                            hit_group_path = str(curr_p.GetPath())
                            #  /World/layout ,
                            if hit_keyword in shelf_like_keywords:
                                anchor = curr_p
                                while anchor.GetParent() and str(anchor.GetParent().GetPath()) != "/World/layout":
                                    anchor = anchor.GetParent()
                                hit_group_path = str(anchor.GetPath())
                            break
                    if is_obstacle:
                        break
                    curr_p = curr_p.GetParent()

                if not is_obstacle:
                    continue

                try:
                    bbox = bbox_cache.ComputeWorldBound(prim)
                    if not bbox or bbox.ComputeAlignedRange().IsEmpty():
                        continue

                    bbox_range = bbox.ComputeAlignedRange()
                    min_pt, max_pt = bbox_range.GetMin(), bbox_range.GetMax()
                    x_min, y_min, z_min = float(min_pt[0]), float(min_pt[1]), float(min_pt[2])
                    x_max, y_max, z_max = float(max_pt[0]), float(max_pt[1]), float(max_pt[2])

                    width = x_max - x_min
                    height = y_max - y_min
                    z_height = z_max - z_min
                    if width < 0.05 or height < 0.05:
                        continue

                    planning_band_min = self.planning_z_min - self.drone_height
                    planning_band_max = self.planning_z_max + self.drone_height
                    if z_max < planning_band_min or z_min > planning_band_max:
                        continue

                    if hit_keyword in shelf_like_keywords:
                        # Group shelf footprints
                        prev = shelf_group_bbox.get(hit_group_path)
                        if prev is None:
                            shelf_group_bbox[hit_group_path] = [x_min, y_min, x_max, y_max]
                        else:
                            prev[0] = min(prev[0], x_min)
                            prev[1] = min(prev[1], y_min)
                            prev[2] = max(prev[2], x_max)
                            prev[3] = max(prev[3], y_max)
                    else:
                        obstacles.append([x_min, y_min, z_min, x_max, y_max, z_max])
                    detected_count += 1

                except Exception:
                    continue

            INF_Z = 1000.0
            for _, bbox in shelf_group_bbox.items():
                x_min, y_min, x_max, y_max = bbox
                obstacles.append([x_min, y_min, -INF_Z, x_max, y_max, INF_Z])

            print(f"INFO: Detected {detected_count} obstacle primitives, grouped into {len(obstacles)} obstacles.")

        except Exception as e:
            print(f"ERROR: Failed to detect real obstacles: {e}")
            import traceback
            traceback.print_exc()
            obstacles = []

        return obstacles

    def fitness_function(self, position):
        """Compute scalar fitness for a single particle."""
        return self.batch_fitness(position.reshape(1, -1))[0]

    def batch_fitness(self, particles_batch):
        """Vectorized batch fitness for all particles, including 3D waypoint search and penalties."""
        N = particles_batch.shape[0]
        nw = self.num_waypoints
        pts = particles_batch.reshape(N, nw, 3)
        start = np.tile(self.start_pos, (N, 1, 1))
        goal = np.tile(self.goal_pos, (N, 1, 1))
        path = np.concatenate([start, pts, goal], axis=1)

        segs = np.diff(path, axis=1)
        dist_total = np.sum(np.linalg.norm(segs, axis=2), axis=1)

        if len(self.obs_bounds_xyz) == 0:
            return dist_total

        ox = self.obs_bounds_xyz[:, 0]
        oy = self.obs_bounds_xyz[:, 1]
        oz = self.obs_bounds_xyz[:, 2]
        oxw = self.obs_bounds_xyz[:, 3]
        oyh = self.obs_bounds_xyz[:, 4]
        ozh = self.obs_bounds_xyz[:, 5]

        EPS = 1e-9
        S = nw + 1

        p1_xyz = path[:, :-1, :]
        p2_xyz = path[:, 1:, :]

        seg_dx = (p2_xyz[..., 0] - p1_xyz[..., 0])[..., np.newaxis]
        seg_dy = (p2_xyz[..., 1] - p1_xyz[..., 1])[..., np.newaxis]
        seg_dz = (p2_xyz[..., 2] - p1_xyz[..., 2])[..., np.newaxis]
        p1x = p1_xyz[..., 0:1]
        p1y = p1_xyz[..., 1:2]
        p1z = p1_xyz[..., 2:3]

        ox_ = ox[np.newaxis, np.newaxis, :]
        oy_ = oy[np.newaxis, np.newaxis, :]
        oz_ = oz[np.newaxis, np.newaxis, :]
        oxw_ = oxw[np.newaxis, np.newaxis, :]
        oyh_ = oyh[np.newaxis, np.newaxis, :]
        ozh_ = ozh[np.newaxis, np.newaxis, :]

        par_x = np.abs(seg_dx) < EPS
        inv_dx = np.where(par_x, 0.0, 1.0 / np.where(par_x, 1.0, seg_dx))
        tx1 = (ox_ - p1x) * inv_dx
        tx2 = (oxw_ - p1x) * inv_dx
        inside_x = (p1x >= ox_) & (p1x <= oxw_)
        tx_lo = np.where(par_x, np.where(inside_x, -np.inf, np.inf), np.minimum(tx1, tx2))
        tx_hi = np.where(par_x, np.where(inside_x, np.inf, -np.inf), np.maximum(tx1, tx2))

        par_y = np.abs(seg_dy) < EPS
        inv_dy = np.where(par_y, 0.0, 1.0 / np.where(par_y, 1.0, seg_dy))
        ty1 = (oy_ - p1y) * inv_dy
        ty2 = (oyh_ - p1y) * inv_dy
        inside_y = (p1y >= oy_) & (p1y <= oyh_)
        ty_lo = np.where(par_y, np.where(inside_y, -np.inf, np.inf), np.minimum(ty1, ty2))
        ty_hi = np.where(par_y, np.where(inside_y, np.inf, -np.inf), np.maximum(ty1, ty2))

        par_z = np.abs(seg_dz) < EPS
        inv_dz = np.where(par_z, 0.0, 1.0 / np.where(par_z, 1.0, seg_dz))
        tz1 = (oz_ - p1z) * inv_dz
        tz2 = (ozh_ - p1z) * inv_dz
        inside_z = (p1z >= oz_) & (p1z <= ozh_)
        tz_lo = np.where(par_z, np.where(inside_z, -np.inf, np.inf), np.minimum(tz1, tz2))
        tz_hi = np.where(par_z, np.where(inside_z, np.inf, -np.inf), np.maximum(tz1, tz2))

        t_enter = np.maximum(np.maximum(tx_lo, ty_lo), tz_lo)
        t_exit = np.minimum(np.minimum(tx_hi, ty_hi), tz_hi)
        intersects = (
            (t_enter <= t_exit + EPS) &
            (t_exit >= -EPS) &
            (t_enter <= 1.0 + EPS)
        )

        collision_count = np.sum(intersects, axis=(1, 2), dtype=float)
        collision_penalty = np.where(
            collision_count > 0,
            collision_count * self.obstacle_penalty * 2000.0 + 1e6,
            0.0,
        )

        K = 10
        t_vals = np.linspace(0.0, 1.0, K)
        p1_seg = path[:, :-1, :]
        p2_seg = path[:, 1:, :]
        sample_pts = (
            p1_seg[:, :, np.newaxis, :]
            + (p2_seg - p1_seg)[:, :, np.newaxis, :] * t_vals[np.newaxis, np.newaxis, :, np.newaxis]
        )
        sx = sample_pts[..., 0]
        sy = sample_pts[..., 1]
        sz = sample_pts[..., 2]

        sx_e = sx[..., np.newaxis]
        sy_e = sy[..., np.newaxis]
        sz_e = sz[..., np.newaxis]
        dx_d = np.maximum(np.maximum(ox - sx_e, sx_e - oxw), 0)
        dy_d = np.maximum(np.maximum(oy - sy_e, sy_e - oyh), 0)
        dz_d = np.maximum(np.maximum(oz - sz_e, sz_e - ozh), 0)
        dist_obs = np.sqrt(dx_d**2 + dy_d**2 + dz_d**2)
        min_dist = np.min(dist_obs, axis=2)

        margin = self.safety_margin
        within = min_dist < margin
        pf = np.where(within, (margin - min_dist) / margin, 0.0)
        obstacle_penalty_cont = np.sum(pf * self.obstacle_penalty * 2, axis=(1, 2))

        goal_ref = self.goal_pos[np.newaxis, np.newaxis, :]
        start_ref = self.start_pos[np.newaxis, np.newaxis, :]
        direct_len = max(np.linalg.norm(self.goal_pos - self.start_pos), 1e-6)

        dist_to_goal_nodes = np.linalg.norm(path - goal_ref, axis=2)  # (N, nw+2)
        away_steps = np.maximum(dist_to_goal_nodes[:, 1:] - dist_to_goal_nodes[:, :-1], 0.0)
        backtrack_penalty = np.sum(away_steps, axis=1) * 30.0

        # 2) Penalize waypoints too close to start
        wp = path[:, 1:-1, :]  # (N, nw, 3)
        dist_from_start = np.linalg.norm(wp - start_ref, axis=2)
        alphas = np.linspace(1.0 / (nw + 1), nw / (nw + 1), nw)
        target_progress = 0.35 * direct_len * alphas
        start_sticky_penalty = np.sum(
            np.maximum(target_progress[np.newaxis, :] - dist_from_start, 0.0),
            axis=1,
        ) * 12.0

        v_prev = path[:, 1:-1, :] - path[:, :-2, :]
        v_next = path[:, 2:, :] - path[:, 1:-1, :]
        norm_prev = np.linalg.norm(v_prev, axis=2)
        norm_next = np.linalg.norm(v_next, axis=2)
        cos_turn = np.sum(v_prev * v_next, axis=2) / (norm_prev * norm_next + 1e-6)
        turn_penalty = np.sum(np.maximum(0.0, 0.3 - cos_turn), axis=1) * 10.0

        climb_cost = np.sum(np.abs(segs[..., 2]), axis=1) * 8.0
        return (
            dist_total
            + collision_penalty
            + obstacle_penalty_cont
            + climb_cost
            + backtrack_penalty
            + start_sticky_penalty
            + turn_penalty
        )

    def update_fitness(self):
        """Main runtime loop for PSO optimization, validation, and flight execution."""
        fitness_all = self.batch_fitness(self.particles)  # (N,)

        improved = fitness_all < self.personal_best_fitness
        self.personal_best[improved] = self.particles[improved].copy()
        self.personal_best_fitness[improved] = fitness_all[improved]

        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            best_pts = self.global_best.reshape(self.num_waypoints, 3)
            self.global_best_path = [self.start_pos.copy()] + list(best_pts) + [self.goal_pos.copy()]

        best_pts = self.global_best.reshape(self.num_waypoints, 3)
        dist_to_goal = np.linalg.norm(best_pts[-1] - self.goal_pos)
        if dist_to_goal <= self.goal_tolerance:
            if not self.goal_reached:
                print(f"Goal reached within {self.goal_tolerance}m tolerance: {self.global_best}")
            self.goal_reached = True

    def update_particles(self):
        """Vectorized particle update including z-axis waypoint search."""
        N = self.num_particles

        # PSO parameter adaptation based on progress
        progress = min(1.0, self.pso_iteration / max(float(self.pso_max_iterations), 1.0))
        w = 0.9 - 0.5 * progress
        c1 = 2.4 - 1.2 * progress
        c2 = 1.0 + 1.2 * progress
        vmax = 0.8 - 0.4 * progress
        self.pso_iteration += 1

        r1 = np.random.random((N, self.particle_dim))
        r2 = np.random.random((N, self.particle_dim))

        cognitive = c1 * r1 * (self.personal_best - self.particles)
        social = c2 * r2 * (self.global_best - self.particles)
        self.velocities = w * self.velocities + cognitive + social

        bad_vel = ~np.isfinite(self.velocities).all(axis=1)
        if bad_vel.any():
            self.velocities[bad_vel] = np.random.uniform(-0.1, 0.1, (bad_vel.sum(), self.particle_dim))

        np.clip(self.velocities, -vmax, vmax, out=self.velocities)
        self.particles += self.velocities

        bad_pos = ~np.isfinite(self.particles).all(axis=1)
        for i in np.where(bad_pos)[0]:
            for j in range(self.num_waypoints):
                self.particles[i, j * 3] = np.random.uniform(self.space_min_x, self.space_max_x)
                self.particles[i, j * 3 + 1] = np.random.uniform(self.space_min_y, self.space_max_y)
                self.particles[i, j * 3 + 2] = np.random.uniform(self.planning_z_min, self.planning_z_max)

        x_idx = np.arange(0, self.particle_dim, 3)
        y_idx = np.arange(1, self.particle_dim, 3)
        z_idx = np.arange(2, self.particle_dim, 3)
        self.particles[:, x_idx] = np.clip(self.particles[:, x_idx], self.space_min_x, self.space_max_x)
        self.particles[:, y_idx] = np.clip(self.particles[:, y_idx], self.space_min_y, self.space_max_y)
        self.particles[:, z_idx] = np.clip(self.particles[:, z_idx], self.planning_z_min, self.planning_z_max)

        trial_fitness = self.batch_fitness(self.particles)
        colliding = trial_fitness > 1e6
        if colliding.any():
            idx = np.where(colliding)[0]
            alphas = np.linspace(1.0 / (self.num_waypoints + 1), self.num_waypoints / (self.num_waypoints + 1), self.num_waypoints)
            guide_x = (1.0 - alphas) * self.start_pos[0] + alphas * self.goal_pos[0]
            guide_y = (1.0 - alphas) * self.start_pos[1] + alphas * self.goal_pos[1]
            guide_z = (1.0 - alphas) * self.start_pos[2] + alphas * self.goal_pos[2]
            repair_std = max((self.space_max_x - self.space_min_x) * 0.25,
                             (self.space_max_y - self.space_min_y) * 0.25, 2.0)
            noise_x = np.random.normal(0.0, repair_std, size=(len(idx), self.num_waypoints))
            noise_y = np.random.normal(0.0, repair_std, size=(len(idx), self.num_waypoints))
            noise_z = np.random.normal(0.0, 0.3, size=(len(idx), self.num_waypoints))
            self.particles[idx[:, None], x_idx] = np.clip(guide_x + noise_x, self.space_min_x, self.space_max_x)
            self.particles[idx[:, None], y_idx] = np.clip(guide_y + noise_y, self.space_min_y, self.space_max_y)
            self.particles[idx[:, None], z_idx] = np.clip(guide_z + noise_z, self.planning_z_min, self.planning_z_max)

    def inject_diversity(self, fraction=0.2):
        """Initialize dynamic waypoint count and particle state from scene complexity."""
        count = max(1, int(self.num_particles * fraction))
        idx = np.random.choice(self.num_particles, size=count, replace=False)

        x_idx = np.arange(0, self.particle_dim, 3)
        y_idx = np.arange(1, self.particle_dim, 3)
        z_idx = np.arange(2, self.particle_dim, 3)

        self.particles[idx[:, None], x_idx] = np.random.uniform(
            self.space_min_x, self.space_max_x, size=(count, self.num_waypoints)
        )
        self.particles[idx[:, None], y_idx] = np.random.uniform(
            self.space_min_y, self.space_max_y, size=(count, self.num_waypoints)
        )
        self.particles[idx[:, None], z_idx] = np.random.uniform(
            self.planning_z_min, self.planning_z_max, size=(count, self.num_waypoints)
        )
        self.velocities[idx] = np.random.uniform(-0.2, 0.2, size=(count, self.particle_dim))
        self.personal_best_fitness[idx] = np.inf

    def visualize_pso_step(self):
        """Visualize current PSO state, obstacles, and best path."""
        self.visualize_frame_count += 1
        if self.viewport_retry_count < 10 and (self.visualize_frame_count % 30 == 0):
            if not self.layout_applied:
                self.enforce_embedded_split_layout()
            if not self.top_view_locked:
                self.set_top_view_camera(force=False)
            self.viewport_retry_count += 1
        self.update_follow_camera_view()
        self.draw.clear_points()

        drone_z = float(self.start_pos[2])
        for obstacle in self.obstacles:
            obs_x, obs_y, obs_z_min, obs_x_max, obs_y_max, obs_z_max = obstacle
            obs_width  = obs_x_max - obs_x
            obs_height = obs_y_max - obs_y

            is_blocking = (obs_z_max >= drone_z)
            frame_color = [1.0, 0.0, 0.0, 1.0] if is_blocking else [0.0, 0.5, 1.0, 0.5]

            obs_points = []
            obs_colors = []
            obs_sizes = []

            num_boundary_points = 20
            for i in range(num_boundary_points):
                x = obs_x + (obs_width * i / num_boundary_points)
                obs_points.append([x, obs_y, drone_z])
                obs_colors.append([0.5, 0.5, 0.5, 0.7])
                obs_sizes.append(8.0)
                obs_points.append([x, obs_y_max, drone_z])
                obs_colors.append([0.5, 0.5, 0.5, 0.7])
                obs_sizes.append(8.0)

            for i in range(num_boundary_points):
                y = obs_y + (obs_height * i / num_boundary_points)
                obs_points.append([obs_x, y, drone_z])
                obs_colors.append([0.5, 0.5, 0.5, 0.7])
                obs_sizes.append(8.0)
                obs_points.append([obs_x_max, y, drone_z])
                obs_colors.append([0.5, 0.5, 0.5, 0.7])
                obs_sizes.append(8.0)

            self.draw.draw_points(obs_points, obs_colors, obs_sizes)

            box_points = [
                [obs_x,     obs_y,     drone_z],
                [obs_x_max, obs_y,     drone_z],
                [obs_x_max, obs_y_max, drone_z],
                [obs_x,     obs_y_max, drone_z],
                [obs_x,     obs_y,     drone_z],
            ]
            line_colors = [frame_color] * len(box_points)
            line_sizes = [8.0] * len(box_points)
            self.draw.draw_points(box_points, line_colors, line_sizes)

        if not getattr(self, 'path_visible', False):
            colors = []
            sizes = []
            draw_pts = []
    
            for particle in self.particles:
                pts = particle.reshape(self.num_waypoints, 3)
                for pt in pts:
                    draw_pts.append(pt.tolist())
                    
                    min_distance = float('inf')
                    for obstacle in self.obstacles:
                        obs_x, obs_y, obs_z_min, obs_x_max, obs_y_max, obs_z_max = obstacle
                        h_half = self.drone_height / 2.0
                        if obs_z_max < (drone_z - h_half) or obs_z_min > (drone_z + h_half):
                            continue
                        dx = max(obs_x - pt[0], pt[0] - obs_x_max, 0)
                        dy = max(obs_y - pt[1], pt[1] - obs_y_max, 0)
                        distance = np.sqrt(dx**2 + dy**2)
                        min_distance = min(min_distance, distance)
        
                    if min_distance < self.safety_margin:
                        colors.append([1.0, 0.3, 0.3, 1.0])
                        sizes.append(15.0)
                    elif min_distance < self.safety_margin * 2:
                        colors.append([1.0, 0.6, 0.2, 1.0])
                        sizes.append(12.0)
                    else:
                        colors.append([0.0, 0.2, 0.8, 1.0])
                        sizes.append(10.0)
    
            self.draw.draw_points(draw_pts, colors, sizes)

        self.draw.draw_points([self.start_pos.tolist()], [[0, 1, 0, 1]], [30.0])
        self.draw.draw_points([self.goal_pos.tolist()], [[1, 0, 0, 1]], [30.0])
        self.draw_elastic_zone()

        if self.path_visible and len(self.global_best_path) >= 2:
            route_points = []
            for i in range(len(self.global_best_path) - 1):
                start_point = self.global_best_path[i]
                end_point = self.global_best_path[i + 1]
                num_interpolation_points = 8
                for j in range(num_interpolation_points + 1):
                    t = j / num_interpolation_points
                    interpolated_point = start_point + t * (end_point - start_point)
                    route_points.append(interpolated_point.tolist())

            route_colors = [[0.7, 0.0, 1.0, 0.9]] * len(route_points)
            route_sizes = [4.0] * len(route_points)
            self.draw.draw_points(route_points, route_colors, route_sizes)

        best_pts = self.global_best.reshape(self.num_waypoints, 3)
        self.draw.draw_points(
            best_pts.tolist(),
            [[0.8, 0.8, 0, 1]] * self.num_waypoints,
            [15.0] * self.num_waypoints
        )
        for wp in best_pts:
            self.draw_elastic_zone(
                center=wp.tolist(),
                color=[1.0, 0.9, 0.0, 0.25],
                radius=self.drone_body_radius
            )

    def draw_elastic_zone(self, center=None, color=None, radius=None):
        """Render a drone-size spherical point cloud as an elastic safety zone.

        Args:
            center: Sphere center [x, y, z], defaults to goal.
            color: RGBA color [r, g, b, a], defaults to translucent red.
            radius: Sphere radius, defaults to drone_body_radius.
        """
        if center is None:
            center = self.goal_pos
        if color  is None:
            color  = [1.0, 0.2, 0.2, 0.35]
        if radius is None:
            radius = self.drone_body_radius

        num_points = 40
        elastic_points = []

        # ( Fibonacci sphere )
        golden = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(num_points):
            y_off = 1.0 - (i / float(num_points - 1)) * 2.0
            r_xy  = np.sqrt(max(1.0 - y_off**2, 0.0))
            theta = golden * i
            x = center[0] + radius * r_xy * np.cos(theta)
            y = center[1] + radius * r_xy * np.sin(theta)
            z = center[2] + radius * y_off
            elastic_points.append([x, y, z])

        elastic_colors = [color] * len(elastic_points)
        elastic_sizes  = [5.0]  * len(elastic_points)
        self.draw.draw_points(elastic_points, elastic_colors, elastic_sizes)

    def reset_pso(self):
        """Reset PSO state while preserving current start and goal."""
        self.compute_dynamic_waypoints()
        self.pso_iteration = 0
        
        self.global_best = self.particles[0].copy()
        self.global_best_fitness = self.fitness_function(self.global_best)
        
        best_pts = self.global_best.reshape(self.num_waypoints, 3)
        self.global_best_path = [self.start_pos.copy()] + list(best_pts) + [self.goal_pos.copy()]
        self.goal_reached = False
        self.path_visible = False
        print("PSO state reset.")

    def check_path_collision(self):
        """Validate current best path with exact 3D slab-intersection checks.

        Returns:
            tuple[bool, int]: (has_collision, colliding_segment_count)
        """
        if not self.obstacles:
            return False, 0

        path = self.global_best_path   # list of np.ndarray  [x, y, z]
        obs_arr_all = np.array(self.obstacles)   # (M, 6): [xmin,ymin,zmin,xmax,ymax,zmax]

        # ══  AABB(3D): batch_fitness  ══════════
        inflate_xy = self.safety_margin + self.drone_body_radius
        inflate_z  = self.drone_height / 2.0

        ox_all  = obs_arr_all[:, 0] - inflate_xy
        oy_all  = obs_arr_all[:, 1] - inflate_xy
        oz_all  = obs_arr_all[:, 2] - inflate_z
        oxw_all = obs_arr_all[:, 3] + inflate_xy
        oyh_all = obs_arr_all[:, 4] + inflate_xy
        ozh_all = obs_arr_all[:, 5] + inflate_z

        EPS = 1e-9
        total_collision_segs = 0

        for i in range(len(path) - 1):
            p1 = np.array(path[i],   dtype=float)
            p2 = np.array(path[i+1], dtype=float)

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dz = p2[2] - p1[2]

            # ── X  Slab ──
            if abs(dx) < EPS:
                tx_lo = np.where((p1[0] >= ox_all) & (p1[0] <= oxw_all), -np.inf,  np.inf)
                tx_hi = np.where((p1[0] >= ox_all) & (p1[0] <= oxw_all),  np.inf, -np.inf)
            else:
                tx1 = (ox_all  - p1[0]) / dx
                tx2 = (oxw_all - p1[0]) / dx
                tx_lo = np.minimum(tx1, tx2)
                tx_hi = np.maximum(tx1, tx2)

            # ── Y  Slab ──
            if abs(dy) < EPS:
                ty_lo = np.where((p1[1] >= oy_all) & (p1[1] <= oyh_all), -np.inf,  np.inf)
                ty_hi = np.where((p1[1] >= oy_all) & (p1[1] <= oyh_all),  np.inf, -np.inf)
            else:
                ty1 = (oy_all  - p1[1]) / dy
                ty2 = (oyh_all - p1[1]) / dy
                ty_lo = np.minimum(ty1, ty2)
                ty_hi = np.maximum(ty1, ty2)

            # ── Z  Slab ──
            if abs(dz) < EPS:
                tz_lo = np.where((p1[2] >= oz_all) & (p1[2] <= ozh_all), -np.inf,  np.inf)
                tz_hi = np.where((p1[2] >= oz_all) & (p1[2] <= ozh_all),  np.inf, -np.inf)
            else:
                tz1 = (oz_all  - p1[2]) / dz
                tz2 = (ozh_all - p1[2]) / dz
                tz_lo = np.minimum(tz1, tz2)
                tz_hi = np.maximum(tz1, tz2)

            t_enter = np.maximum(np.maximum(tx_lo, ty_lo), tz_lo)
            t_exit  = np.minimum(np.minimum(tx_hi, ty_hi), tz_hi)

            hit = ((t_enter <= t_exit + EPS) &
                   (t_exit  >= -EPS)         &
                   (t_enter <= 1.0 + EPS))
            if hit.any():
                total_collision_segs += 1

        return total_collision_segs > 0, total_collision_segs

    def run(self):
        """Main runtime loop for PSO optimization, validation, and flight execution."""
        print("=" * 60)
        print("Starting PSO Path Planning Simulation!")
        print(f"Environment: Warehouse with Shelves")
        print(f"Obstacles detected: {len(self.obstacles)}")
        print(f"PSO Configuration: {self.num_particles} particles, max 1000 iterations")
        print(f"Path: {self.start_pos} -> {self.goal_pos}")
        print(f"Planning Z-range: [{self.planning_z_min:.2f}, {self.planning_z_max:.2f}]")
        print(f"Goal tolerance: {self.goal_tolerance}m")
        print("Press Play to start simulation.")
        print("=" * 60)

        simulation_count = 0

        while simulation_app.is_running():
            simulation_count += 1
            print(f"\n=== Simulation Round {simulation_count} ===")

            # Start PSO optimization
            self.stop_video_recording()
            self.start_video_recording(simulation_count)

            self.reset_pso()
            self.timeline.pause()  # Pause timeline for PSO optimization
            print("PSO optimization paused.")

            step_count = 0
            max_steps = 500
            self.pso_max_iterations = max_steps
            visualize_interval = 40
            update_interval = 3
            best_checkpoint = np.inf
            stagnant_count = 0

            try:
                while simulation_app.is_running() and step_count < max_steps:
                    # ===== PSO  =====
                    if step_count % update_interval == 0:
                        self.update_particles()
                        self.update_fitness()

                    if step_count % visualize_interval == 0:
                        self.visualize_pso_step()
                        simulation_app.update()

                    if step_count % 100 == 0:
                        dist_to_goal = np.linalg.norm(self.global_best.reshape(self.num_waypoints, 3)[-1] - self.goal_pos)
                        print(f"PSO Iteration: {step_count}, Best Fitness: {self.global_best_fitness:.3f}, Distance to Goal: {dist_to_goal:.3f}m")

                    if self.goal_reached:
                        print("Goal reached.")
                        break

                    if step_count % 50 == 0:
                        simulation_app.update()

                    if step_count % 50 == 0:
                        if self.global_best_fitness < best_checkpoint - 1e-3:
                            best_checkpoint = self.global_best_fitness
                            stagnant_count = 0
                        else:
                            stagnant_count += 1
                        if stagnant_count >= 2 and stagnant_count < 4:
                            self.inject_diversity(fraction=0.2)
                            print("PSO injecting diversity, 20% particles randomized.")
                        if stagnant_count >= 4:
                            print("PSO optimization stagnant, terminating.")
                            break

                    step_count += 1

                    if self.goal_reached:
                        print(f"Goal reached at iteration {step_count}, fitness: {self.global_best_fitness:.3f}")
                        break

                if not simulation_app.is_running():
                    self.handle_manual_close()
                    break

                self.path_visible = True
                self.visualize_pso_step()
                
                print(f"===  {simulation_count}  PSO  ===")
                print(f": {self.global_best_fitness:.3f}")

                # =====================================================
                # Validate path using Slab Method
                # Retry PSO if collisions detected (up to MAX_RETRY)
                # =====================================================
                MAX_RETRY = 5
                retry_count = 0
                has_collision, col_segs = self.check_path_collision()

                while has_collision and retry_count < MAX_RETRY:
                    retry_count += 1
                    print(f"\n⚠ []  {col_segs} ！")
                    print(f"  ->  {retry_count}/{MAX_RETRY} ()...")

                    self.compute_dynamic_waypoints()
                    self.global_best = self.particles[0].copy()
                    self.global_best_fitness = self.fitness_function(self.global_best)

                    retry_steps = 500
                    for _ in range(retry_steps):
                        self.update_particles()
                        self.update_fitness()
                        if not simulation_app.is_running():
                            break
                        if _ % 50 == 0:
                            simulation_app.update()

                    best_pts = self.global_best.reshape(self.num_waypoints, 3)
                    self.global_best_path = (
                        [self.start_pos.copy()] + list(best_pts) + [self.goal_pos.copy()]
                    )
                    self.path_visible = True
                    self.visualize_pso_step()

                    has_collision, col_segs = self.check_path_collision()
                    print(f"  : {self.global_best_fitness:.3f},"
                          f"collisions: {col_segs}")

                if has_collision:
                    print(f"\nWARNING: Path collision after {MAX_RETRY} retries, {col_segs} segments colliding.")
                    self.safety_margin += 0.2
                    self.refresh_obstacle_cache()
                    print(f"Increased safety margin to {self.safety_margin:.2f}.")
                    self.world.reset()
                    self.stop_video_recording()
                    continue
                else:
                    print(f"\nSUCCESS: Collision-free path found.")

                print("Initializing...")

                #   4.  decel_dist  0
                traj = []
                t_total = 0.0
                dt      = 0.02    # 50Hz control frequency
                v_max   = 2.0     # Maximum velocity (m/s)
                v_min   = 0.3     # Minimum velocity (m/s)
                a_max   = 1.0     # Maximum acceleration (m/s²)
                path_points = self.global_best_path
                n_pts = len(path_points)

                # --- Step 1: Compute segment distances ---
                seg_dists = []
                for i in range(n_pts - 1):
                    seg_dists.append(np.linalg.norm(path_points[i+1] - path_points[i]))
                total_path_len = sum(seg_dists)

                # / (v²=2as  -> s=v²/2a)
                acc_dist  = v_max**2 / (2.0 * a_max)   # Distance to accelerate from 0 to v_max
                decel_dist = acc_dist                    # Distance to decelerate from v_max to 0

                # --- Step 2: Compute waypoint speed limits ---
                # Based on turn sharpness (cosine of turn angle)
                waypoint_speed_limit = [v_max] * n_pts
                waypoint_speed_limit[0]  = 0.0
                waypoint_speed_limit[-1] = 0.0
                for i in range(1, n_pts - 1):
                    d_in  = path_points[i]   - path_points[i-1]
                    d_out = path_points[i+1] - path_points[i]
                    norm_in  = np.linalg.norm(d_in)
                    norm_out = np.linalg.norm(d_out)
                    if norm_in > 1e-6 and norm_out > 1e-6:
                        cos_a = np.dot(d_in, d_out) / (norm_in * norm_out)
                        cos_a = np.clip(cos_a, -1.0, 1.0)
                        # cos_a = 1 (straight) -> v_max; cos_a = -1 (U-turn) -> v_min
                        turn_factor = (cos_a + 1.0) / 2.0
                        waypoint_speed_limit[i] = v_min + turn_factor * (v_max - v_min)

                # --- Step 3: Generate trajectory with dt ---
                cum_dist = [0.0]
                for d in seg_dists:
                    cum_dist.append(cum_dist[-1] + d)

                def speed_profile(s):
                    """Trapezoidal speed profile as a function of path arc-length s."""
                    if s < acc_dist:
                        return max(v_min, np.sqrt(2.0 * a_max * s))
                    remaining = total_path_len - s
                    if remaining < decel_dist:
                        return max(0.0, np.sqrt(2.0 * a_max * remaining))
                    return v_max

                p0 = path_points[0]
                yaw = 0.0
                traj.append([0.0, p0[0], p0[1], p0[2], 0, 0, 0, 0,0,0, 0,0,0, yaw, 0.0])

                for i in range(n_pts - 1):
                    p_start = path_points[i]
                    p_end   = path_points[i + 1]
                    seg_len = seg_dists[i]
                    if seg_len < 1e-6:
                        continue

                    seg_dir = (p_end - p_start) / seg_len
                    yaw = np.arctan2(seg_dir[1], seg_dir[0])

                    # Generate trajectory points with dt
                    s_seg = 0.0
                    while s_seg < seg_len:
                        s_global = cum_dist[i] + s_seg

                        v_trap = speed_profile(s_global)

                        alpha   = s_seg / seg_len
                        v_limit = (1 - alpha) * waypoint_speed_limit[i] + alpha * waypoint_speed_limit[i+1]
                        v_limit = max(v_limit, 0.01)

                        v_now = min(v_trap, v_limit)
                        v_now = max(v_now, 0.0)

                        p_now = p_start + seg_dir * s_seg
                        vel_vec = seg_dir * v_now
                        t_total += dt
                        row = [t_total,
                               p_now[0], p_now[1], p_now[2],
                               vel_vec[0], vel_vec[1], vel_vec[2],
                               0, 0, 0, 0, 0, 0,
                               yaw, 0.0]
                        traj.append(row)

                        s_seg += v_now * dt if v_now > 0.01 else dt * v_min

                    p_end_arr = np.array(p_end)
                    v_wp = waypoint_speed_limit[i+1]
                    traj.append([t_total,
                                 p_end_arr[0], p_end_arr[1], p_end_arr[2],
                                 seg_dir[0]*v_wp, seg_dir[1]*v_wp, seg_dir[2]*v_wp,
                                 0, 0, 0, 0, 0, 0,
                                 yaw, 0.0])

                last_p = path_points[-1]
                traj.append([t_total + 5.0,
                             last_p[0], last_p[1], last_p[2],
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             yaw, 0.0])
                traj.append([t_total + 15.0,
                             last_p[0], last_p[1], last_p[2],
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             yaw, 0.0])
                print(f"Trajectory generation complete: {len(traj)} control points, estimated flight time {t_total:.1f}s.")

                
                # NonlinearController requires time-reversed trajectory (flip axis=0)
                controller = self.drone._backends[0]
                controller.trajectory = controller.read_trajectory_from_csv(csv_path)
                controller.max_index, _ = controller.trajectory.shape
                controller.total_time = 0.0
                controller.index = 0
                controller.reveived_first_state = False
                
                print("Trajectory imported successfully. Drone is ready for flight.")
                
                self.world.reset()
                self.timeline.play()

                collision_occurred = False
                reached_goal = False
                while simulation_app.is_running():
                    # Note: PSO optimization is paused during flight execution
                    self.visualize_pso_step()
                    self.world.step(render=True)
                    self.update_follow_camera_view()

                    # get_world_pose() returns (position, [qw, qx, qy, qz])
                    current_pose = self.drone.get_world_pose()
                    if current_pose and current_pose[0] is not None:
                        drone_pos = current_pose[0]
                        drone_quat = current_pose[1]
                        
                        #  [qw, qx, qy, qz]  Euler Angles 
                        from scipy.spatial.transform import Rotation
                        r = Rotation.from_quat([drone_quat[1], drone_quat[2], drone_quat[3], drone_quat[0]])
                        euler_angles = r.as_euler('xyz', degrees=True)
                        roll, pitch = euler_angles[0], euler_angles[1]
                        
                        # Check for collision: excessive tilt (>60°) or too low (Z<0.3m)
                        if abs(roll) > 60 or abs(pitch) > 60 or drone_pos[2] < 0.3:
                            print(f"\nCOLLISION DETECTED! (Roll: {roll:.1f}°, Pitch: {pitch:.1f}°, Z: {drone_pos[2]:.2f}m)")
                            collision_occurred = True

                        dist_to_goal_live = np.linalg.norm(np.array(drone_pos) - self.goal_pos)
                        arrive_threshold = max(self.goal_tolerance, self.goal_radius)
                        if dist_to_goal_live <= arrive_threshold:
                            print(f"\nSUCCESS: Goal reached (distance: {dist_to_goal_live:.3f}m, threshold: {arrive_threshold:.3f}m)")
                            reached_goal = True
                            self.timeline.stop()
                            self.stop_video_recording()
                            break
                            
                        import time
                        time.sleep(0.01)
                                
                    if collision_occurred:
                        print(f",...")
                        self.safety_margin += 0.2
                        self.refresh_obstacle_cache()
                        print(f"Updated safety margin to: {self.safety_margin:.2f}")
                        for _ in range(100):
                            self.world.step(render=True)
                        self.timeline.stop()
                        break

                if not simulation_app.is_running() and not reached_goal and not collision_occurred:
                    self.handle_manual_close()

                self.stop_video_recording()
                        
                # Handle collision: reset and retry PSO optimization
                if collision_occurred:
                    print("Collision occurred, resetting for PSO retry...")
                    self.world.reset()
                    continue  # Continue PSO optimization
                else:
                    if reached_goal:
                        print("Flight completed successfully.")
                        rerun_choice = "n"
                        while True:
                            try:
                                rerun_choice = input("？[Y/N]: ").strip().lower()
                            except EOFError:
                                rerun_choice = "n"
                            if rerun_choice in ("y", "yes", "n", "no"):
                                break
                            print(" Y  N.")

                        if rerun_choice in ("y", "yes"):
                            print(",.")
                            self.start_pos, self.goal_pos = self.generate_random_positions()
                            self.goal_reached = False
                            self.path_visible = False
                            self.refresh_obstacle_cache()
                            print(f"New start position: {self.start_pos}")
                            print(f"New goal position: {self.goal_pos}")

                            # Reset drone pose via API 
                            try:
                                self.timeline.stop()
                                self.world.reset()
                                if hasattr(self.drone, "set_world_pose"):
                                    self.drone.set_world_pose(
                                        self.start_pos.tolist(),
                                        Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
                                    )
                            except Exception as pose_err:
                                print(f"WARNING: Failed to reset drone pose: {pose_err}")

                            continue  # Continue to next simulation

                        print("Simulation completed. Exiting.")
                        while simulation_app.is_running():
                            self.visualize_pso_step()
                            self.update_follow_camera_view()
                            simulation_app.update()
                            time.sleep(0.01)
                    break

            except Exception as e:
                print(f"Error during initialization: {e}")
                import traceback
                traceback.print_exc()
                self.stop_video_recording()
                break

        self.stop_video_recording()
        carb.log_warn("PSOWarehouseRealObstacles Simulation App is closing.")
        simulation_app.close()


if __name__ == "__main__":
    sim = PSOWarehouseRealObstacles()
    sim.run()
