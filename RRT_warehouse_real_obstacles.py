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


class AIWarehouseRoutePlanner:
    """
    AI-Based Warehouse Route Planning Simulation
    Integrates real-scene obstacle extraction using Isaac Sim USD context.

    Architecture Capabilities:
    - Automated geometric primitive traversal from USD Stage.
    - Obstacle identification through precise Bounding Box evaluation.
    - Runtime free-space scanning for feasible initialization.
    - Collision-aware random start and goal generation within environment constraints.
    """

    def __init__(self):
        try:
            print("Loading Pegasus modules...")
            from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
            from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
            from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

            self.timeline = timeline

            print("Initializing Pegasus Interface...")
            self.pg = PegasusInterface()

            print("Initializing World...")
            self.pg._world = World(**self.pg._world_settings)
            self.world = self.pg.world
            print("World initialization completed.")

            print("Loading warehouse environment...")
            self.pg.load_environment(SIMULATION_ENVIRONMENTS["Warehouse with Shelves"])
            print("Warehouse environment loaded.")

            # Wait for full scene loading so USD Stage completes asset resolution.
            print("Waiting for full scene asset resolution...")
            for _ in range(30):
                simulation_app.update()
            print("Scene asset resolution completed.")

            # Use the global draw interface
            self.draw = draw_interface
            print("Debug draw initialized.")

            # ====== Core improvement: detect obstacles from real scene geometry. ======
            print("=" * 60)
            print("Starting real-scene obstacle detection from USD Stage...")
            print("=" * 60)
            self.obstacle_penalty = 500.0
            self.safety_margin = 1.0  # Increase safety margin to reserve turning inertia clearance.
            # Initialize drone volume parameters for start/goal validity checks.
            self.drone_body_radius = 0.275
            self.drone_height = 0.30
            self.goal_radius = self.drone_body_radius
            self.flight_z = 1.2
            self.planning_z_min = 0.8
            self.planning_z_max = 2.2
            self.obstacles = self.detect_real_obstacles()

            if not self.obstacles:
                print("WARNING: No obstacles detected. AI optimization will run in obstacle-free space.")
            else:
                print(f"Detected {len(self.obstacles)} real obstacles in total.")

            print("Scanning free space and generating random start/goal points...")
            self.start_pos, self.goal_pos = self.generate_random_positions()
            print(f"Generated start point: {self.start_pos}")
            print(f"Generated goal point: {self.goal_pos}")

            print("Creating Iris drone...")
            import sys, os
            sys.path.insert(0, '/home/mirdc_ju/PegasusSimulator/examples/utils')
            from nonlinear_controller import NonlinearController
            
            config_multirotor = MultirotorConfig()
            
            # Use the advanced NonlinearController from Pegasus examples.
            # Do not set trajectory initially; assign it after AI path computation.
            controller = NonlinearController(
                trajectory_file=None,
                Kp=[15.0, 15.0, 15.0],  # Increase position gain (Kp) for tighter trajectory tracking.
                Kd=[10.0, 10.0, 10.0]   # Tune derivative gain to avoid oscillation.
            )
            config_multirotor.backends = [controller]
            config_multirotor.init_pos = self.start_pos.tolist()

            self.drone = Multirotor(
                "/World/Iris_AI",
                ROBOTS['Iris'],
                0,
                self.start_pos.tolist(),
                Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
                config=config_multirotor,
            )
            print("Iris drone created successfully.")

            print("Waiting for drone initialization...")
            for _ in range(10):
                self.world.step(render=False)
            print("Drone initialization wait completed.")

            self.world.reset()
            print("Simulation environment reset completed.")

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

        # Core RRT optimization parameters.
        self.num_particles = 60
        self.goal_tolerance = 0.05
        self.obs_bounds_xyz = np.empty((0, 6), dtype=float)
        self.refresh_obstacle_cache()

        # Default waypoint count; refined later by compute_dynamic_waypoints().
        self.num_waypoints = 3
        self.particle_dim = self.num_waypoints * 3
        self.particles = np.zeros((self.num_particles, self.particle_dim))
        self.global_best = np.zeros(self.particle_dim)
        self.global_best_fitness = np.inf

        self.global_best_path = [self.global_best.copy()]
        self.goal_reached = False
        self.path_visible = False
        self.optim_iteration = 0
        self.optim_max_iterations = 500
        self.rrt_step_size = 1.0
        self.rrt_goal_bias = 0.20
        self.rrt_expand_per_iter = 30
        self.rrt_goal_threshold = 1.2
        self.rrt_rewire_radius = 2.2
        self.rrt_nodes = []
        self.rrt_parents = []
        self.rrt_costs = []
        self.rrt_best_goal_idx = None
        self.rrt_recent_new_nodes = []
        self.rrt_recent_new_edges = []
        self.rrt_added_last = 0
        self.rrt_rewire_count_last = 0

        # Automatic per-round recording state.
        self.recording_enabled = True
        self.recording_process = None
        self.recording_output_path = ""
        self.recording_log_path = ""
        self.recording_log_file = None
        self.recording_round_index = 0

        # Initialize planner state based on current start/goal.
        self.compute_dynamic_waypoints()

        print("Scene setup complete. Drone is ready. RRT path planning initialized.")

    def ensure_camera_prim(self, camera_path):
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(camera_path).IsValid():
            UsdGeom.Camera.Define(stage, camera_path)
        return stage.GetPrimAtPath(camera_path)

    def setup_dual_viewports(self):
        """Create dual viewports: top-down main view and third-person chase view."""
        try:
            self.ensure_camera_prim(self.top_camera_path)
            self.ensure_camera_prim(self.follow_camera_path)

            self.top_view_window = get_active_viewport_window()
            if self.top_view_window is None:
                print("WARNING: Main viewport not found. Skipping dual-view setup.")
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
            print("Dual viewport established: top view + third-person drone view.")
        except Exception as e:
            print(f"WARNING: Dual viewport initialization failed: {e}")

    def enforce_embedded_split_layout(self):
        """Dock chase viewport to the right side at 50% width in split mode."""
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
        """Bind the main viewport camera to /OmniverseKit_Top when available."""
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
                # fallback: Top ,
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
            print(f"WARNING: Failed to configure top-view camera: {e}")

    def update_follow_camera_view(self):
        """Update the third-person follow camera using current drone pose."""
        if not getattr(self, "follow_view_window", None):
            return

        try:
            current_pose = self.drone.get_world_pose()
            if not current_pose or current_pose[0] is None or current_pose[1] is None:
                return

            drone_pos = np.array(current_pose[0], dtype=float)
            drone_quat = current_pose[1]
            body_rotation = Rotation.from_quat([drone_quat[1], drone_quat[2], drone_quat[3], drone_quat[0]])

            # Camera angle range: 60~80 degrees
            # eye->target dx=1.9, dz=-3.75, arctan(3.75/1.9)=63°
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
        """Try to find Isaac Sim X11 window ID; return None if unavailable."""
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
        """Start ffmpeg recording of the Isaac Sim window on Linux/X11."""
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
            print(f"WARNING: Required tools not found: {', '.join(missing_tools)}")
            return False

        display = os.environ.get("DISPLAY")
        if not display:
            print("WARNING: DISPLAY environment variable not found; skipping auto recording.")
            return False

        window_id = self._find_isaac_window_id()
        if not window_id:
            print("WARNING: Isaac Sim window ID not found; skipping auto recording (requires xdotool and searchable window).")
            return False

        movies_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "movies")
        os.makedirs(movies_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_output_path = os.path.join(movies_dir, f"ai_route_round_{int(round_index):03d}_{ts}.mp4")
        self.recording_log_path = os.path.join(movies_dir, f"ai_route_round_{int(round_index):03d}_{ts}.log")

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
                print(f"WARNING: Recorder exited immediately after start. Check log: {self.recording_log_path}")
                self.stop_video_recording()
                return False
            print(f"Started window recording for round {int(round_index)}: {self.recording_output_path}")
            return True
        except Exception as e:
            print(f"WARNING: Failed to start recording: {e}")
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
        """Stop ffmpeg recording process and finalize output file."""
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
            print(f"Recording saved: {self.recording_output_path}")

    def handle_manual_close(self):
        """Handle manual Isaac Sim window close by stopping and saving recording."""
        if self.recording_process:
            print("\n[Manual close] Isaac Sim window closed. Stopping recording and saving file.")
            self.stop_video_recording()

    def generate_random_positions(self):
        """Generate collision-free start/goal positions from scanned workspace bounds."""
        fixed_z = 1.2

        # If obstacle scan exists, infer bounds from wall-like structures.
        if getattr(self, 'obstacles', None):
            # Strategy: estimate interior flyable region from wall AABBs.
            wall_obs = [
                obs for obs in self.obstacles
                if obs[5] >= fixed_z
            ]
            if wall_obs:
                # X: [wall_x_max_left_side, wall_x_min_right_side]
                # Y: [wall_y_max_bottom_side, wall_y_min_top_side]
                # Fall back to outer obstacle envelope if interior estimate fails.
                xs_lo = sorted(obs[0] for obs in wall_obs)  # x_min
                xs_hi = sorted(obs[3] for obs in wall_obs)  # x_max
                ys_lo = sorted(obs[1] for obs in wall_obs)  # y_min
                ys_hi = sorted(obs[4] for obs in wall_obs)  # y_max
                # X : x_max (20%) ~ x_min (80%)
                p20_x = xs_hi[int(len(xs_hi) * 0.20)]
                p80_x = xs_lo[int(len(xs_lo) * 0.80)]
                p20_y = ys_hi[int(len(ys_hi) * 0.20)]
                p80_y = ys_lo[int(len(ys_lo) * 0.80)]

                if p80_x > p20_x and p80_y > p20_y:
                    min_x, max_x = p20_x + 0.3, p80_x - 0.3
                    min_y, max_y = p20_y + 0.3, p80_y - 0.3
                    print("Particle sampling bounds established from interior wall-defined space.")
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
                    print("Failed to infer interior wall bounds; fallback to outer bounds of all obstacles.")
            else:
                min_x, max_x = -20.0, 20.0
                min_y, max_y = -20.0, 20.0
                print("No valid wall obstacles detected; using default sampling bounds.")
        else:
            # No obstacle scan available: use conservative defaults.
            min_x, max_x = -20.0, 20.0
            min_y, max_y = -20.0, 20.0
            print("No obstacles detected; using default sampling bounds.")

        self.space_min_x = min_x
        self.space_max_x = max_x
        self.space_min_y = min_y
        self.space_max_y = max_y

        print(f"Searching start/goal within bounds: X:[{min_x:.1f}, {max_x:.1f}], Y:[{min_y:.1f}, {max_y:.1f}]")

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
            print("WARNING: Could not find safe start or goal; using default fallback points.")
            start_pos = np.array([min_x + 1.0, min_y + 1.0, fixed_z])
            goal_pos = np.array([max_x - 1.0, max_y - 1.0, fixed_z])

        return start_pos, goal_pos

    def is_valid_position(self, pos, buffer=0.2):
        """Check whether position is safely separated from relevant 3D obstacles."""
        drone_z = pos[2]
        h_half = self.drone_height / 2.0
        drone_z_min = drone_z - h_half
        drone_z_max = drone_z + h_half
        for obstacle in self.obstacles:
            obs_x, obs_y, obs_z_min, obs_x_max, obs_y_max, obs_z_max = obstacle
            # Consider only obstacles overlapping the drone vertical envelope.
            if obs_z_max < drone_z_min or obs_z_min > drone_z_max:
                continue
            dx = max(obs_x - pos[0], pos[0] - obs_x_max, 0)
            dy = max(obs_y - pos[1], pos[1] - obs_y_max, 0)
            distance = np.sqrt(dx**2 + dy**2)
            if distance < self.safety_margin + buffer:
                return False
        return True

    def compute_dynamic_waypoints(self):
        """Choose waypoint count from scene complexity and initialize RRT planner state."""
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
        self.init_rrt_planner()

    def _path_to_fixed_waypoints(self, path_points):
        """Resample arbitrary path length into a fixed number of waypoints."""
        pts = np.array(path_points, dtype=float)
        if len(pts) < 2:
            return np.tile(self.start_pos, (self.num_waypoints, 1))

        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = np.sum(seg)
        if total < 1e-6:
            return np.tile(pts[0], (self.num_waypoints, 1))

        cum = np.concatenate([[0.0], np.cumsum(seg)])
        targets = np.linspace(total / (self.num_waypoints + 1), total * self.num_waypoints / (self.num_waypoints + 1), self.num_waypoints)
        out = np.zeros((self.num_waypoints, 3), dtype=float)
        for i, t in enumerate(targets):
            j = np.searchsorted(cum, t, side='right') - 1
            j = int(np.clip(j, 0, len(seg) - 1))
            local = (t - cum[j]) / max(seg[j], 1e-6)
            out[i] = pts[j] + local * (pts[j + 1] - pts[j])
        return out

    def smooth_path_with_geometric(self, path_points, passes=2, corner_weight=0.25, samples_per_segment=4):
        """Smooth a 3D path with a geometric corner-cutting scheme (Chaikin-style)."""
        pts = [np.array(p, dtype=float) for p in path_points]
        if len(pts) < 3:
            return pts

        w = float(np.clip(corner_weight, 0.05, 0.45))
        n_passes = int(np.clip(passes, 1, 4))
        smooth_pts = pts

        for _ in range(n_passes):
            refined = [smooth_pts[0].copy()]
            for i in range(1, len(smooth_pts) - 1):
                p_prev = smooth_pts[i - 1]
                p_curr = smooth_pts[i]
                p_next = smooth_pts[i + 1]

                q = (1.0 - w) * p_curr + w * p_prev
                r = (1.0 - w) * p_curr + w * p_next

                if np.linalg.norm(q - refined[-1]) > 1e-9:
                    refined.append(q)
                if np.linalg.norm(r - refined[-1]) > 1e-9:
                    refined.append(r)

            if np.linalg.norm(smooth_pts[-1] - refined[-1]) > 1e-9:
                refined.append(smooth_pts[-1].copy())
            smooth_pts = refined

        sps = int(np.clip(samples_per_segment, 1, 16))
        densified = [smooth_pts[0].copy()]
        for i in range(len(smooth_pts) - 1):
            p0 = smooth_pts[i]
            p1 = smooth_pts[i + 1]
            for k in range(1, sps + 1):
                t = k / float(sps)
                densified.append((1.0 - t) * p0 + t * p1)

        return densified

    def _polyline_collision_free(self, path_points):
        """Return True when all polyline segments are collision-free."""
        if len(path_points) < 2:
            return True
        for i in range(len(path_points) - 1):
            if not self._segment_collision_free(np.array(path_points[i], dtype=float), np.array(path_points[i + 1], dtype=float)):
                return False
        return True

    def _segment_collision_free(self, p1, p2):
        """3D segment-vs-inflated-AABB collision test."""
        if len(self.obs_bounds_xyz) == 0:
            return True

        obs = self.obs_bounds_xyz
        ox, oy, oz = obs[:, 0], obs[:, 1], obs[:, 2]
        oxw, oyh, ozh = obs[:, 3], obs[:, 4], obs[:, 5]
        dx, dy, dz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
        EPS = 1e-9

        if abs(dx) < EPS:
            tx_lo = np.where((p1[0] >= ox) & (p1[0] <= oxw), -np.inf, np.inf)
            tx_hi = np.where((p1[0] >= ox) & (p1[0] <= oxw), np.inf, -np.inf)
        else:
            tx1 = (ox - p1[0]) / dx
            tx2 = (oxw - p1[0]) / dx
            tx_lo = np.minimum(tx1, tx2)
            tx_hi = np.maximum(tx1, tx2)

        if abs(dy) < EPS:
            ty_lo = np.where((p1[1] >= oy) & (p1[1] <= oyh), -np.inf, np.inf)
            ty_hi = np.where((p1[1] >= oy) & (p1[1] <= oyh), np.inf, -np.inf)
        else:
            ty1 = (oy - p1[1]) / dy
            ty2 = (oyh - p1[1]) / dy
            ty_lo = np.minimum(ty1, ty2)
            ty_hi = np.maximum(ty1, ty2)

        if abs(dz) < EPS:
            tz_lo = np.where((p1[2] >= oz) & (p1[2] <= ozh), -np.inf, np.inf)
            tz_hi = np.where((p1[2] >= oz) & (p1[2] <= ozh), np.inf, -np.inf)
        else:
            tz1 = (oz - p1[2]) / dz
            tz2 = (ozh - p1[2]) / dz
            tz_lo = np.minimum(tz1, tz2)
            tz_hi = np.maximum(tz1, tz2)

        t_enter = np.maximum(np.maximum(tx_lo, ty_lo), tz_lo)
        t_exit = np.minimum(np.minimum(tx_hi, ty_hi), tz_hi)
        hit = (t_enter <= t_exit + EPS) & (t_exit >= -EPS) & (t_enter <= 1.0 + EPS)
        return not bool(np.any(hit))

    def init_rrt_planner(self):
        """Initialize RRT* planner state and seed an initial feasible/global-best path."""
        self.rrt_nodes = [self.start_pos.copy()]
        self.rrt_parents = [-1]
        self.rrt_costs = [0.0]
        self.rrt_best_goal_idx = None
        self.rrt_recent_new_nodes = []
        self.rrt_recent_new_edges = []
        self.rrt_added_last = 0
        self.rrt_rewire_count_last = 0
        self.optim_iteration = 0
        self.goal_reached = False

        # Seed with direct start->goal path when collision-free.
        if self._segment_collision_free(self.start_pos, self.goal_pos):
            seed_path = [self.start_pos.copy(), self.goal_pos.copy()]
            wp = self._path_to_fixed_waypoints(seed_path)
            self.global_best = wp.reshape(-1)
            self.global_best_fitness = self.fitness_function(self.global_best)
            self.global_best_path = seed_path
            self.goal_reached = True
            self.rrt_best_goal_idx = 0
        else:
            wp = self._path_to_fixed_waypoints([self.start_pos.copy(), self.goal_pos.copy()])
            self.global_best = wp.reshape(-1)
            self.global_best_fitness = self.fitness_function(self.global_best)
            self.global_best_path = [self.start_pos.copy()] + list(wp) + [self.goal_pos.copy()]

        self.particles[:, :] = self.global_best[np.newaxis, :]

    def _sample_rrt_target(self):
        if np.random.rand() < self.rrt_goal_bias:
            return self.goal_pos.copy()
        return np.array([
            np.random.uniform(self.space_min_x, self.space_max_x),
            np.random.uniform(self.space_min_y, self.space_max_y),
            np.random.uniform(self.planning_z_min, self.planning_z_max),
        ])

    def _nearest_rrt_index(self, point):
        nodes = np.array(self.rrt_nodes)
        d = np.linalg.norm(nodes - point[np.newaxis, :], axis=1)
        return int(np.argmin(d))

    def _neighbor_rrt_indices(self, point, radius):
        nodes = np.array(self.rrt_nodes)
        d = np.linalg.norm(nodes - point[np.newaxis, :], axis=1)
        return np.where(d <= radius)[0].tolist()

    def _propagate_cost_delta(self, root_idx, delta):
        """Propagate cost delta through descendants after a rewiring update."""
        if abs(delta) < 1e-9:
            return
        stack = [root_idx]
        while stack:
            cur = stack.pop()
            self.rrt_costs[cur] += delta
            children = [i for i, p in enumerate(self.rrt_parents) if p == cur]
            stack.extend(children)

    def _steer_towards(self, src, dst):
        vec = dst - src
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            return src.copy()
        step = min(self.rrt_step_size, dist)
        return src + (vec / dist) * step

    def _backtrack_rrt_path(self, end_idx):
        out = []
        cur = end_idx
        while cur >= 0:
            out.append(self.rrt_nodes[cur])
            cur = self.rrt_parents[cur]
        out.reverse()
        if np.linalg.norm(out[-1] - self.goal_pos) > 1e-6:
            out.append(self.goal_pos.copy())
        return out

    def update_neural_optimizer(self):
        """Run one AI planning iteration using RRT* expansion and rewiring."""
        if self.goal_reached:
            return

        self.rrt_recent_new_nodes = []
        self.rrt_recent_new_edges = []
        self.rrt_added_last = 0
        self.rrt_rewire_count_last = 0

        for _ in range(self.rrt_expand_per_iter):
            sample = self._sample_rrt_target()
            near_idx = self._nearest_rrt_index(sample)
            near = self.rrt_nodes[near_idx]
            new_node = self._steer_towards(near, sample)

            if not self._segment_collision_free(near, new_node):
                continue

            # RRT* Step 1: choose the minimum-cost feasible parent in neighborhood.
            neighbors = self._neighbor_rrt_indices(new_node, self.rrt_rewire_radius)
            if near_idx not in neighbors:
                neighbors.append(near_idx)

            best_parent = near_idx
            best_cost = self.rrt_costs[near_idx] + np.linalg.norm(new_node - near)
            for nb in neighbors:
                if nb == near_idx:
                    continue
                cand_parent = self.rrt_nodes[nb]
                if not self._segment_collision_free(cand_parent, new_node):
                    continue
                cand_cost = self.rrt_costs[nb] + np.linalg.norm(new_node - cand_parent)
                if cand_cost < best_cost:
                    best_cost = cand_cost
                    best_parent = nb

            self.rrt_nodes.append(new_node)
            self.rrt_parents.append(best_parent)
            self.rrt_costs.append(best_cost)
            new_idx = len(self.rrt_nodes) - 1
            self.rrt_added_last += 1
            self.rrt_recent_new_nodes.append(new_node.copy())
            self.rrt_recent_new_edges.append((self.rrt_nodes[best_parent].copy(), new_node.copy()))

            # RRT* Step 2: rewire nearby nodes through new_node when cheaper.
            for nb in neighbors:
                if nb == best_parent or nb == new_idx:
                    continue
                nb_node = self.rrt_nodes[nb]
                if not self._segment_collision_free(new_node, nb_node):
                    continue
                rewired_cost = self.rrt_costs[new_idx] + np.linalg.norm(nb_node - new_node)
                if rewired_cost + 1e-9 < self.rrt_costs[nb]:
                    old_cost = self.rrt_costs[nb]
                    old_parent = self.rrt_parents[nb]
                    self.rrt_parents[nb] = new_idx
                    self._propagate_cost_delta(nb, rewired_cost - old_cost)
                    self.rrt_rewire_count_last += 1
                    self.rrt_recent_new_edges.append((new_node.copy(), nb_node.copy()))

            # Try to connect to goal and keep only better global candidates.
            if np.linalg.norm(new_node - self.goal_pos) <= self.rrt_goal_threshold:
                if self._segment_collision_free(new_node, self.goal_pos):
                    # Update best only when candidate fitness improves.
                    rrt_path = self._backtrack_rrt_path(new_idx)
                    wp = self._path_to_fixed_waypoints(rrt_path)
                    candidate = wp.reshape(-1)
                    fit = self.fitness_function(candidate)
                    if fit < self.global_best_fitness:
                        self.global_best_fitness = fit
                        self.global_best = candidate.copy()
                        self.global_best_path = [np.array(p, dtype=float) for p in rrt_path]
                        self.rrt_best_goal_idx = new_idx
                    self.goal_reached = True
                    break

        self.optim_iteration += 1
        self.particles[:, :] = self.global_best[np.newaxis, :]

    def inject_diversity(self, fraction=0.25):
        """Inject random exploration nodes into the RRT tree during stagnation."""
        count = max(5, int(self.num_particles * fraction))
        for _ in range(count):
            sample = self._sample_rrt_target()
            near_idx = self._nearest_rrt_index(sample)
            near = self.rrt_nodes[near_idx]
            new_node = self._steer_towards(near, sample)
            if self._segment_collision_free(near, new_node):
                self.rrt_nodes.append(new_node)
                self.rrt_parents.append(near_idx)
                self.rrt_costs.append(self.rrt_costs[near_idx] + np.linalg.norm(new_node - near))

    def refresh_obstacle_cache(self):
        """Cache inflated 3D obstacle bounds for fast AI collision queries."""
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
        """
        Recursively detect planning obstacles from the USD stage.
        Shelf-like structures are merged as 2D footprints with effectively infinite height
        so the planner is forced to use aisle corridors, while irrelevant high-altitude
        structures are filtered out.
        """
        obstacles = []
        try:
            from pxr import Usd, UsdGeom, Gf
            stage = omni.usd.get_context().get_stage()
            if not stage:
                return obstacles

            all_prims = list(stage.Traverse())
            print(f"\n  [Obstacle scan] Found {len(all_prims)} prims. Starting recursive filtering...")

            purpose_tokens = [UsdGeom.Tokens.default_]
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purpose_tokens)

            # Whitelist obstacle classes; blacklist irrelevant scene geometry.
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
                            # For shelf-like groups, collapse under /World/layout direct children.
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

                    # Keep only obstacles overlapping the planning altitude band.
                    planning_band_min = self.planning_z_min - self.drone_height
                    planning_band_max = self.planning_z_max + self.drone_height
                    if z_max < planning_band_min or z_min > planning_band_max:
                        continue

                    if hit_keyword in shelf_like_keywords:
                        # Merge shelf group parts into one footprint.
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

            # Add merged shelf footprints as near-infinite-height walls.
            INF_Z = 1000.0
            for _, bbox in shelf_group_bbox.items():
                x_min, y_min, x_max, y_max = bbox
                obstacles.append([x_min, y_min, -INF_Z, x_max, y_max, INF_Z])

            print(f"  Detected {detected_count} obstacle parts; merged shelf groups into {len(obstacles)} planning obstacles.")

        except Exception as e:
            print(f"Obstacle detection error: {e}")
            import traceback
            traceback.print_exc()
            obstacles = []

        return obstacles

    def fitness_function(self, position):
        """Scalar fitness for one candidate path vector."""
        return self.batch_fitness(position.reshape(1, -1))[0]

    def batch_fitness(self, particles_batch):
        """Vectorized batch fitness over all particles.

        Includes 3D waypoint optimization, collision penalties, and path-quality terms.
        """
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

        # Additional path-quality penalties: backtracking, start-sticky, and turning.
        goal_ref = self.goal_pos[np.newaxis, np.newaxis, :]
        start_ref = self.start_pos[np.newaxis, np.newaxis, :]
        direct_len = max(np.linalg.norm(self.goal_pos - self.start_pos), 1e-6)

        # 1) Backtracking penalty: penalize steps moving away from the goal.
        dist_to_goal_nodes = np.linalg.norm(path - goal_ref, axis=2)  # (N, nw+2)
        away_steps = np.maximum(dist_to_goal_nodes[:, 1:] - dist_to_goal_nodes[:, :-1], 0.0)
        backtrack_penalty = np.sum(away_steps, axis=1) * 30.0

        # 2) Start-sticky penalty: enforce minimum progress from start across waypoints.
        wp = path[:, 1:-1, :]  # (N, nw, 3)
        dist_from_start = np.linalg.norm(wp - start_ref, axis=2)
        alphas = np.linspace(1.0 / (nw + 1), nw / (nw + 1), nw)
        target_progress = 0.35 * direct_len * alphas
        start_sticky_penalty = np.sum(
            np.maximum(target_progress[np.newaxis, :] - dist_from_start, 0.0),
            axis=1,
        ) * 12.0

        # 3) Turning penalty: discourage sharp zig-zag turns.
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
        """Legacy compatibility stub; AI variant evaluates fitness inside RRT* updates."""
        return

    def update_particles(self):
        """Legacy compatibility wrapper; delegate one expansion step to RRT*."""
        self.update_neural_optimizer()

    def _legacy_inject_diversity_disabled(self, fraction=0.2):
        """Legacy PSO hook retained for compatibility; unused in AI/RRT* mode."""
        return

    def visualize_pso_step(self):
        """Visualize current AI optimization state, planner graph, and safety overlays."""
        self.visualize_frame_count += 1
        # Retry viewport layout/camera lock at low frequency during startup.
        if self.viewport_retry_count < 10 and (self.visualize_frame_count % 30 == 0):
            if not self.layout_applied:
                self.enforce_embedded_split_layout()
            if not self.top_view_locked:
                self.set_top_view_camera(force=False)
            self.viewport_retry_count += 1
        self.update_follow_camera_view()
        self.draw.clear_points()

        # --- RRT*/RRT visualization: full tree + recent edges/rewires ---
        if hasattr(self, 'rrt_nodes') and len(self.rrt_nodes) > 1:
            # 1) Full tree skeleton.
            tree_pts = []
            tree_cols = []
            tree_sz = []
            edge_stride = max(1, len(self.rrt_nodes) // 1200)
            for i in range(1, len(self.rrt_nodes), edge_stride):
                p = self.rrt_parents[i]
                if p < 0:
                    continue
                a = np.array(self.rrt_nodes[p], dtype=float)
                b = np.array(self.rrt_nodes[i], dtype=float)
                for t in (0.0, 0.5, 1.0):
                    q = a + t * (b - a)
                    tree_pts.append(q.tolist())
                    tree_cols.append([0.55, 0.55, 0.55, 0.35])
                    tree_sz.append(2.5)
            if tree_pts:
                self.draw.draw_points(tree_pts, tree_cols, tree_sz)

            # 2) Recent added/rewired edges.
            recent_edge_pts = []
            recent_edge_cols = []
            recent_edge_sz = []
            for a, b in getattr(self, 'rrt_recent_new_edges', []):
                a = np.array(a, dtype=float)
                b = np.array(b, dtype=float)
                for t in (0.0, 0.33, 0.66, 1.0):
                    q = a + t * (b - a)
                    recent_edge_pts.append(q.tolist())
                    recent_edge_cols.append([0.2, 0.95, 1.0, 0.85])
                    recent_edge_sz.append(5.0)
            if recent_edge_pts:
                self.draw.draw_points(recent_edge_pts, recent_edge_cols, recent_edge_sz)

            # 3) Recently added nodes.
            if getattr(self, 'rrt_recent_new_nodes', []):
                self.draw.draw_points(
                    [np.array(n, dtype=float).tolist() for n in self.rrt_recent_new_nodes],
                    [[1.0, 0.95, 0.1, 1.0]] * len(self.rrt_recent_new_nodes),
                    [10.0] * len(self.rrt_recent_new_nodes),
                )

        # Draw obstacle envelopes and no-fly frames at current flight altitude.
        drone_z = float(self.start_pos[2])
        for obstacle in self.obstacles:
            obs_x, obs_y, obs_z_min, obs_x_max, obs_y_max, obs_z_max = obstacle
            obs_width  = obs_x_max - obs_x
            obs_height = obs_y_max - obs_y

            # Red frame for blocking obstacles, blue for fly-over obstacles.
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

            # No-fly frame points (red=blocking, blue=non-blocking).
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

        # Before final trajectory lock, draw candidate particles.
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

        # Draw start marker.
        self.draw.draw_points([self.start_pos.tolist()], [[0, 1, 0, 1]], [30.0])
        # Draw goal marker.
        self.draw.draw_points([self.goal_pos.tolist()], [[1, 0, 0, 1]], [30.0])
        self.draw_elastic_zone()

        # Draw dense final route points.
        if self.path_visible and len(self.global_best_path) >= 2:
            route_points = []
            for i in range(len(self.global_best_path) - 1):
                start_point = self.global_best_path[i]
                end_point = self.global_best_path[i + 1]
                num_interpolation_points = 8   # Reduce clutter while preserving continuity.
                for j in range(num_interpolation_points + 1):
                    t = j / num_interpolation_points
                    interpolated_point = start_point + t * (end_point - start_point)
                    route_points.append(interpolated_point.tolist())

            route_colors = [[0.7, 0.0, 1.0, 0.9]] * len(route_points)
            route_sizes = [4.0] * len(route_points)   # Reduced point size from thick legacy markers.
            self.draw.draw_points(route_points, route_colors, route_sizes)

        # Draw current best waypoints.
        best_pts = self.global_best.reshape(self.num_waypoints, 3)
        self.draw.draw_points(
            best_pts.tolist(),
            [[0.8, 0.8, 0, 1]] * self.num_waypoints,
            [15.0] * self.num_waypoints
        )
        # Draw a drone-size translucent zone at each waypoint.
        for wp in best_pts:
            self.draw_elastic_zone(
                center=wp.tolist(),
                color=[1.0, 0.9, 0.0, 0.25], 
                radius=self.drone_body_radius
            )

    def draw_elastic_zone(self, center=None, color=None, radius=None):
        """Render a drone-size spherical point cloud (elastic/safety zone).

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

        # Uniform sphere sampling using Fibonacci-sphere distribution.
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
        """Reset AI optimization state while preserving current start and goal."""
        self.compute_dynamic_waypoints()
        self.optim_iteration = 0

        best_pts = self.global_best.reshape(self.num_waypoints, 3)
        self.global_best_path = [self.start_pos.copy()] + list(best_pts) + [self.goal_pos.copy()]
        self.goal_reached = False
        self.path_visible = False
        print("AI optimization state reset. Ready for a new optimization round.")

    def save_convergence_curve(self, fitness_history, simulation_count, timestamp):
        """Save RST/AI fitness convergence data to CSV and generate a PNG plot."""
        if not fitness_history:
            return

        movies_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "movies")
        os.makedirs(movies_dir, exist_ok=True)

        csv_out = os.path.join(movies_dir, f"rst_fitness_round_{int(simulation_count):03d}_{timestamp}.csv")
        with open(csv_out, "w") as f:
            f.write("iteration,best_fitness\n")
            for it, fit in fitness_history:
                f.write(f"{it},{fit:.6f}\n")
        print(f"[Convergence] Fitness history saved: {csv_out}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            iterations = [r[0] for r in fitness_history]
            fitnesses  = [r[1] for r in fitness_history]

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(iterations, fitnesses, linewidth=1.5, color="darkorange")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Best Fitness")
            ax.set_title(f"RST/AI Fitness Convergence \u2014 Round {simulation_count}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            png_out = csv_out.replace(".csv", ".png")
            fig.savefig(png_out, dpi=150)
            plt.close(fig)
            print(f"[Convergence] Convergence plot saved: {png_out}")
        except Exception as e:
            print(f"[Convergence] WARNING: Could not generate plot: {e}")

    def check_path_collision(self):
        """Validate current best path with exact 3D slab-intersection checks.

        Returns:
            tuple[bool, int]: (has_collision, colliding_segment_count)
        """
        if not self.obstacles:
            return False, 0

        path = self.global_best_path   # list of np.ndarray [x, y, z]
        obs_arr_all = np.array(self.obstacles)   # (M, 6): [xmin,ymin,zmin,xmax,ymax,zmax]

        # ══ AABB(3D): batch_fitness ══════════
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

            # ── X Slab ──
            if abs(dx) < EPS:
                tx_lo = np.where((p1[0] >= ox_all) & (p1[0] <= oxw_all), -np.inf,  np.inf)
                tx_hi = np.where((p1[0] >= ox_all) & (p1[0] <= oxw_all),  np.inf, -np.inf)
            else:
                tx1 = (ox_all  - p1[0]) / dx
                tx2 = (oxw_all - p1[0]) / dx
                tx_lo = np.minimum(tx1, tx2)
                tx_hi = np.maximum(tx1, tx2)

            # ── Y Slab ──
            if abs(dy) < EPS:
                ty_lo = np.where((p1[1] >= oy_all) & (p1[1] <= oyh_all), -np.inf,  np.inf)
                ty_hi = np.where((p1[1] >= oy_all) & (p1[1] <= oyh_all),  np.inf, -np.inf)
            else:
                ty1 = (oy_all  - p1[1]) / dy
                ty2 = (oyh_all - p1[1]) / dy
                ty_lo = np.minimum(ty1, ty2)
                ty_hi = np.maximum(ty1, ty2)

            # ── Z Slab ──
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
        """Main runtime loop for planning, validation, and flight execution."""
        print("=" * 60)
        print("AI warehouse path planning simulation (RRT-optimized) is ready.")
        print(f"Scene: Warehouse with Shelves")
        print(f"Obstacle count: {len(self.obstacles)} (auto-detected from scene)")
        print(f"AI parameters: {self.num_particles} candidate networks per round, 500 iterations")
        print(f"Start: {self.start_pos}  ->  Goal: {self.goal_pos}")
        print(f"Planning altitude range: Z:[{self.planning_z_min:.2f}, {self.planning_z_max:.2f}]")
        print(f"Goal tolerance: {self.goal_tolerance}m")
        print("Press Play to start simulation, or close the window to exit.")
        print("=" * 60)

        simulation_count = 0

        while simulation_app.is_running():
            simulation_count += 1
            print(f"\n=== Starting simulation round {simulation_count} (AI path search) ===")

            # Per-round split recording: include planning + flight.
            self.stop_video_recording()
            self.start_video_recording(simulation_count)

            self.reset_pso()
            self.timeline.pause()
            print("Waiting for RRT path computation; physics engine is paused.")

            step_count = 0
            max_steps = 500
            self.optim_max_iterations = max_steps
            visualize_interval = 10
            update_interval = 1
            best_checkpoint = np.inf
            stagnant_count = 0
            fitness_history = []
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            try:
                while simulation_app.is_running() and step_count < max_steps:
                    # ===== RRT =====
                    if step_count % update_interval == 0:
                        self.update_neural_optimizer()
                        fitness_history.append((step_count, float(self.global_best_fitness)))

                    if step_count % visualize_interval == 0:
                        self.visualize_pso_step()
                        simulation_app.update()

                    if step_count % 20 == 0:
                        dist_to_goal = np.linalg.norm(self.global_best.reshape(self.num_waypoints, 3)[-1] - self.goal_pos)
                        node_count = len(self.rrt_nodes) if hasattr(self, 'rrt_nodes') else 0
                        print(f"AI Iteration: {step_count}, "
                              f"Nodes: {node_count}, "
                              f"Added: {getattr(self, 'rrt_added_last', 0)}, "
                              f"Rewired: {getattr(self, 'rrt_rewire_count_last', 0)}, "
                              f"Fitness: {self.global_best_fitness:.3f}, "
                              f"Dist to Goal: {dist_to_goal:.3f}m")

                    if self.goal_reached:
                        print("Goal reached with precision. Stopping current simulation round.")
                        break

                    if step_count % 50 == 0:
                        simulation_app.update()

                    # Check for stagnation in optimization progress
                    if step_count % 50 == 0:
                        if self.global_best_fitness < best_checkpoint - 1e-3:
                            best_checkpoint = self.global_best_fitness
                            stagnant_count = 0
                        else:
                            stagnant_count += 1
                        if stagnant_count >= 2 and stagnant_count < 4:
                            self.inject_diversity(fraction=0.2)
                            print("AI stagnation detected. Injecting perturbations to increase exploration.")
                        if stagnant_count >= 4:
                            print("No significant AI improvement for a long period; ending this round early.")
                            break

                    step_count += 1

                    if self.goal_reached:
                        print(f"Goal reached at iteration {step_count}, "
                              f"Final fitness: {self.global_best_fitness:.3f}")
                        break

                if not simulation_app.is_running():
                    self.handle_manual_close()
                    break

                # Set path visibility and visualize the current step
                self.path_visible = True
                self.visualize_pso_step()
                
                print(f"=== End of AI optimization round {simulation_count} ===")
                print(f"Final best fitness: {self.global_best_fitness:.3f}")
                self.save_convergence_curve(fitness_history, simulation_count, ts)

                # =====================================================
                # Path Validation: Slab Intersection Method -> Retry AI (MAX_RETRY)
                # Re-optimize if collisions detected
                # =====================================================
                MAX_RETRY = 5
                retry_count = 0
                has_collision, col_segs = self.check_path_collision()

                while has_collision and retry_count < MAX_RETRY:
                    retry_count += 1
                    print(f"\nWARNING [Path validation failed]: Best path still intersects obstacles on {col_segs} segment(s).")
                    print(f"  -> Re-search attempt {retry_count}/{MAX_RETRY} (re-initializing RRT tree)...")

                    # Reinitialize RRT planner
                    self.compute_dynamic_waypoints()

                    retry_steps = 500
                    for _ in range(retry_steps):
                        self.update_neural_optimizer()
                        if not simulation_app.is_running():
                            break
                        if _ % 50 == 0:
                            simulation_app.update()

                    # Update the best path and check for collisions
                    best_pts = self.global_best.reshape(self.num_waypoints, 3)
                    self.global_best_path = (
                        [self.start_pos.copy()] + list(best_pts) + [self.goal_pos.copy()]
                    )
                    self.path_visible = True
                    self.visualize_pso_step()

                    has_collision, col_segs = self.check_path_collision()
                    print(f"  Retry fitness: {self.global_best_fitness:.3f}, "
                          f"Collisions: {col_segs}")

                if has_collision:
                    print(f"\nWARNING: Path still collides after {MAX_RETRY} retries "
                          f"(collisions: {col_segs}).")
                    self.safety_margin += 0.2
                    self.refresh_obstacle_cache()
                    print(f"Increased safety margin to {self.safety_margin:.2f}.")
                    self.world.reset()
                    self.stop_video_recording()
                    continue
                else:
                    print(f"\n[Path validation passed] Path is fully collision-free. Preparing for takeoff.")

                # Apply geometric path smoothing and keep collision-safe fallback.
                raw_path_points = [np.array(p, dtype=float) for p in self.global_best_path]
                smoothed_path_points = self.smooth_path_with_geometric(
                    raw_path_points,
                    passes=2,
                    corner_weight=0.25,
                    samples_per_segment=4,
                )
                if self._polyline_collision_free(smoothed_path_points):
                    self.global_best_path = [np.array(p, dtype=float) for p in smoothed_path_points]
                    print(f"Geometric path smoothing applied (points: {len(raw_path_points)} -> {len(self.global_best_path)}).")
                else:
                    self.global_best_path = raw_path_points
                    print("WARNING: Collision introduced after geometric smoothing; reverted to original path.")

                print("Generating flight trajectory file and preparing drone flight...")

                # =====================================================
                # Flight Trajectory Generation Process
                # =====================================================
                # 1. Initialize trajectory parameters and compute path segments
                # 2. Generate trajectory points with time discretization
                # 3. Save trajectory to CSV file for controller
                # 4. Add hover points at goal position
                traj = []
                t_total = 0.0
                dt      = 0.02    # 50Hz control frequency
                v_max   = 3.2     # Maximum velocity (m/s)
                v_min   = 0.5     # Minimum velocity (m/s)
                a_max   = 2.0     # Maximum acceleration (m/s²)
                path_points = self.global_best_path
                n_pts = len(path_points)

                # --- Step 1: Compute segment distances ---
                seg_dists = []
                for i in range(n_pts - 1):
                    seg_dists.append(np.linalg.norm(path_points[i+1] - path_points[i]))
                total_path_len = sum(seg_dists)

                # Using kinematic equation: v² = 2as, so s = v²/(2a)
                acc_dist  = v_max**2 / (2.0 * a_max)   # Distance to accelerate from 0 to v_max
                decel_dist = acc_dist                    # Distance to decelerate from v_max to 0
                # --- Step 2: Compute waypoint speed limits based on turning angles ---
                # Compute cosine of turning angle
                waypoint_speed_limit = [v_max] * n_pts
                waypoint_speed_limit[0]  = 0.0   # Start point
                waypoint_speed_limit[-1] = 0.0   # End point
                for i in range(1, n_pts - 1):
                    d_in  = path_points[i]   - path_points[i-1]
                    d_out = path_points[i+1] - path_points[i]
                    norm_in  = np.linalg.norm(d_in)
                    norm_out = np.linalg.norm(d_out)
                    if norm_in > 1e-6 and norm_out > 1e-6:
                        cos_a = np.dot(d_in, d_out) / (norm_in * norm_out)
                        cos_a = np.clip(cos_a, -1.0, 1.0)
                        # cos_a = 1 (straight) -> v_max; cos_a = -1 (U-turn) -> v_min
                        turn_factor = (cos_a + 1.0) / 2.0           # 0~1
                        waypoint_speed_limit[i] = v_min + turn_factor * (v_max - v_min)

                # --- Step 3: Interpolate waypoints with time steps ---
                # Compute cumulative distances along the path
                cum_dist = [0.0]
                for d in seg_dists:
                    cum_dist.append(cum_dist[-1] + d)

                def speed_profile(s):
                    """Trapezoidal speed profile as a function of path arc length s."""
                    # Acceleration phase
                    if s < acc_dist:
                        return max(v_min, np.sqrt(2.0 * a_max * s))
                    # Deceleration phase
                    remaining = total_path_len - s
                    if remaining < decel_dist:
                        return max(0.0, np.sqrt(2.0 * a_max * remaining))
                    # Constant velocity phase
                    return v_max

                # Initialize trajectory with start point
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

                    # Generate trajectory points with time discretization
                    s_seg = 0.0   # Segment arc length
                    while s_seg < seg_len:
                        s_global = cum_dist[i] + s_seg   # Global arc length

                        # Compute trapezoidal speed profile
                        v_trap = speed_profile(s_global)

                        # Interpolate speed limits along segment
                        alpha   = s_seg / seg_len                                  # 0->1
                        v_limit = (1 - alpha) * waypoint_speed_limit[i] + alpha * waypoint_speed_limit[i+1]
                        v_limit = max(v_limit, 0.01)   # Minimum speed limit

                        v_now = min(v_trap, v_limit)
                        v_now = max(v_now, 0.0)

                        # Compute current position
                        p_now = p_start + seg_dir * s_seg
                        vel_vec = seg_dir * v_now
                        t_total += dt
                        row = [t_total,
                               p_now[0], p_now[1], p_now[2],
                               vel_vec[0], vel_vec[1], vel_vec[2],
                               0, 0, 0, 0, 0, 0,
                               yaw, 0.0]
                        traj.append(row)

                        # Update segment arc length
                        s_seg += v_now * dt if v_now > 0.01 else dt * v_min

                    # Append waypoint to trajectory
                    p_end_arr = np.array(p_end)
                    v_wp = waypoint_speed_limit[i+1]
                    traj.append([t_total,
                                 p_end_arr[0], p_end_arr[1], p_end_arr[2],
                                 seg_dir[0]*v_wp, seg_dir[1]*v_wp, seg_dir[2]*v_wp,
                                 0, 0, 0, 0, 0, 0,
                                 yaw, 0.0])

                # Add hover points at goal position
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

                
                # NonlinearController flip axis=0 ,
                traj_np = np.flip(np.array(traj), axis=0)
                csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_flight_trajectory.csv")
                np.savetxt(csv_path, traj_np, delimiter=',')
                
                # Load trajectory into controller and reset state
                controller = self.drone._backends[0]
                controller.trajectory = controller.read_trajectory_from_csv(csv_path)
                controller.max_index, _ = controller.trajectory.shape
                controller.total_time = 0.0
                controller.index = 0
                controller.reveived_first_state = False # Has not received the first state
                
                print("Trajectory imported successfully. Drone is about to take off...")
                
                # Resume simulation and reset the world state before takeoff.
                self.world.reset()
                # Force the drone pose to the current start point to avoid stale reset positions.
                try:
                    if hasattr(self.drone, "set_world_pose"):
                        self.drone.set_world_pose(
                            self.start_pos.tolist(),
                            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
                        )
                        # Let pose writes settle in simulation steps.
                        for _ in range(2):
                            self.world.step(render=False)
                    else:
                        print("WARNING: Drone object does not support set_world_pose; reset to new start may fail.")
                except Exception as pose_err:
                    print(f"WARNING: Failed to reset drone position before takeoff: {pose_err}")
                self.timeline.play()

                # Keep rendering and physics active while monitoring flight safety.
                collision_occurred = False
                reached_goal = False
                while simulation_app.is_running():
                    # Keep optimizer stepping disabled during execution flight.
                    self.visualize_pso_step()
                    self.world.step(render=True)
                    self.update_follow_camera_view()

                    # Monitor live flight status
                    # get_world_pose() returns (position, [qw, qx, qy, qz])
                    current_pose = self.drone.get_world_pose()
                    if current_pose and current_pose[0] is not None:
                        drone_pos = current_pose[0]
                        drone_quat = current_pose[1]
                        
                        # [qw, qx, qy, qz] Euler Angles                         from scipy.spatial.transform import Rotation
                        r = Rotation.from_quat([drone_quat[1], drone_quat[2], drone_quat[3], drone_quat[0]])
                        euler_angles = r.as_euler('xyz', degrees=True)
                        roll, pitch = euler_angles[0], euler_angles[1]
                        
                        # Check for collision: excessive tilt (>60°) or too low (Z<0.3m)
                        if abs(roll) > 60 or abs(pitch) > 60 or drone_pos[2] < 0.3:
                            print(f"\n[Crash warning] Physical drone crash detected! (Roll: {roll:.1f}°, Pitch: {pitch:.1f}°, Z: {drone_pos[2]:.2f}m)")
                            collision_occurred = True

                        # Check if drone has reached the goal position
                        dist_to_goal_live = np.linalg.norm(np.array(drone_pos) - self.goal_pos)
                        arrive_threshold = max(self.goal_tolerance, self.goal_radius)
                        if dist_to_goal_live <= arrive_threshold:
                            print(f"\n[Arrival command] Drone reached target point. (Distance: {dist_to_goal_live:.3f}m, Threshold: {arrive_threshold:.3f}m)")
                            reached_goal = True
                            self.timeline.stop()
                            self.stop_video_recording()
                            break
                            
                        # Brief pause for simulation timing
                        import time
                        time.sleep(0.01)
                                
                    if collision_occurred:
                        print(f"Interrupting current flight; increasing safety margin and recomputing path...")
                        self.safety_margin += 0.2
                        self.refresh_obstacle_cache()
                        print(f"New safety margin updated to: {self.safety_margin:.2f}")
                        # Step simulation and stop timeline
                        for _ in range(100):
                            self.world.step(render=True)
                        self.timeline.stop()
                        break

                if not simulation_app.is_running() and not reached_goal and not collision_occurred:
                    self.handle_manual_close()

                self.stop_video_recording()
                        
                # Handle collision: reset and retry PSO optimization
                if collision_occurred:
                    print("Resetting physics world and preparing to restart AI optimization...")
                    self.world.reset()
                    continue  # continue AI
                else:
                    if reached_goal:
                        print("Goal reached. Drone is holding position.")
                        rerun_choice = "n"
                        while True:
                            try:
                                rerun_choice = input("Run again with new start/goal positions? [Y/N]: ").strip().lower()
                            except EOFError:
                                rerun_choice = "n"
                            if rerun_choice in ("y", "yes", "n", "no"):
                                break
                            print("Invalid input. Please enter Y or N.")

                        if rerun_choice in ("y", "yes"):
                            print("Generating new start/goal positions and restarting simulation...")
                            self.start_pos, self.goal_pos = self.generate_random_positions()
                            self.goal_reached = False
                            self.path_visible = False
                            self.refresh_obstacle_cache()
                            print(f"New start position: {self.start_pos}")
                            print(f"New goal position: {self.goal_pos}")

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

                            continue  # Start next simulation round

                        print("Simulation completed. Keeping window open; close the window to exit.")
                        while simulation_app.is_running():
                            self.visualize_pso_step()
                            self.update_follow_camera_view()
                            simulation_app.update()
                            time.sleep(0.01)
                    break     # Exit the main simulation loop

            except Exception as e:
                print(f"Error occurred during simulation: {e}")
                import traceback
                traceback.print_exc()
                self.stop_video_recording()
                break

        self.stop_video_recording()
        carb.log_warn("AIWarehouseRoutePlanner Simulation App is closing.")
        simulation_app.close()


if __name__ == "__main__":
    sim = AIWarehouseRoutePlanner()
    sim.run()
