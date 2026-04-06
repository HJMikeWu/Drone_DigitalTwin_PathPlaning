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

# Acquire the application timeline interface for simulation control
timeline = omni.timeline.get_timeline_interface()

# Pegasus simulator APIs
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

# Scipy and Numpy utilities
import numpy as np
from scipy.spatial.transform import Rotation

# Enable extensions for visual debugging
import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.util.debug_draw", True)
from isaacsim.util.debug_draw import _debug_draw
draw_interface = _debug_draw.acquire_debug_draw_interface()


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
            print("[INFO] Loading Pegasus application modules.")
            from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
            from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
            from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

            self.timeline = timeline

            print("[INFO] Instantiating Pegasus Interface.")
            self.pg = PegasusInterface()

            print("[INFO] Initializing Isaac Sim World instance.")
            self.pg._world = World(**self.pg._world_settings)
            self.world = self.pg.world
            print("[INFO] World contextualization completed.")

            print("[INFO] Constructing the target simulation environment.")
            self.pg.load_environment(SIMULATION_ENVIRONMENTS["Warehouse with Shelves"])
            print("[INFO] Environment instantiation successful.")

            # Ensure asynchronous asset loading is complete
            print("[INFO] Awaiting USD static asset resolution.")
            for _ in range(30):
                simulation_app.update()
            print("[INFO] Core assets parsing resolved.")

            # Assign diagnostic drawing utility
            self.draw = draw_interface
            print("[INFO] Debug visualization utility initialized.")

            print("[INFO] Instantiating Multirotor representation (Iris).")
            import sys, os
            sys.path.insert(0, '/home/mirdc_ju/PegasusSimulator/examples/utils')
            from nonlinear_controller import NonlinearController
            
            config_multirotor = MultirotorConfig()
            
            # Incorporate NonlinearController formulation for tracking control.
            # Trajectory loading is deferred until successful PSO pathway convergence.
            controller = NonlinearController(
                trajectory_file=None,
                Kp=[15.0, 15.0, 15.0],  # Pronounced proportional feedback for aggressive trajectory tracking
                Kd=[10.0, 10.0, 10.0]   # Tailored derivative feedback suppressing transient oscillations
            )
            config_multirotor.backends = [controller]
            config_multirotor.init_pos = [-8.0, -6.0, 1.1]

            self.drone = Multirotor(
                "/World/Iris_PSO",
                ROBOTS['Iris'],
                0,
                [-8.0, -6.0, 1.1],
                Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
                config=config_multirotor,
            )
            print("[INFO] Multirotor asset successfully deployed.")

            print("[INFO] Initializing internal dynamic properties.")
            for _ in range(10):
                self.world.step(render=False)
            print("[INFO] Hardware and physics emulation synchronized.")

            self.world.reset()
            print("[INFO] Global simulation environments formally reset.")

        except Exception as e:
            print(f"[FATAL ERROR] Initialization exception encountered: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Core Particle Swarm Optimization hyperparameters
        self.num_particles = 30
        self.start_pos = np.array([-8.0, -6.0, 1.1])
        self.goal_pos = np.array([6.0, 4.0, 1.1])
        self.goal_tolerance = 0.01  # Convergence acceptance distance (meters)
        self.goal_radius = 0.3      # Visual elastic region radius

        # ====== Environmental Geometric Restraints Formulation ======
        print("=" * 60)
        print("[INFO] Triggering automated USD semantic spatial obstacle assessment.")
        print("=" * 60)
        self.obstacles = self.detect_real_obstacles()
        self.obstacle_penalty = 100.0
        self.safety_margin = 0.8  # Required clearance to compensate for kinematic curvature latency

        if not self.obstacles:
            print("[WARNING] Zero spatial rigid obstacles successfully parsed. PSO defaults to empty geometric domains.")
        else:
            print(f"[SUCCESS] Spatial constraint solver isolated {len(self.obstacles)} volumetric entities.")

        # Assign N-waypoint dimensional topological sequence for complex non-convex constraints
        self.num_waypoints = 3  
        self.particle_dim = self.num_waypoints * 3
        
        self.particles = np.random.uniform(low=-12.0, high=12.0, size=(self.num_particles, self.particle_dim))
        for j in range(self.num_waypoints):
            self.particles[:, j*3 + 2] = 1.1  # Constrain initial topology to a stable navigational horizontal plane
            
        self.velocities = np.random.uniform(-0.1, 0.1, size=(self.num_particles, self.particle_dim))
        self.personal_best = self.particles.copy()
        self.personal_best_fitness = np.full(self.num_particles, np.inf)

        self.global_best = self.particles[0].copy()
        self.global_best_fitness = self.fitness_function(self.global_best)

        self.global_best_path = [self.global_best.copy()]
        self.goal_reached = False
        self.path_visible = False

        self.update_fitness()
        print("[INFO] Environment deployment validation complete. Optimization core stands ready.")

    def detect_real_obstacles(self):
        """
        Dynamically traverse the USD Stage to isolate authentic structural obstructions.
        
        Algorithmic Pipeline:
        1. Full geometric breadth-first USD hierarchy enumeration.
        2. Known semantic prefix association (e.g., shelving units, operational containers).
        3. Real-time Cartesian coordinate processing via aligned Bounding Box volumes.
        4. Rejection heuristics based on architectural relevance (e.g. overhead trusses).
        
        Returns:
            list: Isolated bounds containing nested floats formatted as [x_min, y_min, width, length]
        """
        obstacles = []

        try:
            from pxr import Usd, UsdGeom, Gf

            stage = omni.usd.get_context().get_stage()
            if not stage:
                print("[ERROR] USD Execution Node acquisition failed.")
                return obstacles

            print("\n  [Diagnostic Phase 1] Surveying core environment layout instances.")
            all_prims = list(stage.Traverse())
            print(f"  [Output] Enumerated primitive entities: {len(all_prims)}")

            import os
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_prims_v2.log")
            with open(log_path, "w") as log_file:
                log_file.write(f"=== USD Stage Subtree Primitive Dump ===\n")
                log_file.write(f"Aggregate count: {len(all_prims)} entities\n\n")
                for i, prim in enumerate(all_prims):
                    prim_path = str(prim.GetPath())
                    prim_type = prim.GetTypeName() if prim.GetTypeName() else "Undefined"
                    log_file.write(f"[{i:4d}] {prim_path}  [{prim_type}]\n")
            print(f"  [Output] Diagnostics dumped to: {log_path}")

            # Instantiate hardware acceleration buffer for accelerated BBox intersection calculations
            purpose_tokens = []
            for token_name in ['default_', 'default', 'render']:
                token = getattr(UsdGeom.Tokens, token_name, None)
                if token is not None:
                    purpose_tokens.append(token)
            if not purpose_tokens:
                purpose_tokens = [UsdGeom.Tokens.default_]

            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purpose_tokens)

            # Dictionary bindings for logical mapping corresponding to existing warehouse standards
            obstacle_prefixes = [
                'Shelf_',      # Industry standard storage layouts
                'PalletBin_',  # Generalized floor impediments
                'SM_Wall',     # Exterior layout confines
                'SM_Pillar',   # Internal structural vertical supports
            ]

            print(f"\n  [Diagnostic Phase 2] Enacting semantic classification constraint matching.")

            detected_count = 0

            for prim in all_prims:
                prim_path = str(prim.GetPath())
                prim_name = prim.GetName()

                # Structural hierarchy verification mitigating excessive computational geometry
                is_obstacle_group = False
                for prefix in obstacle_prefixes:
                    if prim_name.startswith(prefix):
                        parent_path = str(prim.GetParent().GetPath()) if prim.GetParent() else ""
                        if parent_path == "/World/layout":
                            is_obstacle_group = True
                            break

                if not is_obstacle_group:
                    continue

                try:
                    bbox = bbox_cache.ComputeWorldBound(prim)
                    if not bbox or bbox.ComputeAlignedRange().IsEmpty():
                        print(f"  [SKIP] Incomplete temporal spatial domain at: {prim_path}")
                        continue

                    bbox_range = bbox.ComputeAlignedRange()
                    min_pt = bbox_range.GetMin()
                    max_pt = bbox_range.GetMax()

                    x_min, y_min, z_min = float(min_pt[0]), float(min_pt[1]), float(min_pt[2])
                    x_max, y_max, z_max = float(max_pt[0]), float(max_pt[1]), float(max_pt[2])

                    width = x_max - x_min
                    length = y_max - y_min
                    height = z_max - z_min

                    # Physical validity threshold
                    if width < 0.1 or length < 0.1:
                        continue

                    # Volumetric height analysis rejecting irrelevant environmental overhead
                    if z_min > 2.5 or z_max < 0.2:
                        continue

                    obstacle = [x_min, y_min, width, length]
                    obstacles.append(obstacle)
                    detected_count += 1

                except Exception as e:
                    print(f"  [ERROR] Boundary domain logic failure for prim: {prim_path}. Cause: {e}")
                    continue

            print(f"\n  [Semantic Parser Resolution]")
            print(f"  [SUCCESS] {detected_count} distinct structural obstructions instantiated.")

        except Exception as e:
            print(f"[FATAL ERROR] Automated spatial parsing unrecoverable failure: {e}")
            import traceback
            traceback.print_exc()
            obstacles = []

        return obstacles

    def fitness_function(self, position):
        """
        Assesses the topological efficiency array generated via the given spatial coordinate positions.
        Balances direct-route displacement optimizations alongside extreme collision probability penalties.
        """
        pts = position.reshape(self.num_waypoints, 3)
        path = [self.start_pos] + list(pts) + [self.goal_pos]
        
        dist_total = 0.0
        for i in range(len(path) - 1):
            dist_total += np.linalg.norm(path[i] - path[i+1])
        
        obstacle_penalty = 0.0
        
        # Ray-march proxy intersections detecting geometric overlaps
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            
            for j in range(1, 11):
                pt = p1 + (p2 - p1) * (j / 10.0)
                for obstacle in self.obstacles:
                    obs_x, obs_y, obs_w, obs_h = obstacle
                    dx = max(obs_x - pt[0], pt[0] - (obs_x + obs_w), 0)
                    dy = max(obs_y - pt[1], pt[1] - (obs_y + obs_h), 0)
                    distance_to_obstacle = np.sqrt(dx**2 + dy**2)
                    
                    if distance_to_obstacle < self.safety_margin:
                        penalty_factor = (self.safety_margin - distance_to_obstacle) / self.safety_margin
                        obstacle_penalty += (self.obstacle_penalty * 2) * penalty_factor

        return dist_total + obstacle_penalty

    def update_fitness(self):
        """Processes synchronous adaptation of generational fitness arrays."""
        for i in range(self.num_particles):
            fitness = self.fitness_function(self.particles[i])
            if fitness < self.personal_best_fitness[i]:
                self.personal_best[i] = self.particles[i].copy()
                self.personal_best_fitness[i] = fitness
                if fitness < self.global_best_fitness:
                    self.global_best = self.particles[i].copy()
                    self.global_best_fitness = fitness
                    
                    # Resolve fragmented state references down to individual spatial frames
                    best_pts = self.global_best.reshape(self.num_waypoints, 3)
                    self.global_best_path = [self.start_pos.copy()] + list(best_pts) + [self.goal_pos.copy()]

        # Evaluate terminal boundary criterion limiters
        best_pts = self.global_best.reshape(self.num_waypoints, 3)
        dist_to_goal = np.linalg.norm(best_pts[-1] - self.goal_pos)
        if dist_to_goal <= self.goal_tolerance:
            if not self.goal_reached:
                print(f"[CONVERGENCE] Absolute threshold criteria metric met ({self.goal_tolerance}m tolerance area).")
            self.goal_reached = True

    def update_particles(self):
        """Calculates discrete temporal velocity modifications across structural limits."""
        w = 0.7   # Momentum inertia weighting
        c1 = 1.5  # Self-cognitive behavioral constraint
        c2 = 1.5  # Universal-social networking constraint

        for i in range(self.num_particles):
            r1 = np.random.random(self.particle_dim)
            r2 = np.random.random(self.particle_dim)

            # Integration kinematics
            cognitive = c1 * r1 * (self.personal_best[i] - self.particles[i])
            social = c2 * r2 * (self.global_best - self.particles[i])
            self.velocities[i] = w * self.velocities[i] + cognitive + social

            # Numerical validity filtering mechanisms
            if np.any(np.isnan(self.velocities[i])) or np.any(np.isinf(self.velocities[i])):
                self.velocities[i] = np.random.uniform(-0.1, 0.1, self.particle_dim)

            self.velocities[i] = np.clip(self.velocities[i], -0.5, 0.5)

            # Execution logic phase displacement
            self.particles[i] += self.velocities[i]

            if np.any(np.isnan(self.particles[i])) or np.any(np.isinf(self.particles[i])):
                self.particles[i] = np.random.uniform(-15.0, 15.0, self.particle_dim)
            
            for j in range(self.num_waypoints):
                self.particles[i, j*3 : j*3+3] = np.clip(self.particles[i, j*3 : j*3+3], [-15, -15, 1.1], [15, 15, 1.1])

    def visualize_pso_step(self):
        """Initiates GPU-bound real-time debug visualization matrix rendering."""
        self.draw.clear_points()

        # Render explicit no-fly spatial barriers corresponding with real geometry constraints
        for obstacle in self.obstacles:
            obs_x, obs_y, obs_w, obs_h = obstacle

            obs_points = []
            obs_colors = []
            obs_sizes = []

            num_boundary_points = 20
            for i in range(num_boundary_points):
                x = obs_x + (obs_w * i / num_boundary_points)
                obs_points.append([x, obs_y, 1.1])
                obs_colors.append([0.5, 0.5, 0.5, 0.7])
                obs_sizes.append(8.0)
                obs_points.append([x, obs_y + obs_h, 1.1])
                obs_colors.append([0.5, 0.5, 0.5, 0.7])
                obs_sizes.append(8.0)

            for i in range(num_boundary_points):
                y = obs_y + (obs_h * i / num_boundary_points)
                obs_points.append([obs_x, y, 1.1])
                obs_colors.append([0.5, 0.5, 0.5, 0.7])
                obs_sizes.append(8.0)
                obs_points.append([obs_x + obs_w, y, 1.1])
                obs_colors.append([0.5, 0.5, 0.5, 0.7])
                obs_sizes.append(8.0)

            self.draw.draw_points(obs_points, obs_colors, obs_sizes)

            # Render logical safety margins
            box_points = [
                [obs_x, obs_y, 1.1],
                [obs_x + obs_w, obs_y, 1.1],
                [obs_x + obs_w, obs_y + obs_h, 1.1],
                [obs_x, obs_y + obs_h, 1.1],
                [obs_x, obs_y, 1.1],
            ]
            line_colors = [[1.0, 0.0, 0.0, 1.0]] * len(box_points)
            line_sizes = [8.0] * len(box_points)
            self.draw.draw_points(box_points, line_colors, line_sizes)

        # Draw particle dispersion (disabled dynamically after flight mode assumes control)
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
                        obs_x, obs_y, obs_w, obs_h = obstacle
                        dx = max(obs_x - pt[0], pt[0] - (obs_x + obs_w), 0)
                        dy = max(obs_y - pt[1], pt[1] - (obs_y + obs_h), 0)
                        distance = np.sqrt(dx**2 + dy**2)
                        min_distance = min(min_distance, distance)
        
                    if min_distance < self.safety_margin:
                        colors.append([1.0, 0.3, 0.3, 1.0])  # Red - High Criticality
                        sizes.append(15.0)
                    elif min_distance < self.safety_margin * 2:
                        colors.append([1.0, 0.6, 0.2, 1.0])  # Orange - Elevated Warning
                        sizes.append(12.0)
                    else:
                        colors.append([0.0, 0.2, 0.8, 1.0])  # Navy - Operationally Safe
                        sizes.append(10.0)
    
            self.draw.draw_points(draw_pts, colors, sizes)

        # Inject absolute starting origin rendering context
        self.draw.draw_points([self.start_pos.tolist()], [[0, 1, 0, 1]], [30.0])
        # Inject absolute goal mapping rendering context
        self.draw.draw_points([self.goal_pos.tolist()], [[1, 0, 0, 1]], [30.0])
        self.draw_elastic_zone()

        if self.path_visible and len(self.global_best_path) >= 2:
            route_points = []
            for i in range(len(self.global_best_path) - 1):
                start_point = self.global_best_path[i]
                end_point = self.global_best_path[i + 1]
                num_interpolation_points = 20
                for j in range(num_interpolation_points + 1):
                    t = j / num_interpolation_points
                    interpolated_point = start_point + t * (end_point - start_point)
                    route_points.append(interpolated_point.tolist())

            route_colors = [[0.6, 0.0, 0.8, 1.0]] * len(route_points)
            route_sizes = [25.0] * len(route_points)
            self.draw.draw_points(route_points, route_colors, route_sizes)

        best_pts = self.global_best.reshape(self.num_waypoints, 3)
        self.draw.draw_points(best_pts.tolist(), [[0.8, 0.8, 0, 1]] * self.num_waypoints, [25.0] * self.num_waypoints)

    def draw_elastic_zone(self):
        """Displays convergence target parameters using localized sub-point mapping constraints."""
        num_points = 50
        elastic_points = []

        for i in range(num_points):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)

            x = self.goal_pos[0] + self.goal_radius * np.sin(phi) * np.cos(theta)
            y = self.goal_pos[1] + self.goal_radius * np.sin(phi) * np.sin(theta)
            z = self.goal_pos[2] + self.goal_radius * np.cos(phi)
            elastic_points.append([x, y, z])

        elastic_colors = [[1, 0, 0, 0.3]] * len(elastic_points)
        elastic_sizes = [5.0] * len(elastic_points)
        self.draw.draw_points(elastic_points, elastic_colors, elastic_sizes)

    def reset_pso(self):
        """Reinitialize global system heuristics parameters and optimization arrays."""
        self.particles = np.random.uniform(low=-15.0, high=15.0, size=(self.num_particles, self.particle_dim))
        for j in range(self.num_waypoints):
            self.particles[:, j*3 + 2] = 1.1
        self.velocities = np.random.uniform(-0.1, 0.1, size=(self.num_particles, self.particle_dim))
        self.personal_best = self.particles.copy()
        self.personal_best_fitness = np.full(self.num_particles, np.inf)
        
        self.global_best = self.particles[0].copy()
        self.global_best_fitness = self.fitness_function(self.global_best)
        
        best_pts = self.global_best.reshape(self.num_waypoints, 3)
        self.global_best_path = [self.start_pos.copy()] + list(best_pts) + [self.goal_pos.copy()]
        self.goal_reached = False
        self.path_visible = False
        print("[INFO] Particle configurations reinitialized correctly.")

    def run(self):
        """Primary simulation processing runtime supervisor loop."""
        print("============================================================")
        print("[SYSTEM MANAGER] PSO Spatial Controller Matrix Enabled.")
        print(f"[STATUS] Target Scenario Definition: Warehouse with Shelves")
        print(f"[STATUS] Active Environmental Colliders: {len(self.obstacles)} (Semantic Parsing Verified)")
        print(f"[STATUS] Iterative Load Parameters: {self.num_particles} Particles, Max Evaluator Cycles: 1000")
        print(f"[STATUS] Operational Objective: {self.start_pos} -> {self.goal_pos}")
        print("============================================================")

        simulation_count = 0

        while simulation_app.is_running():
            simulation_count += 1
            print(f"\n[INFO] Triggering Simulation Phase Execution Profile Series Number: {simulation_count}")

            self.reset_pso()
            self.timeline.pause() 
            print("[INFO] Timeline Execution Halting: Standby for internal PSO convergence processes.")

            step_count = 0
            max_steps = 1000

            try:
                while simulation_app.is_running() and step_count < max_steps:
                    # Execute algorithmic convergence evaluation steps
                    if step_count % 2 == 0:
                        self.update_particles()
                        self.update_fitness()
                        self.visualize_pso_step()

                    if step_count % 100 == 0:
                        dist_to_goal = np.linalg.norm(self.global_best.reshape(self.num_waypoints, 3)[-1] - self.goal_pos)
                        print(f"Cycle Iterations: {step_count}, "
                              f"Registered Fitness Metric: {self.global_best_fitness:.3f}, "
                              f"Deviation to Sub-Point Objective: {dist_to_goal:.3f}m")

                    if self.goal_reached:
                        print("[INFO] Minimum absolute tolerance threshold hit. Simulation halting current cyclic block.")
                        break

                    self.world.step(render=True)
                    step_count += 1

                    if self.goal_reached:
                        print(f"[CONVERGENCE ACHIEVED] Iterations required: {step_count}, "
                              f"Absolute System Fitness Rating: {self.global_best_fitness:.3f}")
                        break

                self.path_visible = True
                self.visualize_pso_step()
                
                print(f"=== Optimization Subroutine Matrix Cycle #{simulation_count} Finalized ===")
                print(f"[METRIC] Ultimate Fitness Measurement Output Evaluated: {self.global_best_fitness:.3f}")
                print("[INFO] Transcribing trajectory vector coordinates into serial CSV layout definitions.")

                # High density 50Hz (dt=0.02s) temporal interpolation.
                # Suppresses kinematic latency intrinsic to the generic NonlinearController step functionality.
                traj = []
                t_total = 0.0
                speed = 2.0  # Hardware flight profile translation rate (m/s)
                dt = 0.02    # Required optimal execution update pulse (50Hz)
                path_points = self.global_best_path
                
                yaw = 0.0
                traj.append([0.0, path_points[0][0], path_points[0][1], path_points[0][2], 0,0,0, 0,0,0, 0,0,0, yaw, 0.0])
                
                for i in range(len(path_points) - 1):
                    p_start = path_points[i]
                    p_end = path_points[i + 1]
                    dist = np.linalg.norm(p_end - p_start)
                    segment_time = dist / speed if dist > 0 else 0.1
                    v = (p_end - p_start) / segment_time if segment_time > 0 else np.array([0.0, 0.0, 0.0])
                    yaw = np.arctan2(v[1], v[0])
                    
                    num_steps = max(1, int(segment_time / dt))
                    for j in range(1, num_steps + 1):
                        t_total += segment_time / num_steps
                        p = p_start + (p_end - p_start) * (j / num_steps)
                        row = [t_total, p[0], p[1], p[2], v[0], v[1], v[2], 0,0,0, 0,0,0, yaw, 0.0]
                        traj.append(row)
                
                # Assign lingering frame allocations providing termination drift buffer.
                last_p = path_points[-1]
                traj.append([t_total + 10.0, last_p[0], last_p[1], last_p[2], 0,0,0, 0,0,0, 0,0,0, yaw, 0.0])
                
                # Resolving backend file manipulation architecture logic via pre-flips
                traj_np = np.flip(np.array(traj), axis=0)
                csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pso_flight_trajectory.csv")
                np.savetxt(csv_path, traj_np, delimiter=',')
                
                # Final execution hardware assignments
                controller = self.drone._backends[0]
                controller.trajectory = controller.read_trajectory_from_csv(csv_path)
                controller.max_index, _ = controller.trajectory.shape
                controller.total_time = 0.0
                controller.index = 0
                controller.reveived_first_state = False 
                
                print("[INFO] Trajectory CSV matrix completely localized. Autonomous hardware routine execution imminent.")
                
                # Context integration & deployment initiation
                self.world.reset() 
                self.timeline.play()

                while simulation_app.is_running():
                    self.visualize_pso_step()
                    self.world.step(render=True)
                break

            except Exception as e:
                print(f"[FATAL EXCEPTION] Realtime processing cycle crashed intrinsically: {e}")
                import traceback
                traceback.print_exc()
                break

        carb.log_warn("Supervisor process evaluating termination parameters... Issuing simulation tear-down protocols.")
        simulation_app.close()


if __name__ == "__main__":
    sim = PSOWarehouseRealObstacles()
    sim.run()
