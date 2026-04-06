import os
import carb
import omni
from isaacsim import SimulationApp
import time

# 1. 在任何 Import 之前啟動 SimulationApp
CONFIG = {"renderer": "RayTracedLighting", "headless": False}
simulation_app = SimulationApp(CONFIG)

print("等待 Isaac Sim 完全啟動...")
start_time = time.time()
while time.time() - start_time < 60:
    simulation_app.update()
    try:
        if hasattr(simulation_app, 'is_app_ready') and simulation_app.is_app_ready():
            print("Isaac Sim 已準備就緒！")
            break
        elif hasattr(simulation_app, 'is_running') and simulation_app.is_running():
            break
    except:
        pass
    time.sleep(0.5)

import omni.timeline
from omni.isaac.core.world import World
import omni.usd
timeline = omni.timeline.get_timeline_interface()

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

import numpy as np
from scipy.spatial.transform import Rotation
import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.util.debug_draw", True)
from isaacsim.util.debug_draw import _debug_draw
draw_interface = _debug_draw.acquire_debug_draw_interface()


class PSOWarehouseRealObstacles3D:
    """
    PSO 3D 路徑規劃 - 支援在貨架層間穿梭的立體避障版本
    核心修改：
    - BBox 改為收集 Mesh 等級精細包圍盒 (x_min, x_max, y_min, y_max, z_min, z_max)
    - PSO 粒子高度 (z) 改為變動學習，上限至 4.0m，下限 0.3m
    - 碰撞判定改為立體 AABB 最短距離
    """

    def __init__(self):
        try:
            self.timeline = timeline
            self.pg = PegasusInterface()
            self.pg._world = World(**self.pg._world_settings)
            self.world = self.pg.world

            self.pg.load_environment(SIMULATION_ENVIRONMENTS["Warehouse with Shelves"])
            
            print("等待場景解析...")
            for _ in range(30):
                simulation_app.update()

            self.draw = draw_interface

            import sys
            # 請確保此路徑與您的電腦環境一致
            sys.path.insert(0, '/home/mirdc_ju/PegasusSimulator/examples/utils')
            from nonlinear_controller import NonlinearController
            
            config_multirotor = MultirotorConfig()
            controller = NonlinearController(
                trajectory_file=None,
                Kp=[15.0, 15.0, 15.0],
                Kd=[10.0, 10.0, 10.0]
            )
            config_multirotor.backends = [controller]
            # 原始啟動點
            config_multirotor.init_pos = [-8.0, -6.0, 1.1]

            self.drone = Multirotor(
                "/World/Iris_PSO_3D",
                ROBOTS['Iris'],
                0,
                [-8.0, -6.0, 1.1],
                Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
                config=config_multirotor,
            )

            for _ in range(10):
                self.world.step(render=False)
            self.world.reset()

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

        # PSO 3D 基本參數
        self.num_particles = 40
        self.start_pos = np.array([-8.0, -6.0, 1.1])
        # 修改目標位置以利展示穿越：稍高一點、位置更深入貨架側邊
        self.goal_pos = np.array([6.0, 4.0, 1.5]) 
        self.goal_tolerance = 0.05
        self.goal_radius = 0.3

        # 第一步：掃描精準立體障礙物
        self.obstacles = self.detect_real_obstacles()
        self.obstacles_np = np.array(self.obstacles) if self.obstacles else np.empty((0, 6))
        self.obstacle_penalty = 100.0
        # 適度增加安全邊距以避免碰到貨架角角，同時保留穿越可能
        self.safety_margin = 0.35 

        self.num_waypoints = 4  # 3D 空間可能需要更多轉折點
        self.particle_dim = self.num_waypoints * 3
        
        self.reset_pso()
        print("初始化完成！")

    def detect_real_obstacles(self):
        """掃描貨架與棧板下的 Mesh 等級子網格"""
        obstacles = []
        try:
            from pxr import Usd, UsdGeom
            stage = omni.usd.get_context().get_stage()
            if not stage:
                return obstacles

            all_prims = list(stage.Traverse())
            purpose_tokens = [UsdGeom.Tokens.default_]
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purpose_tokens)

            obstacle_prefixes = ['Shelf_', 'PalletBin_', 'SM_Wall', 'SM_Pillar']
            detected_count = 0

            for prim in all_prims:
                # 尋找真正存在體積的底層幾何物體
                if prim.GetTypeName() not in ["Mesh", "Cube", "Cylinder"]:
                    continue

                # 追溯它的父母節點是否有符合白名單群組
                is_obstacle = False
                curr_prim = prim
                while curr_prim:
                    name = curr_prim.GetName()
                    if any(name.startswith(pfx) for pfx in obstacle_prefixes):
                        is_obstacle = True
                        break
                    curr_prim = curr_prim.GetParent()

                if not is_obstacle:
                    continue

                # 計算這單一小物件的 3D BBox
                try:
                    bbox = bbox_cache.ComputeWorldBound(prim)
                    if not bbox or bbox.ComputeAlignedRange().IsEmpty():
                        continue

                    bbox_range = bbox.ComputeAlignedRange()
                    min_pt, max_pt = bbox_range.GetMin(), bbox_range.GetMax()

                    x_min, y_min, z_min = float(min_pt[0]), float(min_pt[1]), float(min_pt[2])
                    x_max, y_max, z_max = float(max_pt[0]), float(max_pt[1]), float(max_pt[2])
                    
                    width, height, z_height = x_max - x_min, y_max - y_min, z_max - z_min

                    # 忽略太小的零件 (例如螺絲或極薄的板子) 避免影響效能且不是主要障礙
                    if width < 0.05 or height < 0.05 or z_height < 0.02:
                        continue

                    obstacles.append([x_min, x_max, y_min, y_max, z_min, z_max])
                    detected_count += 1
                except Exception:
                    pass

            print(f"✓ 總共掃描提取了 {detected_count} 個精細 3D 障礙物(層板與柱子)")
        except Exception:
            pass
        return obstacles

    def fitness_function(self, position):
        """向量化加速的完全立體適應度函數：考量 XYZ 距離"""
        pts = position.reshape(self.num_waypoints, 3)
        path = np.vstack([self.start_pos, pts, self.goal_pos])
        
        diffs = np.diff(path, axis=0)
        dist_total = np.sum(np.linalg.norm(diffs, axis=1))
        
        obstacle_penalty = 0.0
        if len(self.obstacles_np) > 0:
            # 向量化採樣每個線段 20 個點
            t = np.linspace(0.05, 1.0, 20)[:, None]  
            sampled_points = path[:-1, None, :] + diffs[:, None, :] * t
            sampled_points = sampled_points.reshape(-1, 3) 
            
            # 使用 NumPy broadcasting 一次性計算所有點與所有障礙物的距離
            pts_x = sampled_points[:, 0:1]
            pts_y = sampled_points[:, 1:2]
            pts_z = sampled_points[:, 2:3]
            
            x_min, x_max = self.obstacles_np[:, 0], self.obstacles_np[:, 1]
            y_min, y_max = self.obstacles_np[:, 2], self.obstacles_np[:, 3]
            z_min, z_max = self.obstacles_np[:, 4], self.obstacles_np[:, 5]
            
            dx = np.maximum(0, np.maximum(x_min - pts_x, pts_x - x_max))
            dy = np.maximum(0, np.maximum(y_min - pts_y, pts_y - y_max))
            dz = np.maximum(0, np.maximum(z_min - pts_z, pts_z - z_max))
            
            dist_obs = np.sqrt(dx**2 + dy**2 + dz**2) 
            
            mask = dist_obs < self.safety_margin
            if np.any(mask):
                penalty_factors = (self.safety_margin - dist_obs[mask]) / self.safety_margin
                obstacle_penalty = np.sum(penalty_factors) * self.obstacle_penalty

        # 懲罰穿越地面的行為
        floor_penalty = np.sum(np.maximum(0, 0.2 - pts[:, 2])) * 100

        return dist_total + obstacle_penalty + floor_penalty

    def reset_pso(self):
        """允許粒子 Z 軸分佈"""
        self.particles = np.random.uniform(low=-12.0, high=12.0, size=(self.num_particles, self.particle_dim))
        for j in range(self.num_waypoints):
            # 給定初始高度隨機分佈
            self.particles[:, j*3 + 2] = np.random.uniform(0.5, 3.5, self.num_particles)
            
        self.velocities = np.random.uniform(-0.2, 0.2, size=(self.num_particles, self.particle_dim))
        self.personal_best = self.particles.copy()
        
        self.personal_best_fitness = np.array([self.fitness_function(p) for p in self.particles])
        best_idx = np.argmin(self.personal_best_fitness)
        self.global_best = self.particles[best_idx].copy()
        self.global_best_fitness = self.personal_best_fitness[best_idx]
        
        best_pts = self.global_best.reshape(self.num_waypoints, 3)
        self.global_best_path = [self.start_pos.copy()] + list(best_pts) + [self.goal_pos.copy()]
        self.goal_reached = False
        self.path_visible = False

    def update_particles(self):
        w = 0.7
        c1, c2 = 1.5, 1.5
        for i in range(self.num_particles):
            r1 = np.random.random(self.particle_dim)
            r2 = np.random.random(self.particle_dim)

            cognitive = c1 * r1 * (self.personal_best[i] - self.particles[i])
            social = c2 * r2 * (self.global_best - self.particles[i])
            self.velocities[i] = w * self.velocities[i] + cognitive + social

            self.velocities[i] = np.clip(self.velocities[i], -0.6, 0.6)
            self.particles[i] += self.velocities[i]

            # 立體裁切：X, Y 在 -15 ~ 15，Z 則在 0.3 ~ 4.0
            for j in range(self.num_waypoints):
                self.particles[i, j*3] = np.clip(self.particles[i, j*3], -15, 15)
                self.particles[i, j*3+1] = np.clip(self.particles[i, j*3+1], -15, 15)
                self.particles[i, j*3+2] = np.clip(self.particles[i, j*3+2], 0.3, 4.0)

    def visualize_pso_step(self):
        self.draw.clear_points()

        # 為了效能，我們只畫出主要幾個角，表示貨架的空間
        box_points = []
        for obstacle in self.obstacles:
            x_min, x_max, y_min, y_max, z_min, z_max = obstacle
            box_points.extend([
                [x_min, y_min, z_max], [x_max, y_min, z_max],
                [x_max, y_max, z_max], [x_min, y_max, z_max],
                [x_min, y_min, z_min], [x_max, y_max, z_min]
            ])
        if box_points:
            self.draw.draw_points(box_points, [[0.5, 0.5, 0.5, 0.4]] * len(box_points), [5.0] * len(box_points))

        if not getattr(self, 'path_visible', False):
            # 粒子點
            colors, sizes, draw_pts = [], [], []
            for particle in self.particles:
                pts = particle.reshape(self.num_waypoints, 3)
                for pt in pts:
                    draw_pts.append(pt.tolist())
                    colors.append([0.3, 0.5, 1.0, 0.8])
                    sizes.append(10.0)
            self.draw.draw_points(draw_pts, colors, sizes)

        self.draw.draw_points([self.start_pos.tolist()], [[0, 1, 0, 1]], [30.0])
        self.draw.draw_points([self.goal_pos.tolist()], [[1, 0, 0, 1]], [30.0])

        if self.path_visible and len(self.global_best_path) >= 2:
            route_points = []
            for i in range(len(self.global_best_path) - 1):
                start_point, end_point = self.global_best_path[i], self.global_best_path[i+1]
                num_interpolation_points = 20
                for j in range(num_interpolation_points + 1):
                    t = j / num_interpolation_points
                    route_points.append((start_point + t * (end_point - start_point)).tolist())
            self.draw.draw_points(route_points, [[0.6, 0.0, 0.8, 1.0]] * len(route_points), [15.0] * len(route_points))
            
            # Yellow points for waypoints
            best_pts = self.global_best.reshape(self.num_waypoints, 3)
            self.draw.draw_points(best_pts.tolist(), [[0.8, 0.8, 0, 1]] * self.num_waypoints, [25.0] * self.num_waypoints)

    def run(self):
        print("=== 載入 3D 立體 PSO 開始執行 ===")
        simulation_count = 0
        while simulation_app.is_running():
            simulation_count += 1
            self.reset_pso()
            self.timeline.pause()

            step_count = 0
            max_steps = 1500  # 3D 難度較高，允許更多迭代

            for i in range(self.num_particles):
                fitness = self.fitness_function(self.particles[i])
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.particles[i].copy()
                    
            try:
                while simulation_app.is_running() and step_count < max_steps:
                    self.update_particles()
                    
                    for i in range(self.num_particles):
                        fitness = self.fitness_function(self.particles[i])
                        if fitness < self.personal_best_fitness[i]:
                            self.personal_best[i] = self.particles[i].copy()
                            self.personal_best_fitness[i] = fitness
                            if fitness < self.global_best_fitness:
                                self.global_best = self.particles[i].copy()
                                self.global_best_fitness = fitness
                                best_pts = self.global_best.reshape(self.num_waypoints, 3)
                                self.global_best_path = [self.start_pos.copy()] + list(best_pts) + [self.goal_pos.copy()]

                    dist_to_goal = np.linalg.norm(self.global_best.reshape(self.num_waypoints, 3)[-1] - self.goal_pos)
                    if step_count % 100 == 0:
                        print(f"迭代: {step_count}/{max_steps}, 最佳適應度: {self.global_best_fitness:.2f}")

                    if dist_to_goal <= self.goal_tolerance and self.global_best_fitness < 200:
                        print(f"已收斂到目標，迭代步數：{step_count}")
                        break

                    # 大幅減少 render 次數以加速迴圈 (從每次 render 改為每 20 次 render)
                    if step_count % 20 == 0:
                        self.visualize_pso_step()
                        self.world.step(render=True)
                    else:
                        simulation_app.update()
                    step_count += 1

                self.path_visible = True
                self.visualize_pso_step()

                # Generate trajectory
                traj, t_total, dt, speed = [], 0.0, 0.02, 1.5
                yaw = 0.0
                traj.append([0.0, self.global_best_path[0][0], self.global_best_path[0][1], self.global_best_path[0][2], 0,0,0, 0,0,0, 0,0,0, yaw, 0.0])
                
                for i in range(len(self.global_best_path) - 1):
                    p_start, p_end = self.global_best_path[i], self.global_best_path[i + 1]
                    dist = np.linalg.norm(p_end - p_start)
                    segment_time = dist / speed if dist > 0 else 0.1
                    v = (p_end - p_start) / segment_time if segment_time > 0 else np.array([0.0, 0.0, 0.0])
                    # X-Y Yaw
                    if np.linalg.norm([v[0], v[1]]) > 0.01:
                        yaw = np.arctan2(v[1], v[0])
                    
                    num_steps = max(1, int(segment_time / dt))
                    for j in range(1, num_steps + 1):
                        t_total += segment_time / num_steps
                        p = p_start + (p_end - p_start) * (j / num_steps)
                        traj.append([t_total, p[0], p[1], p[2], v[0], v[1], v[2], 0,0,0, 0,0,0, yaw, 0.0])
                
                last_p = self.global_best_path[-1]
                traj.append([t_total + 10.0, last_p[0], last_p[1], last_p[2], 0,0,0, 0,0,0, 0,0,0, yaw, 0.0])

                traj_np = np.flip(np.array(traj), axis=0)
                csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pso_flight_trajectory.csv")
                np.savetxt(csv_path, traj_np, delimiter=',')
                
                controller = self.drone._backends[0]
                controller.trajectory = controller.read_trajectory_from_csv(csv_path)
                controller.max_index, _ = controller.trajectory.shape
                controller.total_time = 0.0
                controller.index = 0
                controller.reveived_first_state = False
                
                print("軌跡匯入成功！無人機起飛。")
                self.world.reset()
                self.timeline.play()

                while simulation_app.is_running():
                    self.visualize_pso_step()
                    self.world.step(render=True)
                break

            except Exception as e:
                print(f"錯誤: {e}")
                import traceback
                traceback.print_exc()
                break

        simulation_app.close()

if __name__ == "__main__":
    sim = PSOWarehouseRealObstacles3D()
    sim.run()
