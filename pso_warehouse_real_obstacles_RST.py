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
            print("載入 Pegasus 模組...")
            from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
            from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
            from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

            self.timeline = timeline

            print("初始化 Pegasus Interface...")
            self.pg = PegasusInterface()

            print("初始化 World...")
            self.pg._world = World(**self.pg._world_settings)
            self.world = self.pg.world
            print("World 初始化完成")

            print("載入倉庫場景...")
            self.pg.load_environment(SIMULATION_ENVIRONMENTS["Warehouse with Shelves"])
            print("倉庫場景載入完成")

            # 等待場景完全載入 —— 讓 USD Stage 完成所有資產解析
            print("等待場景資產完全解析...")
            for _ in range(30):
                simulation_app.update()
            print("場景資產解析完成")

            # Use the global draw interface
            self.draw = draw_interface
            print("Debug draw 初始化完成")

            # ====== 核心改進：從真實場景偵測障礙物 ======
            print("=" * 60)
            print("開始從 USD Stage 偵測真實場景障礙物...")
            print("=" * 60)
            self.obstacle_penalty = 500.0
            self.safety_margin = 1.0  # 增加安全邊距，預留轉彎慣性空間
            # 先初始化無人機體積參數，供起終點合法性檢查使用
            self.drone_body_radius = 0.275
            self.drone_height = 0.30
            self.goal_radius = self.drone_body_radius
            self.flight_z = 1.2
            self.planning_z_min = 0.8
            self.planning_z_max = 2.2
            self.obstacles = self.detect_real_obstacles()

            if not self.obstacles:
                print("⚠ 警告：未偵測到任何障礙物！AI 最佳化將在無障礙空間中運行。")
            else:
                print(f"✓ 共偵測到 {len(self.obstacles)} 個真實障礙物")

            print("開始掃描空間並隨機生成起點與終點...")
            self.start_pos, self.goal_pos = self.generate_random_positions()
            print(f"✓ 生成起點: {self.start_pos}")
            print(f"✓ 生成終點: {self.goal_pos}")

            print("創建 Iris 無人機...")
            import sys, os
            sys.path.insert(0, '/home/mirdc_ju/PegasusSimulator/examples/utils')
            from nonlinear_controller import NonlinearController
            
            config_multirotor = MultirotorConfig()
            
            # 使用 Pegasus 範例提供的高級 NonlinearController
            # 初始時先不給軌跡，等 AI 路徑運算完再指定
            controller = NonlinearController(
                trajectory_file=None,
                Kp=[15.0, 15.0, 15.0],  # 提高位置增益 (Kp) 讓軌跡追蹤更緊密
                Kd=[10.0, 10.0, 10.0]   # 微調微分增益避免震盪
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
            print("Iris 無人機創建完成")

            print("等待無人機初始化...")
            for _ in range(10):
                self.world.step(render=False)
            print("無人機初始化等待完成")

            self.world.reset()
            print("模擬環境重置完成")

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
            print(f"初始化過程中發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            raise

        # RRT 路徑最佳化參數
        self.num_particles = 60  # 每輪評估的候選網路數量
        self.goal_tolerance = 0.05  # 目標容差
        self.obs_bounds_xyz = np.empty((0, 6), dtype=float)
        self.refresh_obstacle_cache()

        # num_waypoints 與粒子初始化將由 compute_dynamic_waypoints() 根據起終點距離與路徑複雜度決定
        # 先給預設值讓後續 fitness_function 等方法可以正常存取
        self.num_waypoints = 3
        self.particle_dim = self.num_waypoints * 3
        self.particles = np.zeros((self.num_particles, self.particle_dim))  # 用於可視化候選路徑
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

        # 自動錄影狀態（_move 版本）
        self.recording_enabled = True
        self.recording_process = None
        self.recording_output_path = ""
        self.recording_log_path = ""
        self.recording_log_file = None
        self.recording_round_index = 0

        # 起終點既定，實際動態計算需要多少轉折點
        self.compute_dynamic_waypoints()

        print("場景建置完成，無人機已就緒。RRT 路徑規劃初始化完成。")

    def ensure_camera_prim(self, camera_path):
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(camera_path).IsValid():
            UsdGeom.Camera.Define(stage, camera_path)
        return stage.GetPrimAtPath(camera_path)

    def setup_dual_viewports(self):
        """建立雙 viewport：主視窗俯視圖，第二視窗為無人機第三人稱視角。"""
        try:
            self.ensure_camera_prim(self.top_camera_path)
            self.ensure_camera_prim(self.follow_camera_path)

            self.top_view_window = get_active_viewport_window()
            if self.top_view_window is None:
                print("⚠ 找不到主 viewport，略過雙視角設定。")
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
            print("✓ 已建立雙 viewport：俯視圖 + 無人機第三人稱視角")
        except Exception as e:
            print(f"⚠ 雙 viewport 初始化失敗：{e}")

    def enforce_embedded_split_layout(self):
        """強制將右側 viewport 以 50% 比例內嵌到主 viewport，而非分頁。"""
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
        """將主 viewport 設為 /OmniverseKit_Top。"""
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
                # fallback：若環境沒有內建 Top 相機，使用自建俯視相機
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
            print(f"⚠ 設定俯視相機失敗：{e}")

    def update_follow_camera_view(self):
        """更新無人機第三人稱跟拍相機。"""
        if not getattr(self, "follow_view_window", None):
            return

        try:
            current_pose = self.drone.get_world_pose()
            if not current_pose or current_pose[0] is None or current_pose[1] is None:
                return

            drone_pos = np.array(current_pose[0], dtype=float)
            drone_quat = current_pose[1]
            body_rotation = Rotation.from_quat([drone_quat[1], drone_quat[2], drone_quat[3], drone_quat[0]])

            # 第三人稱：俯角加大到約 60~80 區間，並把鏡頭整體下移確保看見無人機
            # 幾何上 eye->target 約為 dx=1.9, dz=-3.75，俯角約 arctan(3.75/1.9)=63°
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
        """嘗試找到 Isaac Sim 視窗 ID（X11），找不到則回傳 None。"""
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
        """以 ffmpeg 自動錄影 Isaac Sim 視窗（Linux/X11）。"""
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
            print(f"⚠ 自動錄影未啟用，缺少工具: {', '.join(missing_tools)}")
            return False

        display = os.environ.get("DISPLAY")
        if not display:
            print("⚠ 偵測不到 DISPLAY 環境變數，略過自動錄影。")
            return False

        window_id = self._find_isaac_window_id()
        if not window_id:
            print("⚠ 找不到 Isaac Sim 視窗 ID，略過自動錄影（需安裝 xdotool 且視窗可被搜尋）。")
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
                print(f"⚠ 錄影程序啟動後立即結束，請檢查記錄檔：{self.recording_log_path}")
                self.stop_video_recording()
                return False
            print(f"✓ 已啟動第 {int(round_index)} 輪視窗錄影：{self.recording_output_path}")
            return True
        except Exception as e:
            print(f"⚠ 啟動錄影失敗：{e}")
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
        """停止 ffmpeg 錄影並輸出檔案路徑。"""
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
            print(f"✓ 錄影已儲存：{self.recording_output_path}")

    def handle_manual_close(self):
        """手動關閉 Isaac Sim 視窗時，立即停止錄影並存檔。"""
        if self.recording_process:
            print("\n[手動關窗] 偵測到 Isaac Sim 視窗已關閉，停止錄影並儲存檔案。")
            self.stop_video_recording()

    def generate_random_positions(self):
        """根據掃描到的障礙物與空間範圍，動態生成無碰撞的起點與終點"""
        fixed_z = 1.2

        # 若有掃描到障礙物（含牆壁），用它們來決定可用空間範圍
        if getattr(self, 'obstacles', None):
            # ── 精確內部圖境計算 ────────────────────────────────────────
            # 策略：找出「牆壁類」障礙物（SM_Wall 開頭）的 AABB，
            # 取其「內側邊開」作為可飛行空間邊界。
            # 资架(Shelf) 屬內部障礙物，不用來判斷空間圖境。
            wall_obs = [
                obs for obs in self.obstacles
                if obs[5] >= fixed_z  # 只取在飛行高度有體積的障礙物
            ]
            if wall_obs:
                # 左側牆壁的右邊 (x_max) 中最大値 = 廚房左側內壁
                # 右側牆壁的左邊 (x_min) 中最小値 = 廚房右側內壁
                # 同理 Y 方向
                #
                # 符合倏嶺庺建筑片段內側空間的把握：
                #   可行空間 X: [wall_x_max_left_side, wall_x_min_right_side]
                #   可行空間 Y: [wall_y_max_bottom_side, wall_y_min_top_side]
                #
                # 简化商定聊區定義：取全部牆壁 AABB 的
                #   min_x = 各牆壁 x_max 中的最小値 + margin
                #   max_x = 各牆壁 x_min 中的最大値 - margin
                # 如果小于預設則 fallback。
                xs_lo = sorted(obs[0] for obs in wall_obs)  # x_min 排序
                xs_hi = sorted(obs[3] for obs in wall_obs)  # x_max 排序
                ys_lo = sorted(obs[1] for obs in wall_obs)  # y_min 排序
                ys_hi = sorted(obs[4] for obs in wall_obs)  # y_max 排序

                # 廚房内部 X 範圍：左側牆壁的 x_max (20%百分位) ~ 右側牆壁的 x_min (80%百分位)
                p20_x = xs_hi[int(len(xs_hi) * 0.20)]
                p80_x = xs_lo[int(len(xs_lo) * 0.80)]
                p20_y = ys_hi[int(len(ys_hi) * 0.20)]
                p80_y = ys_lo[int(len(ys_lo) * 0.80)]

                if p80_x > p20_x and p80_y > p20_y:
                    min_x, max_x = p20_x + 0.3, p80_x - 0.3
                    min_y, max_y = p20_y + 0.3, p80_y - 0.3
                    print("掃描粒子分佈範圍成功：基於牆壁類障礙物的內部空間")
                else:
                    # fallback：從外假定範圍
                    all_x_min = min(obs[0] for obs in self.obstacles)
                    all_y_min = min(obs[1] for obs in self.obstacles)
                    all_x_max = max(obs[3] for obs in self.obstacles)
                    all_y_max = max(obs[4] for obs in self.obstacles)
                    min_x = all_x_min + 0.8
                    max_x = all_x_max - 0.8
                    min_y = all_y_min + 0.8
                    max_y = all_y_max - 0.8
                    print("掃描粒子分佈範圍失敗：牆壁類障礙物內部空間過小或無法判定，使用全部障礙物的外部邊界作為 fallback")
            else:
                min_x, max_x = -20.0, 20.0
                min_y, max_y = -20.0, 20.0
                print("掃描粒子分佈範圍失敗：未偵測到有效牆壁類障礙物，使用預設範圍。")
        else:
            # 預設範圍
            min_x, max_x = -20.0, 20.0
            min_y, max_y = -20.0, 20.0
            print("掃描粒子分佈範圍失敗：未偵測到任何障礙物，使用預設範圍。")

        self.space_min_x = min_x
        self.space_max_x = max_x
        self.space_min_y = min_y
        self.space_max_y = max_y

        print(f"正在範圍內找尋起止點: X:[{min_x:.1f}, {max_x:.1f}], Y:[{min_y:.1f}, {max_y:.1f}]")

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
            print("⚠ 無法在安全範圍內找到起點或終點，改用預設備援位置。")
            start_pos = np.array([min_x + 1.0, min_y + 1.0, fixed_z])
            goal_pos = np.array([max_x - 1.0, max_y - 1.0, fixed_z])

        return start_pos, goal_pos

    def is_valid_position(self, pos, buffer=0.2):
        """檢查位置是否遠離任意 3D 障礙物，考慮無人機體積和安全邊距。"""
        drone_z = pos[2]
        h_half = self.drone_height / 2.0
        drone_z_min = drone_z - h_half
        drone_z_max = drone_z + h_half
        for obstacle in self.obstacles:
            obs_x, obs_y, obs_z_min, obs_x_max, obs_y_max, obs_z_max = obstacle
            # 只把「與無人機垂直包絡重疊」的障礙物視為有效阻擋
            if obs_z_max < drone_z_min or obs_z_min > drone_z_max:
                continue
            dx = max(obs_x - pos[0], pos[0] - obs_x_max, 0)
            dy = max(obs_y - pos[1], pos[1] - obs_y_max, 0)
            distance = np.sqrt(dx**2 + dy**2)
            if distance < self.safety_margin + buffer:
                return False
        return True

    def compute_dynamic_waypoints(self):
        """依據場景複雜度決定 waypoint 數量，並初始化 RRT 規劃器。"""
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
        """將任意長度路徑重採樣成固定 num_waypoints，供既有流程使用。"""
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

    def _cubic_bezier_point(self, p0, p1, p2, p3, t):
        """3D cubic Bezier 單點。"""
        omt = 1.0 - t
        return (
            (omt ** 3) * p0
            + 3.0 * (omt ** 2) * t * p1
            + 3.0 * omt * (t ** 2) * p2
            + (t ** 3) * p3
        )

    def smooth_path_with_bezier(self, path_points, samples_per_seg=8, smooth_factor=1.0):
        """以 Catmull-Rom 轉 Bezier 的方式平滑 3D 路徑。"""
        pts = [np.array(p, dtype=float) for p in path_points]
        if len(pts) < 3:
            return pts

        s = float(np.clip(smooth_factor, 0.0, 1.5))
        out = [pts[0].copy()]
        n = len(pts)

        for i in range(n - 1):
            p0 = pts[i - 1] if i - 1 >= 0 else pts[i]
            p1 = pts[i]
            p2 = pts[i + 1]
            p3 = pts[i + 2] if i + 2 < n else pts[i + 1]

            # Catmull-Rom -> Bezier 控制點
            c1 = p1 + (p2 - p0) * (s / 6.0)
            c2 = p2 - (p3 - p1) * (s / 6.0)

            for k in range(1, samples_per_seg + 1):
                t = k / float(samples_per_seg)
                out.append(self._cubic_bezier_point(p1, c1, c2, p2, t))

        return out

    def _polyline_collision_free(self, path_points):
        """檢查 polyline 每段是否都無碰撞。"""
        if len(path_points) < 2:
            return True
        for i in range(len(path_points) - 1):
            if not self._segment_collision_free(np.array(path_points[i], dtype=float), np.array(path_points[i + 1], dtype=float)):
                return False
        return True

    def _segment_collision_free(self, p1, p2):
        """3D 線段與障礙物膨脹包圍盒碰撞檢查。"""
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
        """初始化 RRT 樹。"""
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

        # 先用起終點直線作為初始路徑（若可行）
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
        """當 rewiring 改變父節點成本時，更新其所有子孫成本。"""
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
        """相容舊介面：AI 版本改為執行 RRT* 擴展。"""
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

            # RRT* Step 1: 在鄰域內選擇最低成本且可連通的父節點
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

            # RRT* Step 2: rewiring 鄰域節點到新節點，若可降低成本
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

            # 嘗試接到目標
            if np.linalg.norm(new_node - self.goal_pos) <= self.rrt_goal_threshold:
                if self._segment_collision_free(new_node, self.goal_pos):
                    # 目標可達時，僅在路徑品質更好才更新 best
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
        """停滯時往樹中補充隨機節點，增加探索能力。"""
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
        """快取 3D 膨脹障礙物包圍盒，讓 AI 可直接搜尋不同飛行高度。"""
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
        從 USD Stage 遞迴偵測障礙物。
        貨架/棧板/貨架架體以「2D footprint + 無限高牆面」建模，強制走走道。
        並過濾高空桁架類幾何，避免無人機低空規劃被無關障礙影響。
        """
        obstacles = []
        try:
            from pxr import Usd, UsdGeom, Gf
            stage = omni.usd.get_context().get_stage()
            if not stage:
                return obstacles

            all_prims = list(stage.Traverse())
            print(f"\n  [障礙物掃描] 共尋找到 {len(all_prims)} 個 prim，開始遞迴過濾...")

            purpose_tokens = [UsdGeom.Tokens.default_]
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purpose_tokens)

            # 白名單：僅掃牆壁/貨架/推車；黑名單：排除地板、燈具、桁架等無關物件
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
                            # 貨架類群組向上收斂到 /World/layout 的直接子節點，避免同一貨架被拆成多組
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

                    # 僅保留與可規劃高度帶重疊的障礙物，讓中途抬升/下降也會被納入考慮
                    planning_band_min = self.planning_z_min - self.drone_height
                    planning_band_max = self.planning_z_max + self.drone_height
                    if z_max < planning_band_min or z_min > planning_band_max:
                        continue

                    if hit_keyword in shelf_like_keywords:
                        # 將同一貨架群組的所有零件合併為單一 footprint，稍後轉成無限高牆面
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

            # 貨架群組以無限高牆面加入，避免路徑穿越貨架本體
            INF_Z = 1000.0
            for _, bbox in shelf_group_bbox.items():
                x_min, y_min, x_max, y_max = bbox
                obstacles.append([x_min, y_min, -INF_Z, x_max, y_max, INF_Z])

            print(f"  ✓ 成功偵測到 {detected_count} 個障礙物零件，合併貨架後共 {len(obstacles)} 個規劃障礙物。")

        except Exception as e:
            print(f"✗ 障礙物偵測發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            obstacles = []

        return obstacles

    def fitness_function(self, position):
        """單一粒子適應度（小數量評估時才呼叫）"""
        return self.batch_fitness(position.reshape(1, -1))[0]

    def batch_fitness(self, particles_batch):
        """全向量化批次適應度計算：一次對所有粒子執行矩陣運算。

        這個版本把 waypoint 的 z 一併納入搜尋，避免只在平面上繞路，
        導致看似可行但實際穿過貨架或低矮障礙物上緣的情況。
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

        # 額外路徑品質懲罰：避免粒子黏在起點附近、回頭走、過度折返
        goal_ref = self.goal_pos[np.newaxis, np.newaxis, :]
        start_ref = self.start_pos[np.newaxis, np.newaxis, :]
        direct_len = max(np.linalg.norm(self.goal_pos - self.start_pos), 1e-6)

        # 1) 回頭懲罰：若下一節點離目標更遠則加罰
        dist_to_goal_nodes = np.linalg.norm(path - goal_ref, axis=2)  # (N, nw+2)
        away_steps = np.maximum(dist_to_goal_nodes[:, 1:] - dist_to_goal_nodes[:, :-1], 0.0)
        backtrack_penalty = np.sum(away_steps, axis=1) * 30.0

        # 2) 起點黏著懲罰：各 waypoint 至少應有一定前進距離
        wp = path[:, 1:-1, :]  # (N, nw, 3)
        dist_from_start = np.linalg.norm(wp - start_ref, axis=2)
        alphas = np.linspace(1.0 / (nw + 1), nw / (nw + 1), nw)
        target_progress = 0.35 * direct_len * alphas
        start_sticky_penalty = np.sum(
            np.maximum(target_progress[np.newaxis, :] - dist_from_start, 0.0),
            axis=1,
        ) * 12.0

        # 3) 折返/急轉彎懲罰：鼓勵可飛行的平滑路徑
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
        """保留舊介面；AI 版本不需分離式 fitness 更新。"""
        return

    def update_particles(self):
        """保留舊介面；AI 版本改為 RRT 擴展一步。"""
        self.update_neural_optimizer()

    def _legacy_inject_diversity_disabled(self, fraction=0.2):
        """舊 PSO 版本保留占位；AI 版本不使用。"""
        return

    def visualize_pso_step(self):
        """視覺化 AI 當前最佳化狀態"""
        self.visualize_frame_count += 1
        # 僅在啟動初期低頻重試，避免每幀重設導致閃爍
        if self.viewport_retry_count < 10 and (self.visualize_frame_count % 30 == 0):
            if not self.layout_applied:
                self.enforce_embedded_split_layout()
            if not self.top_view_locked:
                self.set_top_view_camera(force=False)
            self.viewport_retry_count += 1
        self.update_follow_camera_view()
        self.draw.clear_points()

        # --- RRT*/RRT 計算可視化：全樹骨架 + 本輪新增/rewire ---
        if hasattr(self, 'rrt_nodes') and len(self.rrt_nodes) > 1:
            # 1) 全樹骨架（灰色）
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

            # 2) 本輪新增/rewire 邊（亮青色）
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

            # 3) 本輪新增節點（黃色）
            if getattr(self, 'rrt_recent_new_nodes', []):
                self.draw.draw_points(
                    [np.array(n, dtype=float).tolist() for n in self.rrt_recent_new_nodes],
                    [[1.0, 0.95, 0.1, 1.0]] * len(self.rrt_recent_new_nodes),
                    [10.0] * len(self.rrt_recent_new_nodes),
                )

        # 繪製障礙物邊界 + 禁飛紅框
        drone_z = float(self.start_pos[2])
        for obstacle in self.obstacles:
            obs_x, obs_y, obs_z_min, obs_x_max, obs_y_max, obs_z_max = obstacle
            obs_width  = obs_x_max - obs_x
            obs_height = obs_y_max - obs_y

            # 僅繪製在飛行高度有體積的障礙物（可飛越者以淺藍色標示）
            is_blocking = (obs_z_max >= drone_z)  # 不可飛越 → 紅框
            frame_color = [1.0, 0.0, 0.0, 1.0] if is_blocking else [0.0, 0.5, 1.0, 0.5]

            obs_points = []
            obs_colors = []
            obs_sizes = []

            num_boundary_points = 20
            for i in range(num_boundary_points):
                # 上邊界
                x = obs_x + (obs_width * i / num_boundary_points)
                obs_points.append([x, obs_y, drone_z])
                obs_colors.append([0.5, 0.5, 0.5, 0.7])
                obs_sizes.append(8.0)
                # 下邊界
                obs_points.append([x, obs_y_max, drone_z])
                obs_colors.append([0.5, 0.5, 0.5, 0.7])
                obs_sizes.append(8.0)

            for i in range(num_boundary_points):
                y = obs_y + (obs_height * i / num_boundary_points)
                # 左邊界
                obs_points.append([obs_x, y, drone_z])
                obs_colors.append([0.5, 0.5, 0.5, 0.7])
                obs_sizes.append(8.0)
                # 右邊界
                obs_points.append([obs_x_max, y, drone_z])
                obs_colors.append([0.5, 0.5, 0.5, 0.7])
                obs_sizes.append(8.0)

            self.draw.draw_points(obs_points, obs_colors, obs_sizes)

            # 禁飛框（紅色 = 阻擋，藍色 = 可飛越）
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

        # 如果已經生成最終軌跡，隱藏粒子只保留紫線與紅框
        if not getattr(self, 'path_visible', False):
            # 繪製粒子
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
                        # 只對與無人機垂直包絡重疊的障礙物計算距離
                        h_half = self.drone_height / 2.0
                        if obs_z_max < (drone_z - h_half) or obs_z_min > (drone_z + h_half):
                            continue
                        dx = max(obs_x - pt[0], pt[0] - obs_x_max, 0)
                        dy = max(obs_y - pt[1], pt[1] - obs_y_max, 0)
                        distance = np.sqrt(dx**2 + dy**2)
                        min_distance = min(min_distance, distance)
        
                    if min_distance < self.safety_margin:
                        colors.append([1.0, 0.3, 0.3, 1.0])  # 紅色警示
                        sizes.append(15.0)
                    elif min_distance < self.safety_margin * 2:
                        colors.append([1.0, 0.6, 0.2, 1.0])  # 橙色警告
                        sizes.append(12.0)
                    else:
                        colors.append([0.0, 0.2, 0.8, 1.0])  # 安全深藍色
                        sizes.append(10.0)
    
            self.draw.draw_points(draw_pts, colors, sizes)

        # 繪製起點（綠色）
        self.draw.draw_points([self.start_pos.tolist()], [[0, 1, 0, 1]], [30.0])
        # 繪製終點（紅色）
        self.draw.draw_points([self.goal_pos.tolist()], [[1, 0, 0, 1]], [30.0])
        # 繪製彈性區
        self.draw_elastic_zone()

        # 最終路徑：僅在最終可視化模式下繪製細線（縮小點容替粗線）
        if self.path_visible and len(self.global_best_path) >= 2:
            route_points = []
            for i in range(len(self.global_best_path) - 1):
                start_point = self.global_best_path[i]
                end_point = self.global_best_path[i + 1]
                num_interpolation_points = 8   # 減少插値點，避免點實太密
                for j in range(num_interpolation_points + 1):
                    t = j / num_interpolation_points
                    interpolated_point = start_point + t * (end_point - start_point)
                    route_points.append(interpolated_point.tolist())

            route_colors = [[0.7, 0.0, 1.0, 0.9]] * len(route_points)
            route_sizes = [4.0] * len(route_points)   # 大幅縮小：25 → 4
            self.draw.draw_points(route_points, route_colors, route_sizes)

        # 繪製當前全局最佳轉折點（黃綠色）＋無人機尺寸圓圈
        best_pts = self.global_best.reshape(self.num_waypoints, 3)
        self.draw.draw_points(
            best_pts.tolist(),
            [[0.8, 0.8, 0, 1]] * self.num_waypoints,
            [15.0] * self.num_waypoints
        )
        # 在每個轉折點畫一個無人機尺寸的圓圈（半透明黃色）——讓發展看清楚轉折點有沒有碰這障礙物
        for wp in best_pts:
            self.draw_elastic_zone(
                center=wp.tolist(),
                color=[1.0, 0.9, 0.0, 0.25],   # 半透明黃色
                radius=self.drone_body_radius
            )

    def draw_elastic_zone(self, center=None, color=None, radius=None):
        """繪製無人機尺寸球體点雲（占空尺寸可視化）
        
        Args:
            center: 球心位置 [x,y,z]，預設為終點
            color : [r,g,b,a]，預設為半透明紅
            radius: 球體半徑，預設為 drone_body_radius
        """
        if center is None:
            center = self.goal_pos
        if color  is None:
            color  = [1.0, 0.2, 0.2, 0.35]
        if radius is None:
            radius = self.drone_body_radius

        num_points = 40
        elastic_points = []

        # 均勻采樣在球面上（利用 Fibonacci sphere 讓分布更均勻）
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
        """重置 AI 最佳化狀態（保留起終點，重新初始化 RRT 樹）。"""
        self.compute_dynamic_waypoints()
        self.optim_iteration = 0

        # 路徑歷史
        best_pts = self.global_best.reshape(self.num_waypoints, 3)
        self.global_best_path = [self.start_pos.copy()] + list(best_pts) + [self.goal_pos.copy()]
        self.goal_reached = False
        self.path_visible = False
        print("AI 最佳化狀態已重置，準備新一輪優化。")

    def check_path_collision(self):
        """使用精確 3D Slab Method 驗證當前最佳路徑每條線段是否穿越障礙物。
        加入 Z 軸過濾：僅對飛行高度有體積的障礙物進行碰撞偵測。

        Returns:
            (bool, int): (有碰撞, 碰撞線段數)
        """
        if not self.obstacles:
            return False, 0

        path = self.global_best_path   # list of np.ndarray  [x, y, z]
        obs_arr_all = np.array(self.obstacles)   # (M, 6): [xmin,ymin,zmin,xmax,ymax,zmax]

        # ══ 膨脹 AABB（3D）：與 batch_fitness 邏輯完全一致 ══════════
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

            # ── X 軸 Slab ──
            if abs(dx) < EPS:
                tx_lo = np.where((p1[0] >= ox_all) & (p1[0] <= oxw_all), -np.inf,  np.inf)
                tx_hi = np.where((p1[0] >= ox_all) & (p1[0] <= oxw_all),  np.inf, -np.inf)
            else:
                tx1 = (ox_all  - p1[0]) / dx
                tx2 = (oxw_all - p1[0]) / dx
                tx_lo = np.minimum(tx1, tx2)
                tx_hi = np.maximum(tx1, tx2)

            # ── Y 軸 Slab ──
            if abs(dy) < EPS:
                ty_lo = np.where((p1[1] >= oy_all) & (p1[1] <= oyh_all), -np.inf,  np.inf)
                ty_hi = np.where((p1[1] >= oy_all) & (p1[1] <= oyh_all),  np.inf, -np.inf)
            else:
                ty1 = (oy_all  - p1[1]) / dy
                ty2 = (oyh_all - p1[1]) / dy
                ty_lo = np.minimum(ty1, ty2)
                ty_hi = np.maximum(ty1, ty2)

            # ── Z 軸 Slab ──
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
        """主運行循環"""
        print("=" * 60)
        print("AI 倉庫路徑規劃模擬（RRT 最佳化版）已準備就緒！")
        print(f"場景：Warehouse with Shelves")
        print(f"障礙物數量：{len(self.obstacles)} 個（從場景自動偵測）")
        print(f"AI 參數：每輪 {self.num_particles} 個網路候選，500 次迭代")
        print(f"起點：{self.start_pos}  →  終點：{self.goal_pos}")
        print(f"規劃高度範圍：Z:[{self.planning_z_min:.2f}, {self.planning_z_max:.2f}]")
        print(f"目標容差：{self.goal_tolerance}m")
        print("按 Play 開始模擬，或直接關閉窗口退出")
        print("=" * 60)

        simulation_count = 0

        while simulation_app.is_running():
            simulation_count += 1
            print(f"\n=== 開始第 {simulation_count} 輪模擬（AI 路徑搜尋）===")

            # 每輪分檔：從 AI 計算開始錄到本輪結束
            self.stop_video_recording()
            self.start_video_recording(simulation_count)

            self.reset_pso()
            self.timeline.pause() # 暫停物理引擎，直到 AI 算完
            print("等待 RRT 計算路徑，物理引擎暫時停用")

            step_count = 0
            max_steps = 500
            self.optim_max_iterations = max_steps
            visualize_interval = 10
            update_interval = 1
            best_checkpoint = np.inf
            stagnant_count = 0

            try:
                while simulation_app.is_running() and step_count < max_steps:
                    # ===== RRT 路徑最佳化開始 =====
                    if step_count % update_interval == 0:
                        self.update_neural_optimizer()

                    if step_count % visualize_interval == 0:
                        self.visualize_pso_step()
                        simulation_app.update()

                    if step_count % 20 == 0:
                        dist_to_goal = np.linalg.norm(self.global_best.reshape(self.num_waypoints, 3)[-1] - self.goal_pos)
                        node_count = len(self.rrt_nodes) if hasattr(self, 'rrt_nodes') else 0
                        print(f"AI 迭代步數: {step_count}, "
                              f"節點數: {node_count}, "
                              f"本輪新增: {getattr(self, 'rrt_added_last', 0)}, "
                              f"本輪rewire: {getattr(self, 'rrt_rewire_count_last', 0)}, "
                              f"最佳適應度: {self.global_best_fitness:.3f}, "
                              f"距目標: {dist_to_goal:.3f}m")

                    if self.goal_reached:
                        print("目標已精準到達，停止當前模擬輪次。")
                        break

                    if step_count % 50 == 0:
                        simulation_app.update()

                    # 長時間沒有進步就提早停止，避免無效運算
                    if step_count % 50 == 0:
                        if self.global_best_fitness < best_checkpoint - 1e-3:
                            best_checkpoint = self.global_best_fitness
                            stagnant_count = 0
                        else:
                            stagnant_count += 1
                        if stagnant_count >= 2 and stagnant_count < 4:
                            self.inject_diversity(fraction=0.2)
                            print("AI 停滯，注入參數擾動以增加探索能力。")
                        if stagnant_count >= 4:
                            print("AI 長時間無顯著改善，提前結束本輪優化。")
                            break

                    step_count += 1

                    if self.goal_reached:
                        print(f"已精準收斂到目標，迭代步數：{step_count}, "
                              f"最佳適應度：{self.global_best_fitness:.3f}")
                        break

                if not simulation_app.is_running():
                    self.handle_manual_close()
                    break

                # 顯示最終路徑
                self.path_visible = True
                self.visualize_pso_step()
                
                print(f"=== 第 {simulation_count} 輪 AI 最佳化結束 ===")
                print(f"最終最佳適應度: {self.global_best_fitness:.3f}")

                # ══════════════════════════════════════════════════════
                # 飛行前路徑安全驗證：精確 Slab Method 碰撞檢測
                # 若路徑仍穿越障礙物 → 重跑 AI 最佳化（最多 MAX_RETRY 次）
                # 完全無碰撞才允許起飛，否則最終使用最佳可得路徑並警告
                # ══════════════════════════════════════════════════════
                MAX_RETRY = 5
                retry_count = 0
                has_collision, col_segs = self.check_path_collision()

                while has_collision and retry_count < MAX_RETRY:
                    retry_count += 1
                    print(f"\n⚠ [路徑驗證失敗] 最佳路徑仍有 {col_segs} 段穿越障礙物！")
                    print(f"  → 第 {retry_count}/{MAX_RETRY} 次重新搜尋（重新初始化 RRT 樹）...")

                    # 重新初始化 RRT 樹（保留起終點）
                    self.compute_dynamic_waypoints()

                    retry_steps = 500
                    for _ in range(retry_steps):
                        self.update_neural_optimizer()
                        if not simulation_app.is_running():
                            break
                        if _ % 50 == 0:
                            simulation_app.update()

                    # 更新最佳路徑列表
                    best_pts = self.global_best.reshape(self.num_waypoints, 3)
                    self.global_best_path = (
                        [self.start_pos.copy()] + list(best_pts) + [self.goal_pos.copy()]
                    )
                    self.path_visible = True
                    self.visualize_pso_step()

                    has_collision, col_segs = self.check_path_collision()
                    print(f"  重跑後適應度: {self.global_best_fitness:.3f}，"
                          f"碰撞線段數: {col_segs}")

                if has_collision:
                    print(f"\n⚠ [警告] 經過 {MAX_RETRY} 次重試仍無法找到完全無碰撞路徑，"
                          f"取消本輪起飛（碰撞線段: {col_segs}）。")
                    self.safety_margin += 0.2
                    self.refresh_obstacle_cache()
                    print(f"  已提高安全邊距至 {self.safety_margin:.2f}，下一輪重新規劃。")
                    self.world.reset()
                    self.stop_video_recording()
                    continue
                else:
                    print(f"\n✓ [路徑驗證通過] 路徑完全無碰撞！準備起飛。")

                # 飛行前圓滑化：對驗證通過的路徑做 Bezier 平滑，若平滑後碰撞則回退原路徑
                raw_path_points = [np.array(p, dtype=float) for p in self.global_best_path]
                smoothed_path_points = self.smooth_path_with_bezier(raw_path_points, samples_per_seg=8, smooth_factor=1.0)
                if self._polyline_collision_free(smoothed_path_points):
                    self.global_best_path = [np.array(p, dtype=float) for p in smoothed_path_points]
                    print(f"✓ 已套用貝茲曲線圓滑化（點數: {len(raw_path_points)} -> {len(self.global_best_path)}）")
                else:
                    self.global_best_path = raw_path_points
                    print("⚠ 貝茲曲線圓滑化後出現碰撞，已回退原始路徑。")

                print("開始生成飛行軌跡檔案並準備無人機飛行...")

                # ====== 梯形速度剖面軌跡生成 ======
                # 策略：
                #   1. 計算整條路徑各航點的累積弧長
                #   2. 依據距起終點距離決定速度（加速/巡航/減速三段）
                #   3. 中間航點根據轉彎角度局部降速（轉彎越急速度越慢）
                #   4. 終點前 decel_dist 以上就開始持續減速至 0
                traj = []
                t_total = 0.0
                dt      = 0.02    # 50Hz 控制頻率
                v_max   = 3.2     # 最高巡航速度 (m/s)（加快）
                v_min   = 0.5     # 轉彎/接近終點時的最低速度 (m/s)（加快）
                a_max   = 2.0     # 最大加速度 / 減速度 (m/s²)（加快）
                path_points = self.global_best_path
                n_pts = len(path_points)

                # --- Step 1: 各航段長度 & 累計弧長 ---
                seg_dists = []
                for i in range(n_pts - 1):
                    seg_dists.append(np.linalg.norm(path_points[i+1] - path_points[i]))
                total_path_len = sum(seg_dists)

                # 加速段/減速段所需距離 (v²=2as  → s=v²/2a)
                acc_dist  = v_max**2 / (2.0 * a_max)   # 從 0 加速到 v_max 的距離
                decel_dist = acc_dist                    # 從 v_max 減速到 0 的距離

                # --- Step 2: 計算各中間航點（非起終點）的「轉彎速度上限」---
                # 轉彎角度越大 → 速度越低 (cosine 映射)
                waypoint_speed_limit = [v_max] * n_pts
                waypoint_speed_limit[0]  = 0.0   # 起點：靜止
                waypoint_speed_limit[-1] = 0.0   # 終點：靜止
                for i in range(1, n_pts - 1):
                    d_in  = path_points[i]   - path_points[i-1]
                    d_out = path_points[i+1] - path_points[i]
                    norm_in  = np.linalg.norm(d_in)
                    norm_out = np.linalg.norm(d_out)
                    if norm_in > 1e-6 and norm_out > 1e-6:
                        cos_a = np.dot(d_in, d_out) / (norm_in * norm_out)
                        cos_a = np.clip(cos_a, -1.0, 1.0)
                        # cos_a = 1  (直線) → 不降速；cos_a = -1 (U 型轉) → 降至 v_min
                        turn_factor = (cos_a + 1.0) / 2.0           # 0~1
                        waypoint_speed_limit[i] = v_min + turn_factor * (v_max - v_min)

                # --- Step 3: 逐線段插值，計算每個 dt 時間步的位置與速度 ---
                # 計算每個航點的累計弧長
                cum_dist = [0.0]
                for d in seg_dists:
                    cum_dist.append(cum_dist[-1] + d)

                def speed_profile(s):
                    """根據距起點弧長 s 決定梯形速度（不考慮轉彎）"""
                    # 加速段
                    if s < acc_dist:
                        return max(v_min, np.sqrt(2.0 * a_max * s))
                    # 減速段
                    remaining = total_path_len - s
                    if remaining < decel_dist:
                        return max(0.0, np.sqrt(2.0 * a_max * remaining))
                    # 巡航段
                    return v_max

                # 第一個點 (起點，靜止)
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

                    # 沿此段以固定 dt 推進：根據當前弧長決定速度
                    s_seg = 0.0   # 在本線段內走的距離
                    while s_seg < seg_len:
                        s_global = cum_dist[i] + s_seg   # 距整條路徑起點的弧長

                        # 全局梯形速度
                        v_trap = speed_profile(s_global)

                        # 轉彎速度上限（線性插值兩端航點的速度限制）
                        alpha   = s_seg / seg_len                                  # 0→1
                        v_limit = (1 - alpha) * waypoint_speed_limit[i] + alpha * waypoint_speed_limit[i+1]
                        v_limit = max(v_limit, 0.01)   # 避免除以零

                        v_now = min(v_trap, v_limit)
                        v_now = max(v_now, 0.0)

                        # 更新位置
                        p_now = p_start + seg_dir * s_seg
                        vel_vec = seg_dir * v_now
                        t_total += dt
                        row = [t_total,
                               p_now[0], p_now[1], p_now[2],
                               vel_vec[0], vel_vec[1], vel_vec[2],
                               0, 0, 0, 0, 0, 0,
                               yaw, 0.0]
                        traj.append(row)

                        # 推進弧長
                        s_seg += v_now * dt if v_now > 0.01 else dt * v_min

                    # 確保精確抵達航點端點
                    p_end_arr = np.array(p_end)
                    v_wp = waypoint_speed_limit[i+1]
                    traj.append([t_total,
                                 p_end_arr[0], p_end_arr[1], p_end_arr[2],
                                 seg_dir[0]*v_wp, seg_dir[1]*v_wp, seg_dir[2]*v_wp,
                                 0, 0, 0, 0, 0, 0,
                                 yaw, 0.0])

                # 終點懸停：給足夠時間讓無人機穩定降至目標點
                last_p = path_points[-1]
                traj.append([t_total + 5.0,
                             last_p[0], last_p[1], last_p[2],
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             yaw, 0.0])
                traj.append([t_total + 15.0,
                             last_p[0], last_p[1], last_p[2],
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             yaw, 0.0])
                print(f"軌跡生成完成：共 {len(traj)} 個控制點，預計飛行時間 {t_total:.1f}s")

                
                # NonlinearController 讀檔是以 flip axis=0 反轉序列的，所以寫檔時要先反轉
                traj_np = np.flip(np.array(traj), axis=0)
                csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_flight_trajectory.csv")
                np.savetxt(csv_path, traj_np, delimiter=',')
                
                # 更新無人機控制器
                controller = self.drone._backends[0]
                controller.trajectory = controller.read_trajectory_from_csv(csv_path)
                controller.max_index, _ = controller.trajectory.shape
                controller.total_time = 0.0
                controller.index = 0
                controller.reveived_first_state = False # 重置飛行狀態
                
                print("軌跡匯入成功！無人機即將起飛...")
                
                # 恢復模擬，準備看無人機飛行
                self.world.reset() # 確保無人機回到初始點
                # 每次起飛前強制把無人機放到「當前 start_pos」，避免 reset 回到舊起點
                try:
                    if hasattr(self.drone, "set_world_pose"):
                        self.drone.set_world_pose(
                            self.start_pos.tolist(),
                            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
                        )
                        # 讓 pose 寫入在下一步生效
                        for _ in range(2):
                            self.world.step(render=False)
                    else:
                        print("⚠ 無人機物件不支援 set_world_pose，可能無法重置到新起點。")
                except Exception as pose_err:
                    print(f"⚠ 起飛前重設無人機位置失敗: {pose_err}")
                self.timeline.play()

                # 保持畫面與物理更新直到迴圈關閉，同時監控碰撞
                collision_occurred = False
                reached_goal = False
                while simulation_app.is_running():
                    # self.update_neural_optimizer() 取消註解這行會讓 RRT 持續擴展，我們現在不需要
                    self.visualize_pso_step()
                    self.world.step(render=True)
                    self.update_follow_camera_view()

                    # 碰撞監控：改為「真實物理墜毀與翻覆」偵測
                    # get_world_pose() 返回 (position, [qw, qx, qy, qz])
                    current_pose = self.drone.get_world_pose()
                    if current_pose and current_pose[0] is not None:
                        drone_pos = current_pose[0]
                        drone_quat = current_pose[1]
                        
                        # 將四元數 [qw, qx, qy, qz] 轉為 Euler Angles 判斷真實物理翻覆
                        from scipy.spatial.transform import Rotation
                        r = Rotation.from_quat([drone_quat[1], drone_quat[2], drone_quat[3], drone_quat[0]])
                        euler_angles = r.as_euler('xyz', degrees=True)
                        roll, pitch = euler_angles[0], euler_angles[1]
                        
                        # 當無人機因為真實物理撞擊導致翻覆 (傾角>60度) 或墜落到地面 (Z<0.3) 視為墜毀！
                        if abs(roll) > 60 or abs(pitch) > 60 or drone_pos[2] < 0.3:
                            print(f"\n[墜機警告] 偵測到無人機物理墜毀！(Roll: {roll:.1f}°, Pitch: {pitch:.1f}°, Z: {drone_pos[2]:.2f}m)")
                            collision_occurred = True

                        # 到點判斷：看到飛機到達定點即下達到點指令並停止錄影存檔
                        dist_to_goal_live = np.linalg.norm(np.array(drone_pos) - self.goal_pos)
                        arrive_threshold = max(self.goal_tolerance, self.goal_radius)
                        if dist_to_goal_live <= arrive_threshold:
                            print(f"\n[到點指令] 飛機已到達定點！(距離目標: {dist_to_goal_live:.3f}m, 閾值: {arrive_threshold:.3f}m)")
                            reached_goal = True
                            self.timeline.stop()
                            self.stop_video_recording()
                            break
                            
                        # 若無墜機，加上一點小延遲讓畫面更新平順
                        import time
                        time.sleep(0.01)
                                
                    if collision_occurred:
                        print(f"中斷當前飛行，增加安全防護距離重新計算路徑...")
                        self.safety_margin += 0.2
                        self.refresh_obstacle_cache()
                        print(f"新的安全邊距 (Safety Margin) 更新為: {self.safety_margin:.2f}")
                        # 墜毀後停留一小段時間讓使用者看見翻覆畫面
                        for _ in range(100):
                            self.world.step(render=True)
                        self.timeline.stop()
                        break

                if not simulation_app.is_running() and not reached_goal and not collision_occurred:
                    self.handle_manual_close()

                self.stop_video_recording()
                        
                # 如果是發生碰撞而破壞內部迴圈，不要跳出外面的大迴圈（即不要執行 break）
                if collision_occurred:
                    print("重置物理世界，準備重新啟動 AI 最佳化...")
                    self.world.reset()
                    continue  # continue 外層大迴圈重新跑 AI 最佳化
                else:
                    if reached_goal:
                        print("本輪已完成到點並存檔。")
                        rerun_choice = "n"
                        while True:
                            try:
                                rerun_choice = input("是否重新生成起始點與終點並重跑一次？[Y/N]: ").strip().lower()
                            except EOFError:
                                rerun_choice = "n"
                            if rerun_choice in ("y", "yes", "n", "no"):
                                break
                            print("請輸入 Y 或 N。")

                        if rerun_choice in ("y", "yes"):
                            print("已選擇重新生成起始點與終點，準備啟動下一輪。")
                            self.start_pos, self.goal_pos = self.generate_random_positions()
                            self.goal_reached = False
                            self.path_visible = False
                            self.refresh_obstacle_cache()
                            print(f"新起點: {self.start_pos}")
                            print(f"新終點: {self.goal_pos}")

                            try:
                                self.timeline.stop()
                                self.world.reset()
                                if hasattr(self.drone, "set_world_pose"):
                                    self.drone.set_world_pose(
                                        self.start_pos.tolist(),
                                        Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
                                    )
                                    for _ in range(2):
                                        self.world.step(render=False)
                            except Exception as pose_err:
                                print(f"⚠ 無法直接重設無人機到新起點: {pose_err}")

                            continue  # 回到外層 while，重跑整個流程

                        print("已選擇不重跑，保留視窗畫面（關閉視窗即結束程式）。")
                        while simulation_app.is_running():
                            self.visualize_pso_step()
                            self.update_follow_camera_view()
                            simulation_app.update()
                            time.sleep(0.01)
                    break     # 否則代表正常結束，或手動關閉，可以跳出模擬大迴圈

            except Exception as e:
                print(f"模擬過程中發生錯誤: {e}")
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
