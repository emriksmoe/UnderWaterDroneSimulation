import copy
import numpy as np
import simpy
import gymnasium as gym
from typing import Optional

from src.utils.position import Position
from src.protocols.dtn_protocol import DTNMessage
from src.simulation.agent_factory import AgentFactory
from src.simulation.processes import (
    sensor_process,
    ship_process,
)
from src.rl.rl_memory import RLMemory
from src.rl.state_builder import build_observation
from analysis.metrics import MetricsCollector


class DroneAoIEnv(gym.Env):
    """
    RL environment wrapper around SimPy DTN simulation.

    One RL step = one movement decision (sensor or ship).
    Reward = negative continuous-time AoI integral accumulated during movement.
    """

    metadata = {"render.modes": []}

    def __init__(self, config, episode_duration: int = 86400, shaping_lambda: float = 0.0, dither: float = 0.0):
        super().__init__()
        self.config = config
        self.episode_duration = episode_duration
        self.shaping_lambda = shaping_lambda
        self.dither = dither
        self.last_action = None

        self._build_sim()

        self.num_sensors = len(self.sensors)
        self.action_space = gym.spaces.Discrete(self.num_sensors + 1)

        sensors_ordered = sorted(self.sensors, key=lambda s: s.id)
        example_obs = build_observation(
            drone=self.drone,
            sensors=sensors_ordered,
            ship=self.ship,
            current_time=0.0,
            config=self.config,
            rl_memory=self.memory,
            episode_duration=self.episode_duration,
        )

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=example_obs.shape, dtype=np.float32
        )

    # ------------------------------------------------------------------
    # BUILD SIMULATION
    # ------------------------------------------------------------------
    def _build_sim(self):
        cfg = copy.deepcopy(self.config)
        cfg.sim_time = self.episode_duration
        self.config = cfg

        self.env = simpy.Environment()

        factory = AgentFactory(self.config, drone_strategy=None)
        self.sensors, self.drones, self.ships = factory.create_all_agents()

        assert len(self.drones) == 1
        assert len(self.ships) == 1

        self.drone = self.drones[0]
        self.ship = self.ships[0]

        # Metrics (kept for logging only – NOT used for RL reward)
        self.ship.metrics = MetricsCollector()

        self.memory = RLMemory()

        # Processes
        for sensor in self.sensors:
            self.env.process(sensor_process(self.env, sensor, self.config, self.ship.metrics))

        self.env.process(ship_process(self.env, self.ship, self.config))

        # RL control
        self.decision_event = self.env.event()
        self.env.process(self._drone_rl_process())

    # ------------------------------------------------------------------
    # DRONE RL PROCESS
    # ------------------------------------------------------------------
    def _drone_rl_process(self):
        """Executes one chosen action, then releases RL step()."""
        while True:
            yield self.env.timeout(0)

            idx = self._current_target_idx

            sensors_ordered = sorted(self.sensors, key=lambda s: s.id)



            if idx < len(sensors_ordered):
                target = sensors_ordered[idx]
                target_type = "sensor"
            else:
                target = self.ship
                target_type = "ship"

            travel_time = self._compute_travel_time(target)
            yield self.env.timeout(travel_time)
            self.drone.position = target.position

            tot_msg = self._handle_arrival(target, target_type)

            operation_time = self.config.communication_hand_shake_time + (tot_msg * self.config.communication_bitrate)
            yield self.env.timeout(operation_time)

            self.decision_event.succeed()
            self.decision_event = self.env.event()


    # ------------------------------------------------------------------
    # GYM API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._build_sim()

        sensors_ordered = sorted(self.sensors, key=lambda s: s.id)
        self.sensor_ids = [s.id for s in sensors_ordered]
        self.num_sensors = len(sensors_ordered)
        self.action_space = gym.spaces.Discrete(self.num_sensors + 1)

        # Ship-side AoI anchor: last delivered generation time per sensor
        self.ship.metrics.last_delivery_time_per_sensor = {sid: 0.0 for sid in self.sensor_ids}
        self.ship.metrics.last_aoi_update_time = 0.0

        self.previous_time = 0.0
        self.total_delivered = 0
        self.last_action = None

        return self._get_obs(), {}

    def step(self, action: int):
        info = {}
        action = int(action)


        self.last_action = action
        self._current_target_idx = action

        #Get AoI before movment
        self.ship.metrics.update_aoi_integral(float(self.env.now))
        aoi_before = self.ship.metrics.aoi_integral

        # Run SimPy until arrival
        self.env.run(until=self.decision_event)

        #Get AoI after movment
        now = float(self.env.now)
        self.ship.metrics.update_aoi_integral(now)
        aoi_after = self.ship.metrics.aoi_integral

        cost = aoi_after - aoi_before

        ESP = self.dither #Small decision cost to avoid dithering
        # Normalize reward for PPO stability
        reward_aoi = -(cost + ESP) / (len(self.sensor_ids) * 1000.0)

        #Additional reward shape
        starve_max = self._max_time_since_any_sensor_visit(now)

        reward_shape = -self.shaping_lambda * starve_max

        reward = reward_aoi + reward_shape #if lambda = 0, no shaping

        obs = self._get_obs()
        done = bool(now >= self.episode_duration)
        truncated = False


        if done or truncated:
    # Use MetricsCollector's finalize method
            episode_metrics = self.ship.metrics.finalize(float(self.env.now))
            
            # time-average AoI (seconds)
            time_avg_aoi = episode_metrics.aoi_integral / (self.episode_duration * len(self.sensor_ids))
            
            info.update({
                "aoi_integral": episode_metrics.aoi_integral,
                "time_avg_aoi": time_avg_aoi,
                "delivered": episode_metrics.delivered,
                "expired": episode_metrics.expired,
                "generated": episode_metrics.generated,
                "delivery_rate": episode_metrics.delivery_rate,
                "dropped_sensor_buffer": episode_metrics.dropped_sensor_buffer,
                "dropped_drone_buffer": episode_metrics.dropped_drone_buffer,
                "ship_visits": episode_metrics.ship_visits,
                "sensor_visits_total": episode_metrics.sensor_visits_total,
            })
        

        return obs, reward, done, truncated, info

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _get_obs(self):
        sensors_ordered = sorted(self.sensors, key=lambda s: s.id)
        return build_observation(
            drone=self.drone,
            sensors= sensors_ordered,
            ship=self.ship,
            current_time=self.env.now,
            config=self.config,
            rl_memory=self.memory,
            episode_duration=self.episode_duration,
        )
    
    def action_masks(self) -> np.ndarray:
        mask = np.ones(self.num_sensors + 1, dtype=bool)

        if self.drone.is_buffer_full(self.config):
            mask[:self.num_sensors] = False  # Block all sensors
            mask[self.num_sensors] = True
            return mask    # Allow ship (last index)

        if self.last_action is not None:
            mask[self.last_action] = False

            # FORCE SHIP: If buffer full, only allow ship action'
        return mask

    def _compute_travel_time(self, target) -> float:
        return self.drone.calculate_travel_time(target.position, self.config)
    

    def _handle_arrival(self, target, target_type) -> int:
        now = float(self.env.now)

        if target_type == "sensor":
            count = self.drone.collect_from_sensor(target, self.config, now)
            tot_msg = count
            sid = int(target.id)
            self.memory.last_sensor_visit_time[sid] = now
            self.memory.last_sensor_pickup_count[sid] = count
            
            # Log sensor visit
            self.ship.metrics.log_sensor_visit(sid)

        elif target_type == "ship":
            delivered_count, delivered_msgs = self.drone.deliver_to_ship(target, self.config, now)
            tot_msg = delivered_count
            self.memory.last_ship_visit_time = now
            self.total_delivered += delivered_count
            
            # Log ship visit
            self.ship.metrics.log_ship_visit()

            for msg in delivered_msgs:
                sid = int(getattr(msg, "source_id", -1))
                if sid >= 0:
                    prev = self.memory.last_known_ship_gen_time.get(sid, 0.0)
                    self.memory.last_known_ship_gen_time[sid] = max(prev, float(msg.generation_time))

        else:
            raise ValueError(f"Unknown target type: {target_type}")

        return tot_msg
    
    def _max_time_since_any_sensor_visit(self, now: float) -> float:
        """Return the maximum time since last visit to any sensor."""
        if not self.sensor_ids:
            return 0.0
        
        worst = 0.0
        denom = float(self.episode_duration)

        for sid in self.sensor_ids:
            last = float(self.memory.last_sensor_visit_time.get(sid, 0.0))
            t = (now - last) / denom
            t = float(np.clip(t, 0.0, 1.0))
            if t > worst:
                worst = t
        return worst