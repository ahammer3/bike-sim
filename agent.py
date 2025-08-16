import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Rider Agent
# ----------------------------
class Rider(ap.Agent):
    def setup(self):
        self.state = 'idle'
        self.trip_time = 0
        self.destination = None
        self.total_wait_time = 0
        self.wait_instances = 0
        self.trips_attempted = 0
        self.trips_completed = 0
        self.wait_location = None
        self.trigger_trip = False  # NEW: model-driven demand triggers trips instead of agent coin-flips

    def step(self):
        grid = self.model.grid
        x, y = grid.positions[self]

        # CHANGED: No random p_trip here. Trip starts only if model set trigger_trip=True this tick.
        if self.state == 'idle':
            if self.trigger_trip:
                self.trigger_trip = False  # reset latch for next ticks
                self.trips_attempted += 1
                if grid.bikes[x, y] > 0:
                    grid.bikes[x, y] -= 1

                    # Destination choice
                    others = [(i, j) for i in range(10) for j in range(10) if (i, j) != (x, y)]
                    if self.model.use_gravity:
                        # NEW: Gravity model p(d|o) ∝ attractiveness[d] * exp(-beta * manhattan(o,d))
                        beta = self.model.beta_gravity
                        attr = self.model.dest_attr_flat
                        def flat_index(coord):
                            return coord[0] * 10 + coord[1]
                        dists = np.array([abs(i - x) + abs(j - y) for (i, j) in others])
                        base_w = np.exp(-beta * dists)
                        attr_w = np.array([attr[flat_index(c)] for c in others])
                        weights = base_w * attr_w
                        weights = weights / weights.sum()
                        self.destination = self.model.nprandom.choice(others, p=weights)
                    else:
                        # Fallback: use provided flat destination weights (filtered & normalized)
                        def flat_index(coord):
                            return coord[0] * 10 + coord[1]
                        w = np.array([self.model.dest_weights_flat[flat_index(c)] for c in others])
                        w = w / w.sum()
                        self.destination = self.model.nprandom.choice(others, p=w)

                    dest_x, dest_y = self.destination
                    base_time = abs(dest_x - x) + abs(dest_y - y)
                    noise = self.model.nprandom.integers(-1, 2)
                    self.trip_time = max(1, base_time + noise)
                    self.state = 'in_transit'
                else:
                    self.model.happiness -= 1

        elif self.state == 'in_transit':
            self.trip_time -= 1
            if self.trip_time <= 0:
                dest_x, dest_y = self.destination
                grid.move_to(self, (dest_x, dest_y))
                if grid.bikes[dest_x, dest_y] >= grid.capacity[dest_x, dest_y]:
                    if self.model.nprandom.random() < self.p.p_reroute:
                        self.model.happiness -= 5
                        possible = [(i, j) for i in range(10) for j in range(10) if (i, j) not in [(x, y), (dest_x, dest_y)]]
                        min_dist = min(abs(i - dest_x) + abs(j - dest_y) for i, j in possible)
                        closest = [(i, j) for i, j in possible if abs(i - dest_x) + abs(j - dest_y) == min_dist]
                        reroute_to = self.model.nprandom.choice(closest)
                        self.destination = reroute_to
                        rx, ry = reroute_to
                        base_time = abs(rx - dest_x) + abs(ry - dest_y)
                        noise = self.model.nprandom.integers(-1, 2)
                        self.trip_time = max(1, base_time + noise)
                    else:
                        self.state = 'waiting'
                        self.wait_location = (dest_x, dest_y)
                        self.wait_instances += 1
                else:
                    grid.bikes[dest_x, dest_y] += 1
                    self.model.happiness += 1
                    self.trips_completed += 1
                    self.state = 'idle'

        elif self.state == 'waiting':
            self.total_wait_time += 1
            self.model.happiness -= 1
            dest_x, dest_y = self.wait_location
            if grid.bikes[dest_x, dest_y] < grid.capacity[dest_x, dest_y]:
                grid.bikes[dest_x, dest_y] += 1
                self.model.happiness += 1
                self.trips_completed += 1
                self.state = 'idle'
                self.wait_location = None

# ----------------------------
# Citi Bike Model
# ----------------------------
class CitiBikeModel(ap.Model):
    def setup(self):
        self.grid = ap.Grid(self, (10, 10))

        init = np.full((10, 10), self.p.initial_bikes)
        caps = np.full((10, 10), self.p.station_capacity)
        self.grid.add_field('bikes', values=init)
        self.grid.add_field('capacity', values=caps)

        self.happiness = 0
        self.agents = ap.AgentList(self, self.p.n_agents, Rider)

        # --- Demand parameters (NEW) ---
        # Base arrival rate per station per step
        self.lambda_base = getattr(self.p, 'lambda_base', 0.05)
        # Station multipliers (100,) → reshape to (10,10)
        sm = getattr(self.p, 'lambda_station_multipliers', [1.0] * 100)
        self.lambda_station_mult = np.array(sm, dtype=float).reshape(10, 10)
        # Hourly profile (24,) multipliers for time-of-day pattern
        # CHANGED: robustly handle None by replacing with default profile before np.array
        hp = getattr(self.p, 'hourly_profile', None)
        if hp is None:
            hp = [0.3,0.2,0.2,0.2,0.3,0.6,1.0,1.4,1.6,1.2,1.0,0.9,
                  0.9,1.0,1.2,1.4,1.6,1.8,1.6,1.2,1.0,0.7,0.5,0.4]
        self.hourly_profile = np.array(hp, dtype=float)
        # CHANGED: validate length = 24 to avoid indexing errors
        if self.hourly_profile.size != 24:
            raise ValueError("hourly_profile must have length 24")
        # Normalize hourly profile to mean 1.0 to preserve lambda_base scale
        self.hourly_profile = self.hourly_profile / float(self.hourly_profile.mean())

        # --- Destination model params (NEW) ---
        self.use_gravity = getattr(self.p, 'use_gravity', True)
        self.beta_gravity = float(getattr(self.p, 'beta_gravity', 0.3))
        # Destination attractiveness, default uniform ones
        da = getattr(self.p, 'dest_attractiveness', [1.0] * 100)
        self.dest_attr_flat = np.array(da, dtype=float)
        # CHANGED: also keep a normalized flat destination weight vector for fallback mode
        dw = getattr(self.p, 'dest_weights', [1/100] * 100)
        self.dest_weights_flat = np.array(dw, dtype=float)
        self.dest_weights_flat = self.dest_weights_flat / self.dest_weights_flat.sum()

        # --- Placement with optional station_weights (kept) ---
        all_coords = [(i, j) for i in range(10) for j in range(10)]
        if hasattr(self.p, 'station_weights'):
            station_weights = np.array(self.p.station_weights, dtype=float)
            station_weights = station_weights / station_weights.sum()
        else:
            station_weights = np.full(100, 1/100)
        chosen_indices = self.nprandom.choice(len(all_coords), size=self.p.n_agents, p=station_weights)
        positions = [all_coords[i] for i in chosen_indices]
        self.grid.add_agents(self.agents, positions)

    def _trigger_demand(self):
        """NEW: Generate Poisson trip requests at each station this step and trigger idle riders."""
        # Build map from cell -> list of idle agents present
        idle_map = {}
        for a in self.agents:
            if a.state == 'idle':
                pos = self.grid.positions[a]
                idle_map.setdefault(pos, []).append(a)
        # Current hour-of-day from model time t (1 step ≈ 1 minute)
        hour = int((self.t % (24*60)) // 60)
        hour_mult = self.hourly_profile[hour]
        # For each station, sample arrivals and trigger that many idle riders (if available)
        for i in range(10):
            for j in range(10):
                lam = self.lambda_base * self.lambda_station_mult[i, j] * hour_mult
                k = self.nprandom.poisson(lam)
                if k <= 0:
                    continue
                idle_here = idle_map.get((i, j), [])
                if not idle_here:
                    continue
                n = min(k, len(idle_here))
                # random subset of idle agents to trigger
                idx = self.nprandom.choice(len(idle_here), size=n, replace=False)
                for ii in np.atleast_1d(idx):
                    idle_here[int(ii)].trigger_trip = True

    def step(self):
        # NEW: generate demand first, then let agents act
        self._trigger_demand()
        self.agents.step()

    def end(self):
        wait_times = [a.total_wait_time for a in self.agents]
        wait_counts = [a.wait_instances for a in self.agents]
        trip_attempts = [a.trips_attempted for a in self.agents]
        trip_successes = [a.trips_completed for a in self.agents]

        avg_wait_time = sum(wait_times) / sum(wait_counts) if sum(wait_counts) else 0
        trip_success_rate = sum(trip_successes) / sum(trip_attempts) if sum(trip_attempts) else 0

        self.report('happiness', self.happiness)
        self.report('avg_wait_time', avg_wait_time)
        self.report('trip_success_rate', trip_success_rate)

# ----------------------------
# Plot Results (unchanged)
# ----------------------------
def plot_results(results):
    x = [r['initial_bikes'] for r in results]

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(x, [r['happiness'] for r in results], marker='o')
    plt.title('Happiness Score')
    plt.xlabel('Initial Bikes')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(x, [r['avg_wait_time'] for r in results], marker='o', color='orange')
    plt.title('Average Wait Time')
    plt.xlabel('Initial Bikes')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(x, [r['trip_success_rate'] for r in results], marker='o', color='green')
    plt.title('Trip Success Rate')
    plt.xlabel('Initial Bikes')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ----------------------------
# Run Experiments (updated params)
# ----------------------------
def run_experiments():
    results = []
    for init_bikes in range(1, 31):
        params = {
            'initial_bikes': init_bikes,
            'station_capacity': 30,
            'n_agents': 500,
            'steps': 200,
            'p_reroute': 0.5,
            # Demand & destination settings
            'lambda_base': 0.05,  # average arrivals per station per minute
            'lambda_station_multipliers': [1.0] * 100,
            'hourly_profile': None,  # use default profile if None
            'use_gravity': True,
            'beta_gravity': 0.3,
            'dest_attractiveness': [1.0] * 100,
            # Keep density params available
            'station_weights': [1/100] * 100,
            'dest_weights': [1/100] * 100
        }
        model = CitiBikeModel(params)
        output = model.run(display=False)
        results.append({
            'initial_bikes': init_bikes,
            'happiness': output.reporters['happiness'],
            'avg_wait_time': output.reporters['avg_wait_time'],
            'trip_success_rate': output.reporters['trip_success_rate']
        })
    return results

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    results = run_experiments()
    plot_results(results)
