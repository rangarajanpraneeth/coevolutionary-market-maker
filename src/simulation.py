#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

tf.get_logger().setLevel("ERROR")


class MarketData:
    def __init__(
        self, ohlcv_filename, volatility_window=20, momentum_window=5, rsi_window=14
    ):
        self.filename = ohlcv_filename
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Missing data file: {self.filename}")

        self.data = pd.read_csv(self.filename)
        required_cols = ["High", "Low", "Volume"]

        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

        original_len = len(self.data)
        self.data.dropna(subset=required_cols, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        self.data["mid_price"] = (self.data["High"] + self.data["Low"]) / 2.0
        self.max_steps = len(self.data)

        self.data["returns"] = self.data["mid_price"].pct_change()
        self.data["volatility"] = (
            self.data["returns"].rolling(window=volatility_window).std().fillna(0)
        )
        self.data["price_change"] = self.data["mid_price"].diff()
        self.data["momentum"] = (
            self.data["price_change"].rolling(window=momentum_window).mean().fillna(0)
        )

        rolling_vol_mean = (
            self.data["Volume"].rolling(window=volatility_window).mean().fillna(1)
        )
        self.data["norm_volume"] = (
            self.data["Volume"] / (rolling_vol_mean + 1e-6)
        ).fillna(0)

        delta = self.data["mid_price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / (loss + 1e-6)
        self.data["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    def get_market_state(self, step):
        if step >= self.max_steps:
            return None, None, None, None, None
        row = self.data.loc[step]
        return (
            row["mid_price"],
            row["volatility"],
            row["momentum"],
            row["norm_volume"],
            row["rsi"],
        )


class DQNAgent:
    def __init__(self, state_size, action_size, nn_architecture, drl_params):
        self.state_size = state_size
        self.actions = [(0.1, 0.1), (0.2, 0.2), (0.4, 0.4), (0.3, 0.1), (0.1, 0.3)]
        self.action_size = action_size

        self.inventory = 0
        self.cash = 0.0

        default_params = {
            "gamma": 0.95,
            "epsilon": 1.0,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "learning_rate": 0.001,
            "replay_buffer_maxlen": 2000,
            "batch_size": 32,
        }

        self.drl_params = default_params
        self.drl_params.update(drl_params or {})

        self.replay_buffer = deque(maxlen=self.drl_params["replay_buffer_maxlen"])
        self.gamma = self.drl_params["gamma"]
        self.epsilon = self.drl_params["epsilon"]
        self.epsilon_decay = self.drl_params["epsilon_decay"]
        self.epsilon_min = self.drl_params["epsilon_min"]
        self.learning_rate = self.drl_params["learning_rate"]
        self.batch_size = self.drl_params["batch_size"]
        self.nn_architecture = nn_architecture

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(
            Dense(
                self.nn_architecture[0]["units"],
                input_dim=self.state_size,
                activation=self.nn_architecture[0]["activation"],
            )
        )
        for layer in self.nn_architecture[1:]:
            model.add(Dense(layer["units"], activation=layer["activation"]))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def format_state(
        self, inventory, volatility, momentum, norm_volume, rsi, step, max_steps
    ):
        norm_inventory = np.tanh(inventory / 5.0)
        norm_time = step / max_steps
        norm_rsi = (rsi / 50.0) - 1.0
        norm_vol = np.tanh(norm_volume - 1.0)
        return np.array(
            [[norm_inventory, volatility, momentum, norm_vol, norm_rsi, norm_time]]
        )

    def remember(self, state, action_idx, reward, next_state, done):
        self.replay_buffer.append((state, action_idx, reward, next_state, done))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        minibatch = random.sample(self.replay_buffer, self.batch_size)
        states = np.squeeze(np.array([item[0] for item in minibatch]))
        actions = np.array([item[1] for item in minibatch])
        rewards = np.array([item[2] for item in minibatch])
        next_states = np.squeeze(np.array([item[3] for item in minibatch]))
        dones = np.array([item[4] for item in minibatch])

        target_q_next = self.target_model.predict(next_states, verbose=0)
        best_actions_next = np.argmax(
            self.model.predict(next_states, verbose=0), axis=1
        )
        target_q_values = target_q_next[range(self.batch_size), best_actions_next]
        targets = rewards + self.gamma * target_q_values * (1 - dones)

        current_q = self.model.predict(states, verbose=0)
        for i in range(self.batch_size):
            current_q[i, actions[i]] = targets[i]

        self.model.fit(states, current_q, epochs=1, verbose=0)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class Market:
    def __init__(self, nn_architecture, drl_params, ohlcv_filename, output_dir):
        self.output_dir = output_dir or "."
        os.makedirs(self.output_dir, exist_ok=True)

        base = os.path.basename(ohlcv_filename)
        self.prefix = os.path.splitext(base)[0].lower()

        self.market_data = MarketData(ohlcv_filename=ohlcv_filename)
        self.step = 0

        self.state_size = 6
        self.action_size = 5
        self.drl_agent = DQNAgent(
            self.state_size, self.action_size, nn_architecture, drl_params
        )

        self.INITIAL_CAPITAL = 100000.0
        self.drl_agent.cash = self.INITIAL_CAPITAL

        self.UPDATE_TARGET_EVERY = 20

        self.egt_strategies = ["aggressive", "passive", "random", "momentum"]
        self.egt_proportions = np.array([0.25, 0.25, 0.25, 0.25])
        self.egt_total_payoffs = np.zeros(len(self.egt_strategies))
        self.egt_total_trades = np.zeros(len(self.egt_strategies))
        self.N_EGT_AGENTS_PER_STEP = 15
        self.EVOLVE_EVERY = 50

        self.transaction_cost_per_trade = 0.01

        self.history = {
            "step": [],
            "mid_price": [],
            "drl_profit": [],
            "portfolio_value": [],
            "drl_inventory": [],
            "chosen_spread_width": [],
            "epsilon": [],
            "reward": [],
            "egt_prop_aggressive": [],
            "egt_prop_passive": [],
            "egt_prop_random": [],
            "egt_prop_momentum": [],
        }

        self.last_drl_value = self.INITIAL_CAPITAL

    def get_egt_action(self, strategy_name, mid_price, momentum):
        action_type = random.choice(["buy", "sell"])

        if strategy_name == "aggressive":
            price_offset = random.uniform(0.02, 0.08)
        elif strategy_name == "passive":
            price_offset = random.uniform(0.05, 0.15)
        elif strategy_name == "momentum":
            if momentum > 0:
                action_type = "buy"
            elif momentum < 0:
                action_type = "sell"
            price_offset = random.uniform(0.04, 0.12)
        else:
            price_offset = random.uniform(0.01, 0.5)

        return (
            ("buy", mid_price + price_offset)
            if action_type == "buy"
            else ("sell", mid_price - price_offset)
        )

    def evolve_population(self):
        fitness = self.egt_total_payoffs / (self.egt_total_trades + 1e-6)
        positive_fitness = fitness - np.min(fitness) + 1
        avg_fitness = np.dot(self.egt_proportions, positive_fitness)
        if avg_fitness == 0:
            return

        self.egt_proportions = self.egt_proportions * (positive_fitness / avg_fitness)

        mutation = 0.001
        self.egt_proportions = self.egt_proportions * (1 - mutation) + (
            mutation / len(self.egt_strategies)
        )
        self.egt_proportions /= np.sum(self.egt_proportions)

        self.egt_total_payoffs.fill(0)
        self.egt_total_trades.fill(0)

    def run_step(self):
        mid_price, volatility, momentum, norm_volume, rsi = (
            self.market_data.get_market_state(self.step)
        )
        if mid_price is None:
            return False

        state = self.drl_agent.format_state(
            self.drl_agent.inventory,
            volatility,
            momentum,
            norm_volume,
            rsi,
            self.step,
            self.market_data.max_steps,
        )
        action_idx = self.drl_agent.choose_action(state)
        bid_spread, ask_spread = self.drl_agent.actions[action_idx]

        inv_bias = np.clip(abs(self.drl_agent.inventory) / 5.0, 1.0, 3.0)
        if self.drl_agent.inventory > 0:
            bid_spread *= inv_bias
            ask_spread /= inv_bias
        elif self.drl_agent.inventory < 0:
            bid_spread /= inv_bias
            ask_spread *= inv_bias

        drl_bid_price = mid_price - bid_spread
        drl_ask_price = mid_price + ask_spread

        step_payoffs = np.zeros(len(self.egt_strategies))
        step_trades = np.zeros(len(self.egt_strategies))
        drl_trades_this_step = 0
        trade_profit_total = 0.0

        sampled_agents = np.random.choice(
            self.egt_strategies, size=self.N_EGT_AGENTS_PER_STEP, p=self.egt_proportions
        )

        for strategy_name in sampled_agents:
            egt_action, egt_price = self.get_egt_action(
                strategy_name, mid_price, momentum
            )
            strat_idx = self.egt_strategies.index(strategy_name)

            if egt_action == "sell" and egt_price <= drl_bid_price:
                self.drl_agent.inventory += 1
                self.drl_agent.cash -= drl_bid_price
                step_payoffs[strat_idx] += drl_bid_price
                step_trades[strat_idx] += 1
                drl_trades_this_step += 1
                trade_profit_total += mid_price - drl_bid_price

            elif egt_action == "buy" and egt_price >= drl_ask_price:
                self.drl_agent.inventory -= 1
                self.drl_agent.cash += drl_ask_price
                step_payoffs[strat_idx] -= drl_ask_price
                step_trades[strat_idx] += 1
                drl_trades_this_step += 1
                trade_profit_total += drl_ask_price - mid_price

        self.egt_total_payoffs += step_payoffs
        self.egt_total_trades += step_trades

        inv = self.drl_agent.inventory
        portfolio_value = self.drl_agent.cash + (inv * mid_price)

        abs_change = portfolio_value - self.last_drl_value
        rel_change = abs_change / max(self.last_drl_value, 1e-8)
        reward = 0.5 * abs_change + 0.5 * (rel_change * 1000.0)

        reward -= 0.002 * abs(inv)
        reward -= 0.0005 * (inv**2)

        reward -= drl_trades_this_step * self.transaction_cost_per_trade

        prev_inv = (
            self.history["drl_inventory"][-1] if self.history["drl_inventory"] else 0
        )
        if abs(inv) < abs(prev_inv):
            reward += 0.003 * (abs(prev_inv) - abs(inv))

        reward += 0.002 * (1 - min(abs(inv) / 10.0, 1.0))
        reward += 0.002 * trade_profit_total
        reward -= 0.0015 * inv

        self.last_drl_value = portfolio_value

        next_data_tuple = self.market_data.get_market_state(self.step + 1)
        next_price, next_vol, next_mom, next_norm_vol, next_rsi = next_data_tuple
        done = next_price is None

        next_state = (
            state
            if done
            else self.drl_agent.format_state(
                inv,
                next_vol,
                next_mom,
                next_norm_vol,
                next_rsi,
                self.step + 1,
                self.market_data.max_steps,
            )
        )

        self.drl_agent.remember(state, action_idx, reward, next_state, done)
        self.drl_agent.replay()

        if self.step % self.EVOLVE_EVERY == 0:
            self.evolve_population()
        if self.step % self.UPDATE_TARGET_EVERY == 0:
            self.drl_agent.update_target_model()

        self.history["step"].append(self.step)
        self.history["mid_price"].append(mid_price)
        self.history["drl_profit"].append(portfolio_value)
        self.history["portfolio_value"].append(portfolio_value)
        self.history["drl_inventory"].append(inv)
        self.history["chosen_spread_width"].append(bid_spread + ask_spread)
        self.history["epsilon"].append(self.drl_agent.epsilon)
        self.history["reward"].append(reward)

        for i, strat in enumerate(self.egt_strategies):
            self.history[f"egt_prop_{strat}"].append(self.egt_proportions[i])

        self.step += 1
        return True

    def run_simulation(self):
        while self.run_step():
            if self.step % 100 == 0:
                print(
                    f"Step {self.step}/{self.market_data.max_steps}... Epsilon: {self.drl_agent.epsilon:.2f}"
                )

        print("Simulation complete.")
        self.print_summary()
        self.plot_results()

    def print_summary(self):
        if not self.history["portfolio_value"]:
            print("Simulation ended before any history was recorded.")
            return

        from scipy.stats import skew, kurtosis

        pv = np.array(self.history["portfolio_value"], dtype=float)
        final_portfolio = pv[-1]

        if len(pv) > 1:
            step_returns = np.diff(pv) / pv[:-1]
        else:
            step_returns = np.array([0.0])

        mean_ret = float(np.mean(step_returns)) if len(step_returns) > 0 else 0.0
        std_ret = float(np.std(step_returns)) if len(step_returns) > 0 else 0.0
        skew_ret = float(skew(step_returns)) if len(step_returns) > 1 else 0.0
        kurt_ret = float(kurtosis(step_returns)) if len(step_returns) > 1 else 0.0

        sharpe = (mean_ret / (std_ret + 1e-8)) if len(step_returns) > 1 else 0.0

        rolling_max = np.maximum.accumulate(pv)
        drawdowns = pv / rolling_max - 1.0
        max_drawdown = float(np.min(drawdowns))

        total_return_pct = (final_portfolio / self.INITIAL_CAPITAL - 1.0) * 100.0

        steps_per_year = 13000
        sharpe_annualized = sharpe * np.sqrt(steps_per_year)

        print("\n" + "=" * 30)
        print("=== FINAL RESULTS ===")
        print(f"Starting Capital: ${self.INITIAL_CAPITAL:,.2f}")
        print(f"Final Portfolio Value: ${final_portfolio:,.2f}")
        print(f"Total Return: {total_return_pct:.2f}%")
        print(f"Sharpe (per-step signal/noise): {sharpe:.3f}")
        print(f"Sharpe (annualized): {sharpe_annualized:.3f}")
        print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
        print(f"Mean Return per Step: {mean_ret:.6f}")
        print(f"Std Dev of Return per Step: {std_ret:.6f}")
        print(f"Skewness of Returns: {skew_ret:.6f}")
        print(f"Kurtosis of Returns: {kurt_ret:.6f}")
        print("\nFinal EGT Population Distribution:")
        for i, strat in enumerate(self.egt_strategies):
            print(f"  {strat.capitalize()}: {self.egt_proportions[i]:.3f}")
        print("=" * 30 + "\n")

    def plot_results(self):
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from scipy.ndimage import gaussian_filter, gaussian_filter1d
        from scipy.interpolate import make_interp_spline

        STD_FIGSIZE = (8, 6)
        STD_DPI = 300

        df_history = pd.DataFrame(self.history)
        if df_history.empty:
            print("History is empty, skipping plotting.")
            return

        print("Saving plots...")

        def out(name):
            return os.path.join(self.output_dir, f"{self.prefix}_{name}")

        fig1, ax1 = plt.subplots(figsize=STD_FIGSIZE)
        ax1.plot(
            df_history["step"],
            df_history["portfolio_value"],
            label="Portfolio Value ($)",
            color="darkblue",
        )
        ax1.set_title("DRL Agent Portfolio Value Over Time")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        plt.tight_layout()
        plt.savefig(out("plot_1_portfolio_value.png"), dpi=STD_DPI)
        plt.close(fig1)

        inventory_series = np.array(df_history["drl_inventory"], dtype=float)
        steps_arr = np.array(df_history["step"], dtype=float)

        if len(steps_arr) > 0:
            time_normalized = (
                steps_arr / steps_arr.max() if steps_arr.max() != 0 else steps_arr
            )
        else:
            time_normalized = steps_arr

        bins_x = 256
        bins_y = 256
        heatmap, xedges, yedges = np.histogram2d(
            time_normalized, inventory_series, bins=[bins_x, bins_y], density=True
        )
        heatmap_smooth = gaussian_filter(heatmap, sigma=3)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        im = ax2.imshow(
            heatmap_smooth.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            interpolation="bicubic",
        )

        ax2.plot(
            time_normalized,
            inventory_series,
            color="white",
            alpha=0.25,
            linewidth=0.6,
            label="Inventory",
        )

        if len(inventory_series) > 3:
            smoothed_inventory = gaussian_filter1d(
                inventory_series, sigma=len(inventory_series) / 40
            )
            x_smooth = np.linspace(time_normalized.min(), time_normalized.max(), 1000)
            spline = make_interp_spline(time_normalized, smoothed_inventory, k=3)
            y_smooth = spline(x_smooth)
            ax2.plot(
                x_smooth,
                y_smooth,
                color="white",
                linewidth=2.0,
                alpha=0.95,
                label="Smoothed Avg",
            )

        ax2.set_title("Inventory Distribution Over Time (Heatmap)")
        ax2.set_xlabel("Time [-]")
        ax2.set_ylabel("Inventory [units]")
        ax2.legend(loc="upper left", fontsize=8, facecolor="black", framealpha=0.3)

        cbar = fig2.colorbar(im, ax=ax2)
        cbar.set_label("Relative Frequency Density")

        plt.tight_layout()
        plt.savefig(out("plot_2_inventory_heatmap.png"), dpi=STD_DPI)
        plt.close(fig2)

        pv = df_history["portfolio_value"].to_numpy(dtype=float)
        rolling_max = np.maximum.accumulate(pv)
        drawdowns = pv / rolling_max - 1.0

        fig_dd, ax_dd = plt.subplots(figsize=STD_FIGSIZE)
        ax_dd.plot(df_history["step"], drawdowns * 100.0, color="firebrick")
        ax_dd.set_title("Drawdown (%) Over Time")
        ax_dd.set_xlabel("Time Step")
        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax_dd.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out("plot_support_drawdown.png"), dpi=STD_DPI)
        plt.close(fig_dd)

        if len(pv) > 1:
            step_returns = np.diff(pv) / pv[:-1]
        else:
            step_returns = np.array([0.0])

        fig_ret, ax_ret = plt.subplots(figsize=STD_FIGSIZE)
        ax_ret.hist(step_returns, bins=30, alpha=0.7, edgecolor="black")
        ax_ret.set_title("Distribution of Step Returns")
        ax_ret.set_xlabel("Return per Step")
        ax_ret.set_ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(out("plot_support_return_dist.png"), dpi=STD_DPI)
        plt.close(fig_ret)

        fig3, ax3 = plt.subplots(figsize=STD_FIGSIZE)
        egt_labels = [s.capitalize() for s in self.egt_strategies]
        egt_data = [df_history[f"egt_prop_{s}"] for s in self.egt_strategies]
        ax3.stackplot(df_history["step"], egt_data, labels=egt_labels)
        ax3.set_title("EGT Population Evolution (Replicator Dynamics)")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Proportion of Population")
        ax3.set_ylim(0, 1)
        ax3.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(out("plot_3_egt_dynamics.png"), dpi=STD_DPI)
        plt.close(fig3)

        fig4, ax4 = plt.subplots(figsize=STD_FIGSIZE)
        ax4.plot(
            df_history["step"],
            df_history["chosen_spread_width"],
            label="Chosen Spread Width",
            color="red",
            alpha=0.6,
        )
        ax4.set_title("DRL Agent's Quoted Spread Width")
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Spread Width ($)")
        ax4.grid(True, alpha=0.3)

        ax4_twin = ax4.twinx()
        ax4_twin.plot(
            df_history["step"],
            df_history["epsilon"],
            label="Epsilon",
            color="grey",
            linestyle=":",
        )
        ax4_twin.set_ylabel("Epsilon")

        lines, labels = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines + lines2, labels + labels2, loc="upper right")

        plt.tight_layout()
        plt.savefig(out("plot_4_drl_spread.png"), dpi=STD_DPI)
        plt.close(fig4)

        df_history["rolling_avg_reward"] = (
            df_history["reward"].rolling(window=100, min_periods=1).mean()
        )

        fig5, ax5 = plt.subplots(figsize=STD_FIGSIZE)
        ax5.plot(
            df_history["step"],
            df_history["rolling_avg_reward"],
            label="Rolling Avg. Reward (100 steps)",
            color="purple",
        )
        ax5.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax5.set_title("DRL Agent's Progressive Learning (Rolling Reward)")
        ax5.set_xlabel("Time Step")
        ax5.set_ylabel("Average Reward")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out("plot_5_rolling_reward.png"), dpi=STD_DPI)
        plt.close(fig5)

        fig6, ax6 = plt.subplots(figsize=STD_FIGSIZE)
        halfway_idx = len(df_history["step"]) // 2
        exploiting_spreads = df_history["chosen_spread_width"].iloc[halfway_idx:]

        if not exploiting_spreads.empty:
            spread_widths_rounded = exploiting_spreads.round(2)
            spread_counts = spread_widths_rounded.value_counts().sort_index()
            spread_dist = spread_counts / len(spread_widths_rounded)

            global_min, global_max = 0.001, 0.4
            norm = mcolors.LogNorm(vmin=global_min, vmax=global_max)
            colors = [cm.viridis(norm(v)) for v in spread_dist]

            bars = spread_dist.plot(kind="bar", ax=ax6, color=colors)
            ax6.set_title(f"DRL Learned Policy (Last {len(exploiting_spreads)} Steps)")
            ax6.set_xlabel("Chosen Spread Width ($)")
            ax6.set_ylabel("Proportion of Actions")
            ax6.set_ylim(0, 1)
            ax6.set_xticklabels(spread_dist.index, rotation=0)
            ax6.grid(axis="y", linestyle="--", alpha=0.5)

            for bar, val in zip(bars.patches, spread_dist):
                ax6.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.01,
                    f"{val * 100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
            sm.set_array([])
            cbar = fig6.colorbar(sm, ax=ax6)
            ticks = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t*100:.1f}%" for t in ticks])
            cbar.set_label("Relative Frequency (log scale)", rotation=270, labelpad=15)
        else:
            ax6.set_title("DRL Learned Policy (No data)")

        plt.tight_layout()
        plt.savefig(out("plot_6_policy_dist.png"), dpi=STD_DPI)
        plt.close(fig6)

        fig7, ax7 = plt.subplots(figsize=STD_FIGSIZE)

        df_history["egt_avg"] = df_history[
            [f"egt_prop_{s}" for s in self.egt_strategies]
        ].mean(axis=1)
        df_history["drl_profit_change"] = df_history["drl_profit"].diff().fillna(0)
        df_history["egt_change"] = df_history["egt_avg"].diff().fillna(0)

        rolling_corr = (
            df_history["drl_profit_change"]
            .rolling(window=100, min_periods=10)
            .corr(df_history["egt_change"])
        )

        ax7.plot(
            df_history["step"],
            rolling_corr,
            color="darkorange",
            label="Rolling Corr (DRL Profit vs EGT Mix)",
        )
        ax7.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax7.set_title("Dependency of DRL Agent on EGT Population")
        ax7.set_xlabel("Time Step")
        ax7.set_ylabel("Rolling Correlation")
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out("plot_7_drl_egt_dependency.png"), dpi=STD_DPI)
        plt.close(fig7)

        df_history.to_csv(
            os.path.join(self.output_dir, f"{self.prefix}_history.csv"), index=False
        )

        print(f"All plots + history saved in: {self.output_dir}")
