import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib
from holoocean.sensors import BSTSensor

warnings.filterwarnings("ignore")


class BST_Visualizer:
    def __init__(
        self,
        config=None,
        biomass_function=None,
        salinity_function=None,
        temperature_function=None,
        agent=None,
    ):
        # Position and settings
        self.position = np.array([0.0, 0.0, 0.0])
        self.range_val = 75.0
        self.resolution = 25
        self.config = config or {}

        # Functions - keep original model functions exactly as they were
        if biomass_function is not None:
            self.biomass_function = self.make_bst_compatible(biomass_function)
        else:
            self.biomass_function = BSTSensor.biomass_model

        if salinity_function is not None:
            self.salinity_function = self.make_bst_compatible(salinity_function)
        else:
            self.salinity_function = BSTSensor.salinity_model

        if temperature_function is not None:
            self.temperature_function = self.make_bst_compatible(temperature_function)
        else:
            self.temperature_function = BSTSensor.temperature_model

        # Here we grab the bst range configuration from the original config. We use the next() function, which iterates
        # through an iterator and checks a condition, returning the first item that matches the condition
        self.agent = agent or self.config["main_agent"]
        self.agent_list = self.config["agents"]
        self.sensor_list = next(
            agent["sensors"]
            for agent in self.agent_list
            if agent["agent_name"] == self.agent
        )
        self.bst_config = next(
            sensor["configuration"]
            for sensor in self.sensor_list
            if sensor["sensor_type"] == "BSTSensor"
        )

        self.biomass_range = self.bst_config.get("biomass_range", (0, 3))
        self.salinity_range = self.bst_config.get("salinity_range", (32.75, 34))
        self.temperature_range = self.bst_config.get("temperature_range", (5, 25))
        # print(f"Biomass Range: {self.biomass_range}\nSalinity Range: {self.salinity_range}\nTemperature Range: {self.temperature_range}")

        # Function info for plotting
        self.functions = [
            (self.biomass_function, "Biomass", "kg/m³", "Greens"),
            (self.salinity_function, "Salinity", "psu", "Blues"),
            (self.temperature_function, "Temperature", "°C", "Reds"),
        ]

        # Plot elements
        self.fig = None
        self.axes = None
        self.images = [[None, None] for _ in range(3)]
        self.crosshairs = [[None, None] for _ in range(3)]
        self.value_texts = [[None, None] for _ in range(3)]
        self.colorbars = [[None, None] for _ in range(3)]
        self.initialized = False

        self.setup_plots()

    @staticmethod
    def make_bst_compatible(user_func):
        """
        Wrapper to make user functions compatible with BST visualizer.
        NOTE: BST visualizer requires handling of arrays of locations because it's generating
        hundreds of heat map points all at once, so this wrapper makes it a lot easier for the user

        Users can write simple functions like:
        def my_temp(location):
            x, y, z = location
            return some_calculation

        Then wrap it: compatible_func = make_bst_compatible(my_temp)
        """

        def wrapped_func(location, **kwargs):
            location = np.array(location)

            # Handle single point
            if location.ndim == 1:
                try:
                    result = user_func(location)
                    return float(result) if np.isfinite(result) else 0.0
                except:
                    return 0.0

            # Handle array of points - apply function to each point
            results = []
            for point in location:
                try:
                    result = user_func(point)
                    # Set above-surface points to zero
                    if point[2] > 0:
                        result = 0.0
                    results.append(result if np.isfinite(result) else 0.0)
                except:
                    results.append(0.0)

            return np.array(results)

        return wrapped_func

    def setup_plots(self):
        """Initialize matplotlib plots"""
        try:
            matplotlib.use("TkAgg")
            plt.ioff()

            self.fig, self.axes = plt.subplots(3, 2, figsize=(10, 11))
            self.fig.patch.set_facecolor("black")
            plt.subplots_adjust(
                hspace=0.8, wspace=0.4, top=0.85, bottom=0.08, left=0.08, right=0.92
            )

            titles = ["Biomass", "Salinity", "Temperature"]
            for i in range(3):
                for j in range(2):
                    ax = self.axes[i, j]
                    ax.set_facecolor("black")

                    plane = "XY" if j == 0 else "XZ"
                    ax.set_title(
                        f"{titles[i]} - {plane} Plane",
                        color="white",
                        fontsize=10,
                        pad=8,
                    )
                    ax.tick_params(colors="white", labelsize=8)

                    if j == 0:  # XY plane
                        ax.set_xlabel("X Position (m)", color="white", fontsize=9)
                        ax.set_ylabel("Y Position (m)", color="white", fontsize=9)
                    else:  # XZ plane
                        ax.set_xlabel("X Position (m)", color="white", fontsize=9)
                        ax.set_ylabel("Z Depth (m)", color="white", fontsize=9)

            self.fig.canvas.draw()
            self.initialized = True

        # This is a fail safe in case the graph doesn't load correctly. It will try a couple other methods before giving up.
        except Exception as e:
            print(f"Error setting up plots: {e}")
            for backend in ["Qt5Agg", "Agg"]:
                try:
                    matplotlib.use(backend)
                    self.fig, self.axes = plt.subplots(3, 2, figsize=(10, 11))
                    self.fig.patch.set_facecolor("black")
                    plt.subplots_adjust(
                        hspace=0.8,
                        wspace=0.4,
                        top=0.85,
                        bottom=0.08,
                        left=0.08,
                        right=0.92,
                    )
                    self.initialized = True
                    print(f"Using {backend} backend")
                    break
                except:
                    continue

    def generate_data_xy(self, func):
        """Generate XY plane data using vectorized locations"""
        try:
            # This creates 1d arrays in the x and y direction around the current position. Then, these arrays
            # are meshed into separate coordinate planes that can be manipulated in matrix operations, which
            # is much more efficient than going point by point. For example, if the range was 1, and pos was (0,0):
            # Y = [[-1, -1, -1], [0, 0, 0], [[1, 1, 1]] X = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], and you can
            # multiply them together to evaluate an equation at all possible X and Y coordinates.
            pos = self.position
            x_range = np.linspace(
                pos[0] - self.range_val, pos[0] + self.range_val, self.resolution
            )
            y_range = np.linspace(
                pos[1] - self.range_val, pos[1] + self.range_val, self.resolution
            )
            X, Y = np.meshgrid(x_range, y_range)

            # .ravel() makes the two 2D grids of X and Y coordinates into 1D arrays of those same X and Y values
            # to be more easily iterated over. Ex: [[x1, x2, x3],[x1,x2,x3]...] -> [x1, x2, x3, x1, x2...]. Then,
            # a 1D z array of equal size is created, and a 2nd dimension is added with column_stack(), grouping
            # each group of three coordinates into their own coordinate to be evaluated. Ex: [[x1, y1, z0]...]
            locations = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, pos[2])])

            # Access correct bst_sensor to probe at
            agents = self.config["agents"]
            for agent in agents:
                if agent["agent_name"] == self.agent:
                    probe_agent = agent
                    break

            sensors = probe_agent["sensors"]
            for sensor in sensors:
                if sensor["sensor_type"] == "BSTSensor":
                    bst_sensor = sensor
                    break

            # Evaluate function across entire vector at once, then turn that 1D array into a 2D array where matrix
            # position represents location for the values. Ex: If positions were 3x3...[[val1, val2, val3],[val4,
            # val5, val6]...]
            Z_data = func(location=locations, **bst_sensor["configuration"]).reshape(
                X.shape
            )

            # This format of return helps map the correct values to their coordinates
            return X, Y, Z_data

        except Exception as e:
            print(f"Error generating XY data: {e}")
            x = np.linspace(-10, 10, 10)
            y = np.linspace(-10, 10, 10)
            X, Y = np.meshgrid(x, y)
            return X, Y, np.zeros_like(X)

    def generate_data_xz(self, func):
        """Generate XZ plane data using vectorized locations"""
        try:
            pos = self.position
            x_range = np.linspace(
                pos[0] - self.range_val, pos[0] + self.range_val, self.resolution
            )
            z_range = np.linspace(
                pos[2] - self.range_val, pos[2] + self.range_val, self.resolution
            )
            X, Z = np.meshgrid(x_range, z_range)

            # Create location array for all points in the grid
            locations = np.column_stack([X.ravel(), np.full(X.size, pos[1]), Z.ravel()])

            agents = self.config["agents"]
            for agent in agents:
                if agent["agent_name"] == self.agent:
                    probe_agent = agent
                    break

            sensors = probe_agent["sensors"]
            for sensor in sensors:
                if sensor["sensor_type"] == "BSTSensor":
                    bst_sensor = sensor
                    break

            # Call function with vectorized locations and additional config parameters
            Z_data = func(location=locations, **bst_sensor["configuration"]).reshape(
                X.shape
            )
            return X, Z, Z_data

        except Exception as e:
            print(f"Error generating XZ data: {e}")
            x = np.linspace(-10, 10, 10)
            z = np.linspace(-10, 10, 10)
            X, Z = np.meshgrid(x, z)
            return X, Z, np.zeros_like(X)

    def update_position(self, position):
        """Update position using array input and refresh plots"""
        self.position = np.array(position, dtype=float)
        # print(f"Position updated: ({self.position[0]:.1f}, {self.position[1]:.1f}, {self.position[2]:.1f})")
        self.update_plots()

    def update_plots(self):
        """Update all plots without recreating colorbars"""
        if not self.initialized:
            print("Plots not initialized")
            return False

        try:
            for i, (func, name, unit, colormap) in enumerate(self.functions):
                # print(f"Updating {name} plots...")

                # Get current value at vehicle position
                try:
                    # print(self.config)
                    agents = self.config["agents"]
                    for agent in agents:
                        if agent["agent_name"] == self.agent:
                            probe_agent = agent
                            break

                    sensors = probe_agent["sensors"]
                    for sensor in sensors:
                        if sensor["sensor_type"] == "BSTSensor":
                            bst_sensor = sensor
                            break
                    # print(bst_sensor)
                    current_value = func(
                        location=self.position, **bst_sensor["configuration"]
                    )
                    current_value = (
                        float(current_value) if np.isfinite(current_value) else 0.0
                    )
                except:
                    current_value = 0.0

                # XY plane (left column)
                try:
                    ax_xy = self.axes[i, 0]
                    X_xy, Y_xy, Z_data_xy = self.generate_data_xy(func)

                    if not np.all(Z_data_xy == 0):
                        if self.images[i][0] is None:
                            self.images[i][0] = ax_xy.imshow(
                                Z_data_xy,
                                extent=[X_xy.min(), X_xy.max(), Y_xy.min(), Y_xy.max()],
                                origin="lower",
                                cmap=colormap,
                                aspect="equal",
                            )
                            self.colorbars[i][0] = plt.colorbar(
                                self.images[i][0], ax=ax_xy, shrink=0.7
                            )
                            self.colorbars[i][0].ax.tick_params(
                                colors="white", labelsize=8
                            )
                            self.colorbars[i][0].set_label(
                                f"{name} ({unit})", color="white", fontsize=8
                            )
                        else:
                            self.images[i][0].set_data(Z_data_xy)
                            self.images[i][0].set_extent(
                                [X_xy.min(), X_xy.max(), Y_xy.min(), Y_xy.max()]
                            )

                            # Update colorbar limits - adjusted for new biomass model
                            limits = [
                                self.biomass_range,
                                self.salinity_range,
                                self.temperature_range,
                            ][i]  # biomass, salinity, temperature
                            self.images[i][0].set_clim(limits[0], limits[1])

                    # Remove old crosshair and value text
                    if self.crosshairs[i][0] is not None:
                        for line in self.crosshairs[i][0]:
                            try:
                                line.remove()
                            except:
                                pass
                    if self.value_texts[i][0] is not None:
                        try:
                            self.value_texts[i][0].remove()
                        except:
                            pass

                    # Create crosshair at vehicle position
                    crosshair_size = self.range_val * 0.1
                    h_line = ax_xy.plot(
                        [
                            self.position[0] - crosshair_size,
                            self.position[0] + crosshair_size,
                        ],
                        [self.position[1], self.position[1]],
                        "w-",
                        linewidth=2,
                        alpha=0.9,
                    )[0]
                    v_line = ax_xy.plot(
                        [self.position[0], self.position[0]],
                        [
                            self.position[1] - crosshair_size,
                            self.position[1] + crosshair_size,
                        ],
                        "w-",
                        linewidth=2,
                        alpha=0.9,
                    )[0]
                    center_dot = ax_xy.plot(
                        self.position[0],
                        self.position[1],
                        "wo",
                        markersize=4,
                        markeredgecolor="black",
                        markeredgewidth=1,
                    )[0]

                    self.crosshairs[i][0] = [h_line, v_line, center_dot]

                    # Add current value text
                    self.value_texts[i][0] = ax_xy.text(
                        0.02,
                        0.98,
                        f"{current_value:.2f} {unit}",
                        transform=ax_xy.transAxes,
                        fontsize=9,
                        color="yellow",
                        verticalalignment="top",
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="black",
                            alpha=0.8,
                            edgecolor="yellow",
                        ),
                    )

                    # Update styling
                    ax_xy.set_xlim(
                        self.position[0] - self.range_val,
                        self.position[0] + self.range_val,
                    )
                    ax_xy.set_ylim(
                        self.position[1] - self.range_val,
                        self.position[1] + self.range_val,
                    )
                    ax_xy.set_title(
                        f"{name} - XY Plane (Z={self.position[2]:.1f}m)",
                        color="white",
                        fontsize=10,
                        pad=8,
                    )
                    ax_xy.tick_params(colors="white", labelsize=8)
                    ax_xy.grid(True, alpha=0.3)
                    ax_xy.set_facecolor("black")

                except Exception as e:
                    print(f"Error updating {name} XY plot: {e}")

                # XZ plane (right column)
                try:
                    ax_xz = self.axes[i, 1]
                    X_xz, Z_xz, Z_data_xz = self.generate_data_xz(func)

                    if not np.all(Z_data_xz == 0):
                        if self.images[i][1] is None:
                            self.images[i][1] = ax_xz.imshow(
                                Z_data_xz,
                                extent=[X_xz.min(), X_xz.max(), Z_xz.min(), Z_xz.max()],
                                origin="lower",
                                cmap=colormap,
                                aspect="equal",
                            )
                            self.colorbars[i][1] = plt.colorbar(
                                self.images[i][1], ax=ax_xz, shrink=0.7
                            )
                            self.colorbars[i][1].ax.tick_params(
                                colors="white", labelsize=8
                            )
                            self.colorbars[i][1].set_label(
                                f"{name} ({unit})", color="white", fontsize=8
                            )
                        else:
                            self.images[i][1].set_data(Z_data_xz)
                            self.images[i][1].set_extent(
                                [X_xz.min(), X_xz.max(), Z_xz.min(), Z_xz.max()]
                            )

                            # Update colorbar limits - adjusted for new biomass model
                            limits = [
                                self.biomass_range,
                                self.salinity_range,
                                self.temperature_range,
                            ][i]  # biomass, salinity, temperature
                            self.images[i][1].set_clim(limits[0], limits[1])

                    # Remove old crosshair and value text
                    if self.crosshairs[i][1] is not None:
                        for line in self.crosshairs[i][1]:
                            try:
                                line.remove()
                            except:
                                pass
                    if self.value_texts[i][1] is not None:
                        try:
                            self.value_texts[i][1].remove()
                        except:
                            pass

                    # Create crosshair at vehicle position
                    crosshair_size = self.range_val * 0.1
                    h_line = ax_xz.plot(
                        [
                            self.position[0] - crosshair_size,
                            self.position[0] + crosshair_size,
                        ],
                        [self.position[2], self.position[2]],
                        "w-",
                        linewidth=2,
                        alpha=0.9,
                    )[0]
                    v_line = ax_xz.plot(
                        [self.position[0], self.position[0]],
                        [
                            self.position[2] - crosshair_size,
                            self.position[2] + crosshair_size,
                        ],
                        "w-",
                        linewidth=2,
                        alpha=0.9,
                    )[0]
                    center_dot = ax_xz.plot(
                        self.position[0],
                        self.position[2],
                        "wo",
                        markersize=4,
                        markeredgecolor="black",
                        markeredgewidth=1,
                    )[0]

                    self.crosshairs[i][1] = [h_line, v_line, center_dot]

                    # Add current value text
                    self.value_texts[i][1] = ax_xz.text(
                        0.02,
                        0.98,
                        f"{current_value:.2f} {unit}",
                        transform=ax_xz.transAxes,
                        fontsize=9,
                        color="yellow",
                        verticalalignment="top",
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="black",
                            alpha=0.8,
                            edgecolor="yellow",
                        ),
                    )

                    # Update styling
                    ax_xz.set_xlim(
                        self.position[0] - self.range_val,
                        self.position[0] + self.range_val,
                    )
                    ax_xz.set_ylim(
                        self.position[2] - self.range_val,
                        self.position[2] + self.range_val,
                    )
                    ax_xz.set_title(
                        f"{name} - XZ Plane (Y={self.position[1]:.1f}m)",
                        color="white",
                        fontsize=10,
                        pad=8,
                    )
                    ax_xz.tick_params(colors="white", labelsize=8)
                    ax_xz.grid(True, alpha=0.3)
                    ax_xz.set_facecolor("black")

                except Exception as e:
                    print(f"Error updating {name} XZ plot: {e}")

            # Update main title
            pos = self.position
            self.fig.suptitle(
                f"BST Data - Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
                color="white",
                fontsize=12,
                y=0.95,
            )

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            # print("Plots updated successfully")
            return True

        except Exception as e:
            print(f"Error updating plots: {e}")
            return False

    def show(self):
        """Show the visualizer"""
        if not self.initialized:
            print("Cannot show - plots not initialized")
            return

        try:
            self.update_plots()
            plt.ion()
            plt.show(block=False)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            # print("Visualizer displayed. Use update_position([x, y, z]) to update.")

        except Exception as e:
            print(f"Error showing plots: {e}")

    def close(self):
        """Close the visualizer"""
        try:
            if self.colorbars:
                for i in range(3):
                    for j in range(2):
                        if self.colorbars[i][j] is not None:
                            try:
                                self.colorbars[i][j].remove()
                            except:
                                pass

            if self.fig:
                plt.close(self.fig)
        except Exception as e:
            print(f"Warning during cleanup: {e}")
        finally:
            print("Visualizer closed")
