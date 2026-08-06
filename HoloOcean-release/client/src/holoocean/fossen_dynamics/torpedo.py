#!/usr/bin/env python3

# """
# torpedo.py:

#    Class for the a cylinder-shaped autonomous underwater vehicle (AUV),
#    which is controlled using fins at the back and a propeller. The
#    default parameters match the REMUS 100 vehicle.

# References:

#     B. Allen, W. S. Vorus and T. Prestero, "Propulsion system performance
#          enhancements on REMUS AUVs," OCEANS 2000 MTS/IEEE Conference and
#          Exhibition. Conference Proceedings, 2000, pp. 1869-1873 vol.3,
#          doi: 10.1109/OCEANS.2000.882209.
#     T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion
#          Control. 2nd. Edition, Wiley. URL: www.fossen.biz/wiley

# Author:     Thor I. Fossen
# Modified:   Braden Meyers & Carter Noh
# """
import math
import numpy as np
from holoocean.fossen_dynamics.control import integralSMC
from holoocean.fossen_dynamics.helper_functions import (
    crossFlowDrag,
    forceLiftDrag,
    Hmtrx,
    m2c,
    gvect,
    ssa,
)
from holoocean.fossen_dynamics.actuator import fin, thruster


# Class Vehicle
class Torpedo:
    """
    Torpedo()
        Rudder angle, stern plane and propeller revolution step inputs

    Torpedo('depthHeadingAutopilot',z_d,psi_d,n_d,V_c,beta_c)
        Depth and heading autopilots

    Inputs:
        z_d:    desired depth, positive downwards (m)
        psi_d:  desired heading angle (deg)
        n_d:    desired propeller revolution (rpm)
        V_c:    current speed (m/s)
        beta_c: current direction (deg)

    :param str controlSystem: autopilot method for controlling the actuators
    :param float r_z: desired depth (m), positive downwards
    :param float r_psi: desired yaw angle (deg)
    :param float r_rpm: desired propeller revolution (rpm)
    :param float V_current: water current speed (m/s)
    :param float beta_current: water current direction (deg)
    """

    def __init__(
        self,
        controlSystem="manualControl",
        r_z=0,
        r_psi=0,
        r_rpm=0,
        V_current=0,
        beta_current=0,
    ):
        self.D2R = np.pi / 180  # deg2rad
        self.config_fnc = configure_torpedo_from_scenario

        if controlSystem == "depthHeadingAutopilot":
            self.controlDescription = (
                "Depth and heading autopilots, z_d = "
                + str(r_z)
                + ", psi_d = "
                + str(r_psi)
                + " deg"
            )
        elif controlSystem == "manualControl":
            print("Control System: Manual Control")
        else:
            self.controlDescription = (
                "Step inputs for stern planes, rudder and propeller"
            )
            controlSystem = "stepInput"

        self.ref_z = r_z
        self.ref_psi = r_psi
        self.ref_n = r_rpm
        self.V_c = V_current
        self.beta_c = beta_current * self.D2R
        self.controlMode = controlSystem

        # Initialize the AUV model
        self.name = "Torpedo-shaped vehicle based on the REMUS 100 AUV (see 'torpedo.py' for more details)"

        # Default parameters and Constants
        self.rho = 1026  # density of water (kg/m^3)
        g = 9.81  # acceleration of gravity (m/s^2)

        self.L = 1.6  # length (m)
        self.diam = 0.19  # cylinder diameter (m)

        self.nu = np.array([0, 0, 0, 0, 0, 0], float)  # velocity vector

        # Hydrodynamics (Fossen 2021, Section 8.4.2)
        self.r_bg = np.array([0, 0, 0.02], float)  # CG w.r.t. to the CO
        self.r_bb = np.array([0, 0, 0], float)  # CB w.r.t. to the CO

        # Parasitic drag coefficient CD_0, i.e. zero lift and alpha = 0
        # F_drag = 0.5 * rho * Cd * (pi * b^2)
        # F_drag = 0.5 * rho * CD_0 * S
        Cd = 0.42  # from Allen et al. (2000)

        self.e = 0.7  # Oswald efficiency factor for vehicle drag
        self.area_fraction = 0.7  # relates vehicle effective area to length and width. pi/4 for a spheroid
        # Added moment of inertia in roll: A44 = r44 * Ix
        r44 = 0.3

        S_fin = 0.00665
        # fin area
        CL_delta_r = 0.5  # rudder lift coefficient
        CL_delta_s = 0.5  # stern-plane lift coefficient

        self.calculate_vehicle_parameters(g, Cd, r44)
        self.calculate_actuator_parameters(S_fin, CL_delta_r, CL_delta_s)

        # Low-speed linear damping matrix parameters
        self.T_surge = 20  # time constant in surge (s)
        self.T_sway = 20  # time constant in sway (s)
        self.T_heave = self.T_sway  # equal for for a cylinder-shaped AUV
        self.zeta_roll = 0.3  # relative damping ratio in roll
        self.zeta_pitch = 0.8  # relative damping ratio in pitch
        self.T_yaw = 1  # time constant in yaw (s)

        # Feed forward gains (Nomoto gain parameters)
        self.K_nomoto = 5.0 / 20.0  # K_nomoto = r_max / delta_max
        self.T_nomoto = self.T_yaw  # Time constant in yaw

        # Heading autopilot reference model
        self.psi_d = 0  # position, velocity and acc. states
        self.r_d = 0
        self.a_d = 0
        self.wn_d = 0.4  # desired natural frequency
        self.zeta_d = 1.0  # desired realtive damping ratio
        self.r_max = 5.0 * math.pi / 180  # maximum yaw rate

        # Heading autopilot (Equation 16.479 in Fossen 2021)
        # sigma = r-r_d + 2*lambda*ssa(psi-psi_d) + lambda^2 * integral(ssa(psi-psi_d))
        # delta = (T_nomoto * r_r_dot + r_r - K_d * sigma
        #       - K_sigma * (sigma/phi_b)) / K_nomoto
        self.lam = 0.1
        self.phi_b = 0.1  # boundary layer thickness
        self.K_d = 0.5  # PID gain
        self.K_sigma = 0.05  # SMC switching gain

        self.e_psi_int = 0  # yaw angle error integral state

        # Depth autopilot
        # ## Outer Loop
        # Fe = 20     #equilibrium force of thruster calculated at 1000 rpm
        # tr_z = 7
        # wn_z = 2.2/tr_z
        # zeta = 0.707
        # self.Kp_z = m * (wn_z**2) / Fe # heave proportional gain, outer loop
        self.Kp_z = 0.1532
        self.wn_d_z = 0.12  # desired natural frequency, reference model
        self.T_z = 100.0  # heave integral gain, outer loop

        ## Inner Loop
        # tr_theta = 7/10
        # wn_th = 2.2/tr_theta
        # self.Kp_theta = Iy * wn_th**2
        # self.Kd_theta = 2 * zeta * wn_th * Iy - self.zeta_pitch
        self.Kp_theta = 39.78  # pitch proportional gain, inner loop
        self.Kd_theta = 17.1
        self.Ki_theta = 0.5
        self.K_w = 0.0  # optional heave velocity feedback gain
        self.theta_max = (
            15 * self.D2R
        )  # There is a physical limit for theta max given - delta max for fins, and COB - COG offset
        self.wn_d_theta = (
            0.25  # LP filter natural frequency when only running inner loop
        )

        # If offset between ref and current depth is larger than threshold only the inner control loop will be active
        self.outer_loop_threshold = (self.theta_max / self.Kp_z) + 1.2
        self.surge_threshold = (
            0.6  # Surge (x velocity) threshold before starting diving
        )

        # Initialized class variables
        self.z_int = 0  # heave position integral state
        self.z_d = 0  # desired position, LP filter initial state
        self.theta_d = 0  # desired theta, LP filter initial state
        self.theta_int = 0  # pitch angle integral state

        self.init_depth = False
        self.init_heading = False
        self.torque_s = 0

    def calculate_vehicle_parameters(self, g, Cd, r44, m=None):
        self.S = (
            self.area_fraction * self.L * self.diam
        )  # S = 70% of rectangle L * diam
        a = self.L / 2  # semi-axes
        b = self.diam / 2

        self.CD_0 = Cd * math.pi * b**2 / self.S

        # Rigid-body mass matrix expressed in CO
        if m is None:
            m = 4 / 3 * math.pi * self.rho * a * b**2  # mass of spheriod
        Ix = (2 / 5) * m * b**2  # moment of inertia
        Iy = (1 / 5) * m * (a**2 + b**2)
        Iz = Iy
        MRB_CG = np.diag([m, m, m, Ix, Iy, Iz])  # MRB expressed in the CG
        H_rg = Hmtrx(self.r_bg)
        self.MRB = H_rg.T @ MRB_CG @ H_rg  # MRB expressed in the CO

        # Weight and buoyancy
        self.W = m * g
        self.B = self.W

        MA_44 = r44 * Ix

        # Lamb's k-factors
        e = math.sqrt(1 - (b / a) ** 2)
        alpha_0 = (2 * (1 - e**2) / pow(e, 3)) * (0.5 * math.log((1 + e) / (1 - e)) - e)
        beta_0 = 1 / (e**2) - (1 - e**2) / (2 * pow(e, 3)) * math.log((1 + e) / (1 - e))

        k1 = alpha_0 / (2 - alpha_0)
        k2 = beta_0 / (2 - beta_0)
        k_prime = (
            pow(e, 4)
            * (beta_0 - alpha_0)
            / ((2 - e**2) * (2 * e**2 - (2 - e**2) * (beta_0 - alpha_0)))
        )

        # Added mass system matrix expressed in the CO
        self.MA = np.diag([m * k1, m * k2, m * k2, MA_44, k_prime * Iy, k_prime * Iy])

        # Mass matrix including added mass
        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # Natural frequencies in roll and pitch
        self.w_roll = math.sqrt(self.W * (self.r_bg[2] - self.r_bb[2]) / self.M[3][3])
        self.w_pitch = math.sqrt(self.W * (self.r_bg[2] - self.r_bb[2]) / self.M[4][4])

    def calculate_actuator_parameters(self, S_fin, CL_delta_r, CL_delta_s):
        a = self.L / 2  # distance from center of mass to fin actuation point
        portSternFin = fin(a=S_fin, CL=CL_delta_s, x=-a, c=0.1, angle=180, rho=self.rho)
        bottomRudderFin = fin(
            a=S_fin, CL=CL_delta_r, x=-a, c=0.1, angle=90, rho=self.rho
        )
        starSternFin = fin(a=S_fin, CL=CL_delta_s, x=-a, c=0.1, angle=0, rho=self.rho)
        topRudderFin = fin(a=S_fin, CL=CL_delta_r, x=-a, c=0.1, angle=270, rho=self.rho)
        prop = thruster(rho=self.rho)

        self.depth_subsystem = [starSternFin, portSternFin]
        self.heading_subsystem = [topRudderFin, bottomRudderFin]
        self.controls = [
            starSternFin,
            bottomRudderFin,
            portSternFin,
            topRudderFin,
            prop,
        ]
        self.dimU = len(self.controls)
        self.u_actual = np.zeros(self.dimU, float)  # control input vector

    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        """
        [nu_dot,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates
        the AUV equations of motion using Euler's method.
        """
        self.actuate(u_control, sampleTime)

        # Current velocities
        u_c = self.V_c * math.cos(self.beta_c - eta[5])  # current surge velocity
        v_c = self.V_c * math.sin(self.beta_c - eta[5])  # current sway velocity

        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float)  # current velocity
        Dnu_c = np.array([nu[5] * v_c, -nu[5] * u_c, 0, 0, 0, 0], float)  # derivative
        nu_r = nu - nu_c  # relative velocity
        alpha = math.atan2(nu_r[2], nu_r[0])  # angle of attack
        U_r = math.sqrt(nu_r[0] ** 2 + nu_r[1] ** 2 + nu_r[2] ** 2)  # relative speed

        # Rigi-body/added mass Coriolis/centripetal matrices expressed in the CO
        CRB = m2c(self.MRB, nu_r)
        CA = m2c(self.MA, nu_r)

        # CA-terms in roll, pitch and yaw can destabilize the model if quadratic
        # rotational damping is missing. These terms are assumed to be zero
        CA[4][0] = 0  # Quadratic velocity terms due to pitching
        CA[0][4] = 0
        CA[4][2] = 0
        CA[2][4] = 0
        CA[5][0] = 0  # Munk moment in yaw
        CA[0][5] = 0
        CA[5][1] = 0
        CA[1][5] = 0

        C = CRB + CA

        # Dissipative forces and moments
        D = np.diag(
            [
                self.M[0][0] / self.T_surge,
                self.M[1][1] / self.T_sway,
                self.M[2][2] / self.T_heave,
                self.M[3][3] * 2 * self.zeta_roll * self.w_roll,
                self.M[4][4] * 2 * self.zeta_pitch * self.w_pitch,
                self.M[5][5] / self.T_yaw,
            ]
        )

        # Linear surge and sway damping
        D[0][0] = D[0][0] * math.exp(-3 * U_r)  # vanish at high speed where quadratic
        D[1][1] = D[1][1] * math.exp(-3 * U_r)  # drag and lift forces dominates

        tau_liftdrag = forceLiftDrag(self.diam, self.S, self.CD_0, alpha, U_r, self.e)
        tau_crossflow = crossFlowDrag(self.L, self.diam, self.diam, nu_r)

        # Restoring forces and moments
        g = gvect(self.W, self.B, eta[4], eta[3], self.r_bg, self.r_bb)

        # General force vector
        tau = np.zeros(6, float)
        for actuator in self.controls:
            tau += actuator.tau(nu_r, nu)

        # AUV dynamics
        tau_sum = tau + tau_liftdrag + tau_crossflow - np.matmul(C + D, nu_r) - g
        nu_dot = Dnu_c + np.matmul(self.Minv, tau_sum)

        return nu_dot, None

    def actuate(self, u_control, sampleTime):
        # In the case of Autopilot u-control = u_actual so not change will happen
        if len(u_control) != len(self.controls):
            raise ValueError(
                f"Expected length of control commands for this vehicle is {len(self.controls)}, but got {len(u_control)}."
            )

        for i in range(len(self.controls)):
            self.controls[i].actuate(sampleTime, u_control[i])

    def stepInput(self, t):
        """
        u_c = stepInput(t) generates step inputs.

        Returns:

            u_control = [ delta_r   rudder angle (rad)
                         delta_s    stern plane angle (rad)
                         n          propeller revolution (rpm) ]
        """
        delta_r = 5 * self.D2R  # rudder angle (rad)
        delta_s = -5 * self.D2R  # stern angle (rad)
        n = 1525  # propeller revolution (rpm)

        if t > 100:
            delta_r = 0

        if t > 50:
            delta_s = 0

        u_control = np.array([delta_r, -delta_r, -delta_s, delta_s, n], float)

        return u_control

    def depthHeadingAutopilot(self, eta, nu_r, sampleTime):
        """
        [delta_r, delta_s, n] = depthHeadingAutopilot(eta,nu_r,sampleTime)
        simultaneously control the heading and depth of the AUV using control
        laws of PID type. Propeller rpm is given as a step command.

        Returns:

            u_control = [ delta_r   rudder angle (rad)
                         delta_s    stern plane angle (rad)
                         n          propeller revolution (rpm) ]

        """
        z = eta[2]  # heave position (depth)
        theta = eta[4]  # pitch angle
        psi = eta[5]  # yaw angle
        surge = nu_r[0]  # surge velocity
        w = nu_r[2]  # heave velocity
        q = nu_r[4]  # pitch rate
        r = nu_r[5]  # yaw rate
        e_psi = psi - self.psi_d  # yaw angle tracking error
        e_r = r - self.r_d  # yaw rate tracking error
        z_ref = self.ref_z  # heave position (depth) setpoint
        psi_ref = self.ref_psi * self.D2R  # yaw angle setpoint

        #######################################################################
        # Propeller command
        #######################################################################
        n = self.ref_n

        #######################################################################
        # Depth autopilot (succesive loop closure)
        #######################################################################
        if surge > self.surge_threshold:
            if not self.init_depth:
                self.z_d = z  # On initialization of the autopilot the commanded depth is set to the current depth
                self.init_depth = True
            # LP filtered desired depth command
            self.z_d = (
                math.exp(-sampleTime * self.wn_d_z) * self.z_d
                + (1 - math.exp(-sampleTime * self.wn_d_z)) * z_ref
            )

            if abs(z_ref - z) > self.outer_loop_threshold:
                # Set theta_d to theta max positive or negative
                theta_d = self.theta_max * np.sign(z - z_ref)
                # Use LP filter on theta when only running inner loop
                self.theta_d = (
                    math.exp(-sampleTime * self.wn_d_theta) * self.theta_d
                    + (1 - math.exp(-sampleTime * self.wn_d_theta)) * theta_d
                )

                # Override low pass filter value for depth for when the outer loop starts again
                saturated = (self.theta_max / -self.Kp_z) * np.sign(
                    z - z_ref
                )  # Value in outer loop that makes controller saturate
                self.z_d = z + saturated
            else:
                # PI controller Outer loop
                self.theta_d = -self.Kp_z * (
                    (self.z_d - z) + (1 / self.T_z) * self.z_int
                )
                self.z_int += sampleTime * (self.z_d - z)

                # If Saturated threshold is set properly this is not necessary
                if abs(self.theta_d) > self.theta_max:
                    self.theta_d = np.sign(self.theta_d) * self.theta_max

            # PID inner loop
            self.torque_s = (
                self.Kp_theta * ssa(self.theta_d - theta)
                - self.Kd_theta * q
                + self.Ki_theta * self.theta_int
                - self.K_w * w
            )

            # Euler's integration method (k+1)
            self.theta_int += sampleTime * ssa(self.theta_d - theta)

        else:
            # Wait until speed is high enough to start depth controller
            self.theta_d = theta  # LP filter starting point update

        #######################################################################
        # Heading autopilot (SMC controller)
        #######################################################################

        wn_d = self.wn_d  # reference model natural frequency
        zeta_d = self.zeta_d  # reference model relative damping factor

        if not self.init_heading:
            self.psi_d = psi  # On initialization of the autopilot the commanded heading is set to the current heading
            self.init_heading = True

        # Integral SMC with 3rd-order reference model
        [delta_r, self.e_psi_int, self.psi_d, self.r_d, self.a_d] = integralSMC(
            self.e_psi_int,
            e_psi,
            e_r,
            self.psi_d,
            self.r_d,
            self.a_d,
            self.T_nomoto,
            self.K_nomoto,
            wn_d,
            zeta_d,
            self.K_d,
            self.K_sigma,
            self.lam,
            self.phi_b,
            psi_ref,
            self.r_max,
            sampleTime,
        )

        self.torque_r = delta_r * 10
        for fin in self.heading_subsystem:
            torque = self.torque_r / len(self.heading_subsystem)
            command = fin.calculate_deflection([0.0, 0.0, torque], nu_r)
            fin.actuate(sampleTime, command)

        for fin in self.depth_subsystem:
            torque = self.torque_s / len(self.depth_subsystem)
            command = fin.calculate_deflection([0.0, torque, 0.0], nu_r)
            fin.actuate(sampleTime, command)

        self.controls[-1].actuate(sampleTime, n)

        u_control = []
        # NOTE: Returns Actual position instead of command. For the torpedo they should be similar most of the time.
        for control_surface in self.controls:
            u_control.append(control_surface.u_actual)

        return u_control


def configure_torpedo_from_scenario(scenario, vehicle_name, vehicle):
    """
    Dynamics Parameters:

    - Environment Parameters

    :param float rho:        Density of water in kg/m^3

    - Vehicle Physical Parameters

    :param float mass:      Mass of vehicle in kilograms
    :param float length:    Length of vehicle in meters
    :param float diam:      Diameter of vehicle in m
    :param float r_bg:      Center of gravity of the vehicle (x, y, z) in body frame x forward, y right, z down
    :param float r_bb:      Center of boyancy of the vehicle (x, y, z) in body frame x forward, y right, z down
    :param float area_fraction: relates vehicle effective area to length and width. pi/4 for a spheroid

    - Low-Speed Linear Damping Matrix Parameters:

    :param float T_surge:   Time constant in surge (s)
    :param float T_sway:    Time constant in sway (s)
    :param float T_yaw:     Time constant in yaw (s)
    :param float zeta_roll: Relative damping ratio in roll
    :param float zeta_ptich: Relative damping ratio in pitch
    :param float K_nomoto:

    - Other Damping Parameters

    :param float r44:       Added moment of inertia in roll: A44 = r44 * Ix
    :param float Cd:        Coefficient of drag
    :param float e:         Oswald efficiency factor for vehicle drag

    Autopilot Paramters:

    - Depth

    :param float wn_d_z:    Damped natural frequency for low pass filter for depth commands
    :param float Kp_z:      Portional gain for depth controller
    :param float T_z:       Time constant for depth controller
    :param float Kp_theta:  Porportional gain for pitch angle for depth controller
    :param float Ki_theta:  Integral gain for pitch angle for depth controller
    :param float Kd_theta:  Derivative gain for pitch angle for depth controller
    :param float wn_d_theta:    Damped natural frequency for low pass filter for pitch controller when only running inner loop
    :param float K_w:       Optional heave velocity feedback gain
    :param float theta_max_deg: Max output of pitch controller inner loop

    - Heading

    :param float wn_d:      Damped natural frequency of input commands for low pass filter
    :param zeta_d:          Damping coefficient
    :param r_max:           (?) Maximum yaw rate
    :param lam:
    :param phi_b:
    :param K_d:             (?) Derivative gain
    :param K_sigma:         (?) SMC switching gain

    Actuator parameters:

    - Fins

    :param fin_count:       Number of equally spaced fins on the vehicle
    :param fin_offset_deg:  Angle offset of first fin around x axis (deg) starting from positive y with z down
    :param fin_center:      radius (m) from COP on the fin to the COM in the YZ plane
    :param fin_area:        Surface area of one side of a fin
    :param x_fin:           Body frame x distance (x forward) from center of mass to fin COP
    :param CL_delta:        Coefficient of lift for fin
    :param deltaMax_fin_deg: Max deflection of the fin (degrees)
    :param fin_speed_max:   Max angular speed of the fin (rad/s)
    :param T_delta:         Time constant for fin actuation. (s)

    - Propellor

    :param nMax:            Max rpm of the thruster
    :param T_n:             Time constant for thruster actuation. (s)
    :param D_prop:          Propeller diameter
    :param t_prop:          Propeller pitch
    :param KT_0:            Thrust coefficient at zero rpm
    :param KQ_0:            Torque coefficient at zero rpm
    :param KT_max:          Max thrust coefficient
    :param KQ_max:          Max torque coefficient
    :param w:               Wake fraction number
    :param Ja_max:          Max advance ratio
    """

    ##### Initialize Default Vehicle parameters (REMUS100) #####
    vehicle.dynamics_parameters = {
        # Environment parameters:
        "rho": 1026,  # Density of water in kg/m^3
        # Vehicle physical parameters:
        "mass": 31.03,  # Mass of vehicle in kg
        "length": 1.6,  # Length of vehicle in m
        "diam": 0.19,  # Diameter of vehicle in m
        "r_bg": [
            0,
            0,
            0.02,
        ],  # Center of gravity of the vehicle (x, y, z) in body frame x forward, y right, z down
        "r_bb": [
            0,
            0,
            0,
        ],  # Center of boyancy of the vehicle (x, y, z) in body frame x forward, y right, z down
        "area_fraction": 0.7,  # relates vehicle effective area to length and width. pi/4 for a spheroid
        # Low-speed linear damping matrix parameters:
        "T_surge": 20,  # Surge time constant (s)
        "T_sway": 20,  # Sway time constant (s)
        "zeta_roll": 0.3,  # Roll damping ratio
        "zeta_pitch": 0.8,  # Pitch damping ratio
        "T_yaw": 1,  # Yaw time constant (s)
        "K_nomoto": 0.25,  # Nomoto gain
        # Other damping parameters:
        "r44": 0.3,  # Added moment of inertia in roll: A44 = r44 * Ix
        "Cd": 0.42,  # Coefficient of drag
        "e": 0.7,  # Oswald efficiency factor for vehicle drag
    }

    # NOTE: Could simplify the controller or enable config of parameters
    # For example enable the user to just set the rise time and damping ratio
    vehicle.autopilot_parameters = {
        "depth": {
            "wn_d_z": 0.12,  # Damped natural frequency for low pass filter for depth commands
            "Kp_z": 0.153,  # Proportional gain for depth controller
            "T_z": 100,  # Time constant for depth controller
            "Kp_theta": 39.78,  # Proportional gain for pitch angle for depth controller
            "Kd_theta": 17.1,  # Derivative gain for pitch angle for depth controller
            "Ki_theta": 0.5,  # Integral gain for pitch angle for depth controller
            "wn_d_theta": 0.25,  # Damped natural frequency for low pass filter for pitch controller  when only running inner loop
            "K_w": 0.0,  # Optional heave velocity feedback gain
            "theta_max_deg": 15,  # Max output of pitch controller inner loop
            "outer_loop_threshold": 2.91,  # Threshold (m) magnitude of the difference between commanded and current depth before starting outer loop
            "surge_threshold": 0.6,  # Minimum surge velocity (m/s) before starting depth autopilot
        },
        "heading": {
            "wn_d": 0.4,  # Damped natural frequency of input commands for low pass filter
            "zeta_d": 1.0,  # Damping coefficient
            "r_max": 0.87,  # (?) Maximum yaw rate
            "lam": 0.1,  #
            "phi_b": 0.1,  #
            "K_d": 0.5,  # (?) Derivative gain
            "K_sigma": 0.05,  # (?) SMC switching gain
        },
    }

    vehicle.actuator_parameters = {
        # Fins:
        "fin_count": 4,  # Number of equally spaced fins on the vehicle
        "fin_offset_deg": 0,  # Angle offset of first fin around x axis (deg) starting from positive y with z down
        # 0 deg: fin on port side
        # 90 deg: fin on bottom
        "fin_center": 0.1,  # radius (m) from COP on the fin to the COM in the YZ plane
        "fin_area": 0.00697,  # Surface area of one side of a fin
        "x_fin": -0.8,  # body frame X distance (x forward) from center of mass to fin COP
        "CL_delta": 0.5,  # Coefficient of lift for fin
        "deltaMax_fin_deg": 15,  # Max deflection of the fin (degrees)
        "fin_speed_max": 0.5,  # Max angular speed of the fin (rad/s)
        "T_delta": 0.1,  # Time constant for fin actuation. (s)
        # Propellor:
        "nMax": 1525,  # Max rpm of the thruster
        "T_n": 1.0,  # Time constant for thruster actuation. (s)
        "D_prop": 0.14,  # Propeller diameter
        "t_prop": 0.1,  # Propeller pitch
        "KT_0": 0.4566,  # Thrust coefficient at zero rpm
        "KQ_0": 0.0700,  # Torque coefficient at zero rpm
        "KT_max": 0.1798,  # Max thrust coefficient
        "KQ_max": 0.0312,  # Max torque coefficient
        "w": 0.056,  # wake fraction number
        "Ja_max": 0.6632,  # Max advance ratio
    }

    if scenario is not None:
        if vehicle_name is None:
            raise ValueError(
                "Vehicle name must be provided if a scenario is specified."
            )

        # Find the correct agent dictionary by agent_name
        agent_dict = None
        for agent in scenario.get("agents", []):
            if agent.get("agent_name") == vehicle_name:
                agent_dict = agent
                break

        if agent_dict is None:
            raise ValueError(
                f"No agent with name {vehicle_name} found in the scenario."
            )

        # Set vehicle parameters from the agent's 'dynamics' if it exists
        dynamics = agent_dict.get("dynamics")
        if dynamics is not None:
            set_torpedo_parameters(dynamics, vehicle)
        else:
            set_torpedo_parameters(vehicle.dynamics_parameters, vehicle)

        # Set autopilot parameters from the agent's 'autopilot' if it exists
        autopilot_parameters = agent_dict.get("autopilot")
        if autopilot_parameters is not None:
            set_torpedo_autopilot_parameters(autopilot_parameters, vehicle)
        else:
            set_torpedo_autopilot_parameters(vehicle.autopilot_parameters, vehicle)

        # Set actuator parameters from the agent's 'actuator' if it exists
        actuator = agent_dict.get("actuator")
        if actuator is not None:
            set_torpedo_actuator_parameters(actuator, vehicle)
        else:
            set_torpedo_actuator_parameters(vehicle.actuator_parameters, vehicle)


def set_torpedo_parameters(dynamics, vehicle):
    """
    Set vehicle dynamics parameters. If not provided, will default to previous value
    """
    vehicle.dynamics_parameters.update(dynamics)

    # Environment parameters:
    vehicle.rho = vehicle.dynamics_parameters.get("rho")
    g = 9.81  # Gravity constant in m/s^2
    # Vehicle physical parameters:
    m = vehicle.dynamics_parameters.get("mass")
    vehicle.r_bg = np.array(vehicle.dynamics_parameters.get("r_bg"))
    vehicle.r_bb = np.array(vehicle.dynamics_parameters.get("r_bb"))
    #### VEHICLE GEOMETRY ####
    vehicle.diam = vehicle.dynamics_parameters.get("diam")
    vehicle.L = vehicle.dynamics_parameters.get("length")
    # Low-speed linear damping matrix parameters:
    vehicle.T_surge = vehicle.dynamics_parameters.get("T_surge")
    vehicle.T_sway = vehicle.dynamics_parameters.get("T_sway")
    vehicle.T_heave = vehicle.T_sway  # equal for for a cylinder-shaped AUV
    vehicle.zeta_roll = vehicle.dynamics_parameters.get("zeta_roll")
    vehicle.zeta_pitch = vehicle.dynamics_parameters.get("zeta_pitch")
    vehicle.T_yaw = vehicle.dynamics_parameters.get("T_yaw")
    # Feed forward gains (Nomoto gain parameters)
    vehicle.T_nomoto = vehicle.T_yaw  # Time constant in yaw
    vehicle.K_nomoto = vehicle.dynamics_parameters.get("K_nomoto")
    # Other damping parameters:
    vehicle.e = vehicle.dynamics_parameters.get("e")
    Cd = vehicle.dynamics_parameters.get("Cd")
    vehicle.area_fraction = vehicle.dynamics_parameters.get("area_fraction")
    r44 = vehicle.dynamics_parameters.get("r44")

    # Update vehicle parameters
    vehicle.calculate_vehicle_parameters(g, Cd, r44, m=m)


def set_torpedo_autopilot_parameters(autopilot, vehicle):
    """
    Set depth and heading parameters from a configuration dictionary.

    :param cfg: Dictionary containing 'depth' and 'heading' sections with their respective parameters.
    """
    # Update depth parameters
    if "depth" in autopilot:
        vehicle.autopilot_parameters["depth"].update(autopilot["depth"])

    # Update heading parameters
    if "heading" in autopilot:
        vehicle.autopilot_parameters["heading"].update(autopilot["heading"])

    # Update heading parameters
    if "surge" in autopilot:
        vehicle.autopilot_parameters["surge"].update(autopilot["surge"])

    depth_cfg = vehicle.autopilot_parameters.get("depth", {})
    heading_cfg = vehicle.autopilot_parameters.get("heading", {})
    surge_cfg = vehicle.autopilot_parameters.get("surge", {})

    #### Surge Parameters
    vehicle.kp_surge = surge_cfg.get("kp_surge")
    vehicle.ki_surge = surge_cfg.get("ki_surge")
    vehicle.kd_surge = surge_cfg.get("kd_surge")

    #### Set depth parameters
    vehicle.wn_d_z = depth_cfg.get(
        "wn_d_z"
    )  # desired natural frequency, reference mode
    vehicle.Kp_z = depth_cfg.get("Kp_z")  # heave proportional gain, outer loop
    vehicle.T_z = depth_cfg.get("T_z")  # heave integral gain, outer loop

    vehicle.Kp_theta = depth_cfg.get("Kp_theta")  # pitch PID controller
    vehicle.Kd_theta = depth_cfg.get("Kd_theta")
    vehicle.Ki_theta = depth_cfg.get("Ki_theta")
    vehicle.K_w = depth_cfg.get("K_w")  # optional heave velocity feedback gain
    vehicle.wn_d_theta = depth_cfg.get(
        "wn_d_theta"
    )  # desired natural frequency for low pass filter for pitch controller when only running inner loop
    vehicle.theta_max = np.deg2rad(
        depth_cfg.get("theta_max_deg")
    )  # Max output of pitch controller inner loop in radians
    # NOTE: Could add logic to calucate the threshold if not provided
    # self.outer_loop_threshold = (self.theta_max/self.Kp_z) + 1.2
    vehicle.outer_loop_threshold = depth_cfg.get(
        "outer_loop_threshold"
    )  # threshold for outer loop to start again
    vehicle.surge_threshold = depth_cfg.get(
        "surge_threshold"
    )  # threshold for surge speed to start depth controller

    ##### heading parameters
    vehicle.wn_d = heading_cfg.get("wn_d")  # desired natural frequency
    vehicle.zeta_d = heading_cfg.get("zeta_d")  # desired realtive damping ratio
    vehicle.r_max = heading_cfg.get("r_max")  # maximum yaw rate
    vehicle.lam = heading_cfg.get("lam")
    vehicle.phi_b = heading_cfg.get("phi_b")  # boundary layer thickness
    vehicle.K_d = heading_cfg.get("K_d")  # PID gain
    vehicle.K_sigma = heading_cfg.get("K_sigma")  # SMC switching gain


def set_torpedo_actuator_parameters(actuator_parameters, vehicle):
    """
    Set fin area limits, time constants, and lift coefficients for control surfaces
    Create fin ojects attached to vehicle
    Fins will start on the starboard side then be placed equally distanced apart.
    Typical configurations will be: Fin Count - 4, Fin Offset - 0 Deg and Fin Count - 3, Fin Offset - 30 Deg
    """

    vehicle.actuator_parameters.update(actuator_parameters)

    # Fin parameters
    fin_count = vehicle.actuator_parameters.get("fin_count")
    offset_start = vehicle.actuator_parameters.get("fin_offset_deg")
    S_fin = vehicle.actuator_parameters.get("fin_area")
    T_delta = vehicle.actuator_parameters.get("T_delta")
    CL_delta = vehicle.actuator_parameters.get("CL_delta")
    # Radius Distance from center of mass to center of pressure on fin
    z_r = vehicle.actuator_parameters.get("fin_center")
    deltaMax = vehicle.actuator_parameters.get("deltaMax_fin_deg")
    # Propeller parameters
    T_n = vehicle.actuator_parameters.get("T_n")
    nMax = vehicle.actuator_parameters.get("nMax")
    D_prop = vehicle.actuator_parameters.get("D_prop")
    t_prop = vehicle.actuator_parameters.get("t_prop")
    KT_0 = vehicle.actuator_parameters.get("KT_0")
    KQ_0 = vehicle.actuator_parameters.get("KQ_0")
    KT_max = vehicle.actuator_parameters.get("KT_max")
    KQ_max = vehicle.actuator_parameters.get("KQ_max")
    w = vehicle.actuator_parameters.get("w")
    Ja_max = vehicle.actuator_parameters.get("Ja_max")

    # NOTE: Could add a logic similar to the vehicle where they are calucated with the same function as init
    a = (
        vehicle.L / 2
    )  # Calulate the distance from the center of mass to the actuation point

    vehicle.controls = []
    vehicle.heading_subsystem = []
    vehicle.depth_subsystem = []
    angles = 360 / fin_count
    # NOTE: Could add logic to print fin locations with the name of the agent.
    # print("Fin Locations:")
    for i in range(fin_count):
        angle_fin = (angles * i + offset_start) % 360
        vehicle.controls.append(
            fin(
                a=S_fin,
                CL=CL_delta,
                x=-a,
                c=z_r,
                angle=angle_fin,
                rho=vehicle.rho,
                T_delta=T_delta,
                deltaMax=deltaMax,
            )
        )
        if (angle_fin < 45 or angle_fin > 315) or (angle_fin > 135 and angle_fin < 225):
            vehicle.depth_subsystem.append(vehicle.controls[i])
        elif (angle_fin > 45 and angle_fin < 135) or (
            angle_fin > 225 and angle_fin < 315
        ):
            vehicle.heading_subsystem.append(vehicle.controls[i])
        # print("Fin", i,"@ angle", angle_fin, "(Deg)")

    vehicle.controls.append(
        thruster(
            T_n,
            nMax,
            vehicle.rho,
            D_prop,
            t_prop,
            KT_0,
            KQ_0,
            KT_max,
            KQ_max,
            w,
            Ja_max,
        )
    )

    # Reset the size of the control lists
    vehicle.dimU = len(vehicle.controls)
    vehicle.u_actual = np.zeros(vehicle.dimU, float)  # control input vector
