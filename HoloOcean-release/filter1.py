import numpy as np
import pandas as pd

# ============================================
# Quaternion utilities
# ============================================
def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_multiply(q, r):
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_from_gyro(omega, dt):
    theta = np.linalg.norm(omega) * dt
    if theta < 1e-8:
        return np.array([1, 0, 0, 0])

    axis = omega / np.linalg.norm(omega)

    dq = np.zeros(4)
    dq[0] = np.cos(theta / 2)
    dq[1:] = axis * np.sin(theta / 2)

    return dq

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])

# ============================================
# Wahba solvers
# ============================================
def wahba_svd(v_body, v_ref, weights):
    B = np.zeros((3, 3))
    for vb, vr, w in zip(v_body, v_ref, weights):
        B += w * np.outer(vb, vr)

    U, _, Vt = np.linalg.svd(B)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    return rotmat_to_quat(R)

def wahba_davenport(v_body, v_ref, weights):
    B = np.zeros((3, 3))
    for vb, vr, w in zip(v_body, v_ref, weights):
        B += w * np.outer(vb, vr)

    S = B + B.T
    sigma = np.trace(B)

    Z = np.array([
        B[1,2] - B[2,1],
        B[2,0] - B[0,2],
        B[0,1] - B[1,0]
    ])

    K = np.zeros((4,4))
    K[0,0] = sigma
    K[0,1:] = Z
    K[1:,0] = Z
    K[1:,1:] = S - sigma * np.eye(3)

    eigvals, eigvecs = np.linalg.eig(K)
    q = eigvecs[:, np.argmax(eigvals)]

    return quat_normalize(q.real)

def rotmat_to_quat(R):
    q = np.zeros(4)
    t = np.trace(R)

    if t > 0:
        S = np.sqrt(t + 1.0) * 2
        q[0] = 0.25 * S
        q[1] = (R[2,1] - R[1,2]) / S
        q[2] = (R[0,2] - R[2,0]) / S
        q[3] = (R[1,0] - R[0,1]) / S
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            S = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2]) * 2
            q[0] = (R[2,1] - R[1,2]) / S
            q[1] = 0.25 * S
            q[2] = (R[0,1] + R[1,0]) / S
            q[3] = (R[0,2] + R[2,0]) / S
        elif i == 1:
            S = np.sqrt(1 + R[1,1] - R[0,0] - R[2,2]) * 2
            q[0] = (R[0,2] - R[2,0]) / S
            q[1] = (R[0,1] + R[1,0]) / S
            q[2] = 0.25 * S
            q[3] = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1 + R[2,2] - R[0,0] - R[1,1]) * 2
            q[0] = (R[1,0] - R[0,1]) / S
            q[1] = (R[0,2] + R[2,0]) / S
            q[2] = (R[1,2] + R[2,1]) / S
            q[3] = 0.25 * S

    return quat_normalize(q)

# ============================================
# Adaptive Weight Functions
# ============================================
def accel_weight(acc):
    g = np.linalg.norm(acc)
    error = abs(g - 1.0)
    return np.exp(-5 * error)

def dvl_weight(dvl):
    speed = np.linalg.norm(dvl)
    if speed < 0.2:
        return 0.0
    return min(1.0, speed)

# ============================================
# Enhanced Fusion Filter
# ============================================
class EigenFusionIMU:
    def __init__(self, kp=2.0, ki=0.05):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.b = np.zeros(3)        #  gyro bias
        self.kp = kp
        self.ki = ki
        self.g_ref = np.array([0,0,-1])

    def update(self, gyro, acc_list, dt, dvl=None):

        # ==========================
        # 1. Bias corrected gyro
        # ==========================
        omega = gyro - self.b

        dq = quat_from_gyro(omega, dt)
        q_pred = quat_normalize(quat_multiply(self.q, dq))

        # ==========================
        # 2. Build Wahba inputs
        # ==========================
        v_body, v_ref, weights = [], [], []

        for acc in acc_list:
            if np.linalg.norm(acc) > 1e-6:
                vb = acc / np.linalg.norm(acc)
                w = accel_weight(acc)

                v_body.append(vb)
                v_ref.append(self.g_ref)
                weights.append(w)

        #  DVL addition
        if dvl is not None and np.linalg.norm(dvl) > 1e-3:
            vb = dvl / np.linalg.norm(dvl)

            R = quat_to_rotmat(q_pred)
            vw = R @ vb
            vw /= np.linalg.norm(vw)

            v_body.append(vb)
            v_ref.append(vw)
            weights.append(dvl_weight(dvl))

        if len(v_body) == 0:
            self.q = q_pred
            return self.q

        # Normalize weights
        weights = np.array(weights)
        weights /= (np.sum(weights) + 1e-6)

        # ==========================
        # 3. Solve Wahba
        # ==========================
        q_svd = wahba_svd(v_body, v_ref, weights)
        q_dav = wahba_davenport(v_body, v_ref, weights)

        if np.dot(q_svd, q_dav) < 0:
            q_dav = -q_dav

        diff = np.linalg.norm(q_svd - q_dav)

        # ==========================
        # 4. Error + Bias update 
        # ==========================
        q_err = quat_multiply(quat_conjugate(q_pred), q_svd)
        e = q_err[1:]

        self.b += self.ki * e * dt   #  bias estimation

        # ==========================
        # 5. Correction
        # ==========================
        corr = np.concatenate(([1.0], self.kp * e))
        corr = quat_normalize(corr)

        self.q = quat_normalize(quat_multiply(q_pred, corr))

        # Debug output (light)
        print("Diff(SVD-Davenport):", diff)

        return self.q

'''
# ============================================
# Example usage
# ============================================
if __name__ == "__main__":
    filt = EigenFusionIMU(kp=1.5, ki=0.05)
    dt = 0.01

    for i in range(300):

        gyro = np.array([0.02, 0.01, 0.015])

        acc1 = np.array([0, 0, 1]) + 0.01*np.random.randn(3)
        acc2 = np.array([0, 0, 1]) + 0.02*np.random.randn(3)
        acc3 = np.array([0, 0, 1]) + 0.03*np.random.randn(3)

        # Simulated DVL velocity
        dvl = np.array([0.5, 0.2, 0.0]) + 0.05*np.random.randn(3)

        q = filt.update(gyro, [acc1, acc2, acc3], dt, dvl)

        if i % 50 == 0:
            print("Orientation:", q)
            print("Estimated Bias:", filt.b)
            print("===================================")
'''


if __name__ == "__main__":
    filt = EigenFusionIMU(kp=1.5, ki=0.05)

    df = pd.read_csv("imu_log3.csv")

    prev_time = None

    for _, row in df.iterrows():

        #  Time step
        t = row['step']
        if prev_time is None:
            prev_time = t
            continue
        dt = t - prev_time
        prev_time = t

        #  Two gyros
        gyro1 = np.array([row['imu1_gx'], row['imu1_gy'], row['imu1_gz']])
        gyro2 = np.array([row['imu2_gx'], row['imu2_gy'], row['imu2_gz']])

        #  Combine gyros
        gyro =  ((gyro1*0.3) + (gyro2*0.2))/0.5

        #  Two accelerometers
        acc1 = np.array([row['imu1_ax'], row['imu1_ay'], row['imu1_az']])
        acc2 = np.array([row['imu2_ax'], row['imu2_ay'], row['imu2_az']])

        #  Run filter (no DVL)
        q = filt.update(gyro, [acc1, acc2], dt)

        print("Orientation:", q)
        print("Bias:", filt.b)
        print("----------------------")
