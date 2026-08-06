import numpy as np

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


# ============================================
# Gyro integration → quaternion increment
# ============================================
def quat_from_gyro(omega, dt):
    theta = np.linalg.norm(omega) * dt
    if theta < 1e-8:
        return np.array([1, 0, 0, 0])

    axis = omega / np.linalg.norm(omega)

    dq = np.zeros(4)
    dq[0] = np.cos(theta / 2)
    dq[1:] = axis * np.sin(theta / 2)

    return dq


# ============================================
# Wahba solver via SVD
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


# ============================================
# Wahba via Davenport q-method
# ============================================
def wahba_davenport(v_body, v_ref, weights):
    B = np.zeros((3, 3))

    for vb, vr, w in zip(v_body, v_ref, weights):
        B += w * np.outer(vb, vr)

    S = B + B.T
    sigma = np.trace(B)

    Z = np.array([
        B[1, 2] - B[2, 1],
        B[2, 0] - B[0, 2],
        B[0, 1] - B[1, 0]
    ])

    K = np.zeros((4, 4))
    K[0, 0] = sigma
    K[0, 1:] = Z
    K[1:, 0] = Z
    K[1:, 1:] = S - sigma * np.eye(3)

    eigvals, eigvecs = np.linalg.eig(K)
    q = eigvecs[:, np.argmax(eigvals)]

    return quat_normalize(q.real)


# ============================================
# Rotation matrix → quaternion
# ============================================
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
# Dual IMU Fusion Filter
# ============================================
class EigenFusionIMU:
    def __init__(self, kp=2.0):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.kp = kp
        self.g_ref = np.array([0.0, 0.0, 1.0])

    def update(self, gyro_fast, acc1, acc2, dt):

        # 1. Gyro propagation
        dq = quat_from_gyro(gyro_fast, dt)
        q_pred = quat_normalize(quat_multiply(self.q, dq))

        # 2. Build Wahba vectors
        v_body, v_ref, weights = [], [], []

        if np.linalg.norm(acc1) > 1e-6:
            v_body.append(acc1 / np.linalg.norm(acc1))
            v_ref.append(self.g_ref)
            weights.append(1.0)

        if np.linalg.norm(acc2) > 1e-6:
            v_body.append(acc2 / np.linalg.norm(acc2))
            v_ref.append(self.g_ref)
            weights.append(1.0)

        if len(v_body) == 0:
            self.q = q_pred
            return self.q

        # 3. Compute BOTH solutions
        q_svd = wahba_svd(v_body, v_ref, weights)
        q_dav = wahba_davenport(v_body, v_ref, weights)

        # Align signs
        if np.dot(q_svd, q_dav) < 0:
            q_dav = -q_dav

        diff = np.linalg.norm(q_svd - q_dav)

        # Optional print
        print("SVD:", q_svd)
        print("DAV:", q_dav)
        print("Diff:", diff)
        print("------------")

        # Use SVD for correction (stable)
        q_meas = q_svd

        # 4. Error
        q_err = quat_multiply(quat_conjugate(q_pred), q_meas)
        e = q_err[1:]

        # 5. Correction
        corr = np.concatenate(([1.0], self.kp * e))
        corr = quat_normalize(corr)

        self.q = quat_normalize(quat_multiply(q_pred, corr))

        return self.q


# ============================================
# Example usage
# ============================================
if __name__ == "__main__":
    filt = EigenFusionIMU(kp=1.2)
    dt = 0.01

    for i in range(300):

        gyro = np.array([0.02, 0.01, 0.015])

        acc1 = np.array([0, 0, 1]) + 0.01 * np.random.randn(3)
        acc2 = np.array([0, 0, 1]) + 0.02 * np.random.randn(3)

        q = filt.update(gyro, acc1, acc2, dt)

        if i % 50 == 0:
            print("Fused q:", q)
            print("====================================")