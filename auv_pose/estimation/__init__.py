"""State estimation: inertial navigation on the state manifold, and its smoother.

Conventions: the world is z up and gravity is ``[0, 0, -9.81]``; the body frame
is HoloOcean's ``IMUSocket``, z down. Quaternions are scalar-first
``[w, x, y, z]`` and rotate body vectors into the world.
"""
