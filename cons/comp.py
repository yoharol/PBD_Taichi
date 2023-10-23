import taichi as ti


# Linear Blend Skinning
@ti.data_oriented
class CompDynPoint2D:

  def __init__(
      self,
      v_p: ti.MatrixField,  # vertex position
      v_p_ref: ti.MatrixField,  # vertex reference position
      v_p_rig: ti.MatrixField,  # rigged position of linear blend skinning
      v_invm: ti.Field,  # inverse mass
      c_p: ti.MatrixField,  # control point position
      c_p_ref: ti.MatrixField,  # control point reference position
      c_rot: ti.MatrixField,  # control point rotation
      v_weights: ti.Field,  # vertex weights
      dt,
      alpha=0.0) -> None:

    self.n_vert = v_p.shape[0]
    self.v_p = v_p
    self.v_invm = v_invm
    self.v_p_rig = v_p_rig
    self.v_p_ref = v_p_ref
    self.n_controls = c_p.shape[0]
    self.c_p = c_p
    self.c_p_ref = c_p_ref
    self.c_rot = c_rot
    self.weights = v_weights
    self.lambdaf = ti.field(dtype=ti.f32, shape=(self.n_controls, 3))
    self.C = ti.field(dtype=ti.f32, shape=(self.n_controls, 3))
    self.delta_lambda = ti.field(dtype=ti.f32, shape=(self.n_controls, 3))
    self.alpha = alpha / (dt * dt)
    self.sum_deriv = ti.field(dtype=ti.f32, shape=(self.n_controls, 3))
    self.sum_deriv_cache = ti.field(dtype=ti.f32, shape=(self.n_controls))

  def init_rest_status(self):
    self.compute_deriv_sum()

  def preupdate_cons(self):
    self.lambdaf.fill(0.0)

  def update_cons(self):
    self.solve_cons()

  @ti.kernel
  def compute_deriv_sum(self):
    self.sum_deriv_cache.fill(0.0)
    for i in range(self.n_vert):
      for j in range(self.n_controls):
        m = 1.0 / self.v_invm[i]
        w = self.weights[i, j]
        self.sum_deriv_cache[j] += m * w * w

  @ti.kernel
  def solve_cons(self):
    self.C.fill(0.0)
    self.sum_deriv.fill(0.0)

    ti.loop_config(serialize=True)
    for i in range(self.n_vert):
      x_c = self.v_p[i] - self.v_p_rig[i]
      for j in range(self.n_controls):
        m = 1.0 / self.v_invm[i]
        w = self.weights[i, j]
        x = self.c_rot[j] @ (self.v_p_ref[i] - self.c_p_ref[j])
        self.C[j, 0] += m * w * x_c[0]
        self.C[j, 1] += m * w * x_c[1]
        self.C[j, 2] += m * w * x_c.dot(ti.Vector([-x[1], x[0]]))
        x = x * x
        self.sum_deriv[j, 2] += m * w * w * (x[1] + x[0])

    ti.loop_config(serialize=True)
    for i in range(self.n_controls):
      self.sum_deriv[i, 0] = self.sum_deriv_cache[i]
      self.sum_deriv[i, 1] = self.sum_deriv_cache[i]
      for j in range(3):
        self.delta_lambda[
            i,
            j] = -(self.C[i, j] + self.alpha * self.lambdaf[i, j]) / (
                self.sum_deriv[i, j] + self.alpha)
        self.lambdaf[i, j] += self.delta_lambda[i, j]

    for i in range(self.n_vert):
      delta_x = ti.Vector([0.0, 0.0])
      for j in range(self.n_controls):
        w = self.weights[i, j]
        delta_x += self.delta_lambda[j, 0] * w * ti.Vector([1.0, 0.0])
        delta_x += self.delta_lambda[j, 1] * w * ti.Vector([0.0, 1.0])
        x = self.c_rot[j] @ (self.v_p_ref[i] - self.c_p_ref[j])
        delta_x += self.delta_lambda[j, 2] * w * ti.Vector([-x[1], x[0]])
      self.v_p[i] += delta_x
