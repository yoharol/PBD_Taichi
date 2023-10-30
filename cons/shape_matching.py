import taichi as ti


@ti.data_oriented
class ShapeMatching2D:

  def __init__(
      self,
      v_p: ti.MatrixField,  # vertex position
      v_p_ref: ti.MatrixField,  # vertex reference position
      v_p_rig: ti.MatrixField,  # rigged position
      v_invm: ti.Field,  # inverse mass
      v_weights: ti.Field,  # vertex weights
      dt: float,
      alpha=0.0) -> None:

    self.n_vert = v_p.shape[0]
    self.v_p = v_p
    self.v_invm = v_invm
    self.v_p_rig = v_p_rig
    self.v_p_ref = v_p_ref
    self.n_controls = v_weights.shape[1]
    self.weights = v_weights
    self.alpha = alpha / (dt * dt)
    print(self.alpha, 1.0 / (1.0 + self.alpha))

  def init_rest_status(self):
    pass

  def preupdate_cons(self):
    pass

  def update_cons(self):
    self.solve_cons()

  @ti.kernel
  def solve_cons(self):
    for i in range(self.n_vert):
      self.v_p[i] += -(self.v_p[i] - self.v_p_rig[i]) / (1.0 + self.alpha)

  @ti.kernel
  def update_selected_cons(self, j: ti.i32):
    for i in range(self.n_vert):
      x_c = self.v_p[i] - self.v_p_rig[i]
      self.v_p[i] += -x_c / (1.0 + self.alpha / self.weights[i, j])
