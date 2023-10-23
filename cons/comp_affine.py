import taichi as ti


# Linear Blend Skinning
@ti.data_oriented
class CompDynAffine2D:

  def __init__(
      self,
      v_p: ti.MatrixField,  # vertex position
      v_p_ref: ti.MatrixField,  # vertex reference position
      v_p_rig: ti.MatrixField,  # rigged position of linear blend skinning
      v_invm: ti.Field,  # inverse mass
      v_weights: ti.Field,  # vertex weights
      dt,
      alpha=0.0) -> None:

    self.n_vert = v_p.shape[0]
    self.n_controls = v_weights.shape[1]
    self.v_p = v_p
    self.v_invm = v_invm
    self.v_p_rig = v_p_rig
    self.v_p_ref = v_p_ref
    self.weights = v_weights
    self.lambdaf = ti.field(dtype=ti.f32, shape=(self.n_controls, 6))
    self.alpha = alpha / (dt * dt)
    self.sum_deriv = ti.field(dtype=ti.f32, shape=(self.n_controls, 3))

  def init_rest_status(self):
    self.compute_deriv_sum()

  def preupdate_cons(self):
    self.lambdaf.fill(0.0)

  def update_cons(self):
    self.solve_cons()

  @ti.kernel
  def compute_deriv_sum(self):
    self.sum_deriv.fill(0.0)
    for i in range(self.n_vert):
      for j in range(self.n_controls):
        m = 1.0 / self.v_invm[i]
        w = self.weights[i, j]
        self.sum_deriv[j, 0] += m * w * w * self.v_p_ref[i][0] * self.v_p_ref[i][0]
        self.sum_deriv[j, 1] += m * w * w * self.v_p_ref[i][1] * self.v_p_ref[i][1]
        self.sum_deriv[j, 2] += m * w * w

  @ti.kernel
  def solve_cons(self):
    for j in range(self.n_controls):
      for kdx in range(3):
        c = ti.Vector([0.0, 0.0])
        delta_lambda = ti.Vector([0.0, 0.0])
        for i in range(self.n_vert):
          x_c = self.v_p[i] - self.v_p_rig[i]
          m = 1.0 / self.v_invm[i]
          w = self.weights[i, j]
          refx = self.v_p_ref[i]
          if kdx == 0:
            c[0] += m * w * refx[0] * x_c[0]
            c[1] += m * w * refx[0] * x_c[1]
          elif kdx == 1:
            c[0] += m * w * refx[1] * x_c[0]
            c[1] += m * w * refx[1] * x_c[1]
          else:
            c[0] += m * w * x_c[0]
            c[1] += m * w * x_c[1]
        
        for k in range(2):
          delta_lambda[k] = -(c[k] + self.alpha * self.lambdaf[j, kdx * 2 + k]) / (
              self.sum_deriv[j, kdx] + self.alpha)
          self.lambdaf[j, kdx * 2 + k] += delta_lambda[k]
        
        for i in range(self.n_vert):
          delta_x = ti.Vector([0.0, 0.0])
          refx = self.v_p_ref[i]
          w = self.weights[i, j]
          if kdx == 0:
            delta_x += w * delta_lambda * refx[0]
          elif kdx == 1:
            delta_x += w * delta_lambda * refx[1]
          else:
            delta_x += w * delta_lambda
          self.v_p[i] += delta_x
