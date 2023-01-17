import taichi as ti

from utils.mathlib import *


@ti.data_oriented
class Deform2D:

  def __init__(self,
               n: int,
               indices,
               invm,
               pos,
               pos_ref,
               face_mass,
               hydro_alpha=0.0,
               devia_alpha=0.0) -> None:
    self.n = n
    self.hydro_lambda = ti.field(dtype=ti.f32, shape=self.n)
    self.devia_lambda = ti.field(dtype=ti.f32, shape=self.n)
    self.hydro_alpha = hydro_alpha
    self.devia_alpha = devia_alpha
    self.indices = indices
    self.invm = invm
    self.face_mass = face_mass
    self.pos = pos
    self.pos_ref = pos_ref

  def init_rest_status(self):
    pass

  def preupdate_cons(self):
    self.devia_lambda.fill(0.0)
    self.hydro_lambda.fill(0.0)

  def update_cons(self, dt):
    self.solve_cons(dt)

  @ti.kernel
  def solve_cons(self, dt: ti.f32):
    for k in range(self.n):
      a = self.indices[k * 3]
      b = self.indices[k * 3 + 1]
      c = self.indices[k * 3 + 2]
      x_1 = self.pos[a]
      x_2 = self.pos[b]
      x_3 = self.pos[c]
      r_1 = self.pos_ref[a]
      r_2 = self.pos_ref[b]
      r_3 = self.pos_ref[c]
      w1 = self.invm[a]
      w2 = self.invm[b]
      w3 = self.invm[c]
      D = ti.Matrix.cols([x_1 - x_3, x_2 - x_3])
      B = ti.Matrix.cols([r_1 - r_3, r_2 - r_3]).inverse()
      F = D @ B

      C_H = F.determinant() - 1
      par_det_F = ti.Matrix([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]])
      CH_H = par_det_F @ (B.transpose())
      par_CH_x1 = get_col0_2d(CH_H)
      par_CH_x2 = get_col1_2d(CH_H)
      par_CH_x3 = -par_CH_x1 - par_CH_x2
      sum_par_CH = w1 * par_CH_x1.norm_sqr() + w2 * par_CH_x2.norm_sqr(
      ) + w3 * par_CH_x3.norm_sqr()
      alpha_tilde_H = self.hydro_alpha / (dt * dt * self.face_mass[k])

      C_D = F.norm_sqr() - 2.0
      CD_H = 2.0 * F @ (B.transpose())
      par_CD_x1 = get_col0_2d(CD_H)
      par_CD_x2 = get_col1_2d(CD_H)
      par_CD_x3 = -par_CD_x1 - par_CD_x2
      sum_par_CD = w1 * par_CD_x1.norm_sqr() + w2 * par_CD_x2.norm_sqr(
      ) + w3 * par_CD_x3.norm_sqr()
      alpha_tilde_D = self.devia_alpha / (dt * dt * self.face_mass[k])

      sum_par_CDH = w1 * par_CD_x1.dot(par_CH_x1) + w2 * par_CD_x2.dot(
          par_CH_x2) + w3 * par_CD_x3.dot(par_CH_x3)
      delta_lambda_H = (sum_par_CDH *
                        (C_D + alpha_tilde_D * self.devia_lambda[k]) -
                        (C_H + alpha_tilde_H * self.hydro_lambda[k]) *
                        (alpha_tilde_D + sum_par_CD)) / (
                            (alpha_tilde_H + sum_par_CH) *
                            (alpha_tilde_D + sum_par_CD) -
                            sum_par_CDH * sum_par_CDH)
      delta_lambda_D = -(C_D + alpha_tilde_D * self.devia_lambda[k] +
                         sum_par_CDH * delta_lambda_H) / (alpha_tilde_D +
                                                          sum_par_CD)

      self.pos[a] += w1 * par_CH_x1 * delta_lambda_H
      self.pos[b] += w2 * par_CH_x2 * delta_lambda_H
      self.pos[c] += w3 * par_CH_x3 * delta_lambda_H
      self.hydro_lambda[k] += delta_lambda_H

      self.pos[a] += w1 * par_CD_x1 * delta_lambda_D
      self.pos[b] += w2 * par_CD_x2 * delta_lambda_D
      self.pos[c] += w3 * par_CD_x3 * delta_lambda_D
      self.devia_lambda[k] += delta_lambda_D
