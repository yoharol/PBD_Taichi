import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)


@ti.func
def get_col0_2d(a):
  return ti.Vector([a[0, 0], a[1, 0]])


@ti.func
def get_col1_2d(a):
  return ti.Vector([a[0, 1], a[1, 1]])


@ti.kernel
def test():

  D = ti.Matrix([[169.59160860288873, 175.2521857548754],
                 [-84.49405345386657, -17.962513991576415]])
  B = ti.Matrix([[0.0, -0.006666666666666667],
                 [0.006666666666666667, 0.006666666666666667]])

  F = D @ B

  C_H = F.determinant() - 1
  par_det_F = ti.Matrix([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]])
  print('par_det_F', par_det_F)
  print('test', par_det_F @ (B.transpose()))
  CH_H = par_det_F @ (B.transpose())
  par_CH_x1 = get_col0_2d(CH_H)
  par_CH_x2 = get_col1_2d(CH_H)
  par_CH_x3 = -par_CH_x1 - par_CH_x2
  # sum_par_CH = w1 * par_CH_x1.norm_sqr() + w2 * par_CH_x2.norm_sqr(
  # ) + w3 * par_CH_x3.norm_sqr()
  # alpha_tilde_H = self.hydro_alpha / (self.dt * self.dt * self.face_mass[k])

  C_D = F.norm_sqr() - 2.0
  CD_H = 2.0 * F @ (B.transpose())
  par_CD_x1 = get_col0_2d(CD_H)
  par_CD_x2 = get_col1_2d(CD_H)
  par_CD_x3 = -par_CD_x1 - par_CD_x2
  # sum_par_CD = w1 * par_CD_x1.norm_sqr() + w2 * par_CD_x2.norm_sqr(
  # ) + w3 * par_CD_x3.norm_sqr()
  # alpha_tilde_D = self.devia_alpha / (self.dt * self.dt * self.face_mass[k])

  print('D', D)
  print('B', B)
  print('F', F)
  print('F@B', F @ B)
  print('F@B^T', F @ B.transpose())
  print('C_H', C_H)
  print('CH_H', CH_H)
  print('par_CH_x1', par_CH_x1)
  print(par_CH_x2)
  print(par_CH_x3)
  print(C_D)
  print(par_CD_x1)
  print(par_CD_x2)
  print(par_CD_x3)
  """sum_par_CDH = w1 * par_CD_x1.dot(par_CH_x1) + w2 * par_CD_x2.dot(
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
                                                      sum_par_CD)"""


test()