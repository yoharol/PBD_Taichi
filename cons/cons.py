# template of constraint class
import taichi as ti


class Constraint:

  def __init__(self) -> None:
    pass

  def init_rest_status(self):
    pass

  def preupdate_cons(self):
    pass

  def update_cons(self):
    self.solve_cons()

  @ti.kernel
  def solve_cons(self):
    pass