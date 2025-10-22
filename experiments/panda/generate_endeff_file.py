from solid.utils import sphere
from solid import scad_render_to_file

r = 0.02

ball = sphere(r)

scad_render_to_file(ball, "experiments/panda/ball.scad", file_header='$fn = 100;')
