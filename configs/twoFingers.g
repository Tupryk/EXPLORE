l_base: { pose: [0, 0, 0.5], multibody: true, multibody_gravity: false }
l_fake(l_base): {  }
l_joint1(l_fake): { joint: transX, limits: [-0.5, 0.5], mass: 0.01, inertia: [0, 0, 0], sampleUniform: 1 }
l_joint2(l_joint1): { joint: transY, limits: [-0.5, 0.5], mass: 0.01, inertia: [0, 0, 0], sampleUniform: 1 }
l_fing(l_joint2): { joint: transZ, limits: [-0.5, 0.5], shape: sphere, size: [0.05], color: [0.5, 1, 1], contact: 1, mass: 0.1, inertia: [0.00025, 0.00025, 0.00025], sampleUniform: 1 }

r_base: { pose: [0, 0, 0.5], multibody: true, multibody_gravity: false }
r_fake(r_base): {  }
r_joint1(r_fake): { joint: transX, limits: [-0.5, 0.5], mass: 0.01, inertia: [0, 0, 0], sampleUniform: 1 }
r_joint2(r_joint1): { joint: transY, limits: [-0.5, 0.5], mass: 0.01, inertia: [0, 0, 0], sampleUniform: 1 }
r_fing(r_joint2): { joint: transZ, limits: [-0.5, 0.5], shape: sphere, size: [0.05], color: [0.5, 1, 1], contact: 1, mass: 0.1, inertia: [0.00025, 0.00025, 0.00025], sampleUniform: 1 }

world: {  }
table(world): { pose: [0, 0, 0.1], shape: box, size: [1, 1, 0.1], color: [0.6], contact: 1 }
obj: { pose: [0, 0, 0.3], limits: [-0.5, -0.5, 0, 0.5, 0.5, 1], shape: sphere, size: [0.1], color: [1, 0.5, 0], contact: 1, mass: 0.2, inertia: [0.002, 0.002, 0.002], sampleUniform: 1 }
wall(table): { pose: [0.45, 0, 0.3], shape: box, size: [0.1, 1, 0.5], color: [0.6], contact: 1 }
box(table): { pose: [-0.05, -0.35, 0.25], shape: box, size: [0.9, 0.3, 0.4], color: [0.6], contact: 1 }
