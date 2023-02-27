# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import matplotlib.pyplot as plt
import polytope as pc
from networks.actor import Actor


# load the weights from the models folder
stateDim = (2,)
actionDim = 2
actor = Actor(stateDim, actionDim, 2, 2, 2)

actor.__call__(np.array([[0, 1]]))
# actor.load_weights('models/actor.h5')

A_border = np.array([[1.0, 0.0],
                    [-1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, -1.0]])
b_border = np.array([5.0, 5.0, 5.0, 5.0])
border_polytope = pc.Polytope(A_border, b_border)
box = pc.bounding_box(border_polytope)

regions = [border_polytope]
cnt = 1
for layer in [actor.l1, actor.l2, actor.l3, actor.lact]:
    cnt += 1
    new_polytopes = []
    weights = layer.weights[0]
    bias = layer.weights[1]

    for polytope in regions:
        cuts = []
        for neuron in range(weights.shape[0]):
            W = weights[:, neuron]
            b = bias[neuron]
            constraint = pc.Polytope(-np.array([W]), np.array([b]))
            cut = pc.Polytope(np.vstack((polytope.A, -W)),
                              np.append(polytope.b, b))
            if cut.volume > 0:
                cuts.append(cut)
        polytope_cut = [polytope]
        for cut in cuts:
            for sub_region in polytope_cut:
                # check if the cut is not already done
                if sub_region.diff(cut).volume > 0:
                    first_sub_polytope = pc.intersect(cut, sub_region)
                    if first_sub_polytope.volume > 0:
                        second_sub_polytope = sub_region.diff(
                            first_sub_polytope)
                        polytope_cut.append(first_sub_polytope)
                        polytope_cut.append(second_sub_polytope)
                        polytope_cut.remove(sub_region)
        for add_cut in polytope_cut:
            new_polytopes.append(add_cut)

    regions = new_polytopes
    print(
        f"there are {len(regions)} regions in the layer {cnt}")
    print("---------------")
fig, ax = plt.subplots()
for region in regions:
    region.plot(ax, linewidth=0.5, linestyle='--')
ax.set_xlim(box[0][0]-1, box[1][0] + 1)
ax.set_ylim(box[0][1]-1, box[1][1] + 1)
ax.set_aspect('equal')
plt.show()

"""
##########
p1_1 = pc.Polytope(-np.vstack((A_border,
                   W1[0, :])), np.append(b_border, b1[0, :]))
p1_2 = pc.Polytope(-np.vstack((A_border,
                   W1[1, :])), np.append(b_border, b1[1, :]))
p1_3 = pc.intersect(p1_1, p1_2)
p1_1 = p1_1.diff(p1_3)
p1_2 = p1_2.diff(p1_3)
##########
W1 = np.matmul(W2, W1)
b1 = np.matmul(W2, b1) + b2
##########
W1[:, 0] = 0  # put the first neuron to zero
print(W1)
print(b1)

p2_1 = pc.Polytope(-np.vstack((A_border,
                   W1[0, :])), np.append(b_border, b1[0, :]))
p2_2 = pc.Polytope(-np.vstack((A_border,
                   W1[1, :])), np.append(b_border, b1[1, :]))
p2_3 = pc.intersect(p2_1, p2_2)
p2_1 = p2_1.diff(p2_3)
p2_2 = p2_2.diff(p2_3)
##########
p2_1 = pc.intersect(p1_2, p2_1)
p2_2 = pc.intersect(p1_2, p2_2)
p2_3 = pc.intersect(p1_2, p2_3)

print(p2_1)

fig, ax = plt.subplots()
p.plot(ax, linewidth=0.5, linestyle='--')
p1_1.plot(ax, linewidth=0.5, linestyle='--')
p1_2.plot(ax, linewidth=0.5, linestyle='--')
p1_3.plot(ax, linewidth=0.5, linestyle='--')
p2_1.plot(ax, linewidth=0.5, linestyle='--')
p2_2.plot(ax, linewidth=0.5, linestyle='--')
# p2_3.plot(ax, linewidth=0.5, linestyle='--')
ax.set_xlim(box[0][0]-1, box[1][0] + 1)
ax.set_ylim(box[0][1]-1, box[1][1] + 1)
ax.set_aspect('equal')
plt.show()
"""
