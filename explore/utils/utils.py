import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


# class BSpline:
#     degree = 2
#     knots = np.zeros(1)
#     ctrlPoints = np.zeros(1)
#     B, Bdot, Bddot = np.zeros(1), np.zeros(1), np.zeros(1)
#     JBtimes = np.zeros(1)
    
#     def __init__(self):
#         pass
        
#     def set(self,
#             degree: int,
#             points: np.ndarray,
#             times: np.ndarray,
#             startVel: np.ndarray=None,
#             endVel: np.ndarray=None):

#         assert len(times.shape) == 1
#         assert len(points.shape) == 2
#         assert points.shape[0] == times.shape[0]

#         self.setKnots(degree, times)
#         self.setCtrlPoints(points, True, True, startVel, endVel)
        
#     def setKnots(self, degree: int, times: np.ndarray):
#         self.degree = degree
#         nCtrls = int(times.shape[0] + 2 * (degree * 0.5))
#         nKnots = nCtrls + degree + 1
#         self.knots = np.zeros(nKnots)

#         for i in range(nKnots):
#             if i <= degree:
#                 self.knots[i] = times[0]
#             elif i >= nCtrls:
#                 self.knots[i] = times[-1]
#             elif degree % 2:
#                 self.knots[i] = times[i-degree]
#             else:
#                 self.knots[i] = 0.5 * ( times[i-degree-1] + times[i-degree] )
                
#     def setCtrlPoints(self,
#                       points: np.ndarray,
#                       addStartDuplicates: bool,
#                       addEndDuplicates: bool,
#                       setStartVel: np.ndarray,
#                       setEndVel: np.ndarray):
        
#         self.ctrlPoints = points
#         for i in range(self.degree/2):
#             if addStartDuplicates: self.ctrlPoints.prepend(points[0])
#             if addEndDuplicates: self.ctrlPoints.append(points[-1])

#         assert self.ctrlPoints.shape[0] == self.knots.shape[0] - self.degree - 1

#         if !!setStartVel and setStartVel.N: setDoubleKnotVel(-1, setStartVel)
#         if !!setEndVel and setEndVel.N: setDoubleKnotVel(points.d0-1, setEndVel)
    
#     def overwriteSmooth(self, points: np.ndarray, times_rel: np.ndarray, time_cut: float):
#         arr x_cut, xDot_cut;
#         eval3(x_cut, xDot_cut, NoArr, time_cut)

#         arr _points(points), _times(times_rel)
#         _times.prepend(0.)
#         _points.prepend(x_cut)
#         self.set(self.degree, _points, _times + time_cut, xDot_cut)
        
#     def eval3(self,
#               x: np.ndarray,
#               xDot: np.ndarray,
#               xDDot: np.ndarray,
#               t: float,
#               Jpoints: np.ndarray,
#               Jtimes: np.ndarray):
        
#         n = ctrlPoints.d1
#         if !!x: x.resize(n).setZero()
#         if !!xDot: xDot.resize(n).setZero()
#         if !!xDDot: xDDot.resize(n).setZero()
#         if !!Jpoints: Jpoints.resize(n, ctrlPoints.d0, n).setZero()
#         if !!Jtimes: Jtimes.resize(n, knots.N).setZero()
        
#         derivative = 0
#         if !!xDot: derivative=1
#         if !!xDDot: derivative=2

#         # Handle out-of-interval cases
#         if t < self.knots[0]:
#             if !!x: x = ctrlPoints[0];
#             if(!!Jpoints) for(uint i=0; i<n; i++) Jpoints(i, 0, i) = 1.;
#             return
        
#         if t>=knots(-1):
#             if(!!x) x = ctrlPoints[-1];
#             if(!!Jpoints) for(uint i=0; i<n; i++) Jpoints(i, -1, i) = 1.;
#             return

#         # Grab range of relevant knots
#         center: int = knots.rankInSorted(t, rai::lowerEqual<double>, true)
#         center -= 1
#         lo: int = center - degree
#         if lo < 0: lo = 0
#         up: int = lo + 2*degree
#         if up > knots.N-1: up = knots.N-1
#         lo: int = up - 2*degree
#         if lo < 0: lo = 0

#         # Compute B-spline coefficients on relevant knots
#         BSpline core;
#         core.degree = degree;
#         core.knots.referToRange(knots, {lo, up+1});
#         core.calcB(t, derivative, (!!Jtimes));

#         # Multiply coefficients with control points
#         arr Jtmp2;
#         b, bd, bdd = 0.0, 0.0, 0.0
#         for j in range(core.B.d0):
#             b = core.B(j, degree)
#             if derivative >= 1: bd = core.Bdot(j, degree)
#             if derivative >= 2: bdd = core.Bddot(j, degree)
#             if(lo+j>=ctrlPoints.d0):
#                 assert np.abs(b) <= 1e-4
#                 continue
#             if !!x: for i in range(n): x[i] += b * ctrlPoints(lo+j,i)
#             if !!xDot: for i in range(n): xDot[i] += bd * ctrlPoints(lo+j,i)
#             if !!xDDot: for i in range(n): xDDot[i] += bdd * ctrlPoints(lo+j,i)
#             if !!Jpoints:
#                 for i in range(n): Jpoints(i, lo+j, i) = b
#             if !!Jtimes:
#                 Jtmp2.resize(n, knots.N).setZero()
#                 Jtmp2.setMatrixBlock(ctrlPoints[lo+j] * ~core.JBtimes(j, degree, {}), 0, lo)
#                 Jtimes += Jtmp2


def randint_excluding(low: int, high: int, exclude: int):
    # Only for positive values
    if exclude >= 0:
        x = np.random.randint(low, high - 1)
        return x if x < exclude else x + 1
    else:
        x = np.random.randint(low, high)
        return x
        

def k_means(vectors: list[list[float]], k: int, cost_func, max_iter: int=100) -> list[int]:
    
    centroid_idxs = np.random.choice(len(vectors), k, replace=False)
    centroids = vectors[centroid_idxs]

    for _ in range(max_iter):
        # Assign points to the nearest centroid
        labels = []
        for vec in vectors:
            distances = [cost_func([vec], [c]) for c in centroids]
            labels.append(np.argmin(distances))
        labels = np.array(labels)
        
        # Update centroids
        new_centroids = np.array([
            vectors[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        
        # Convergence check
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    
    return labels


def sample_cluster_balanced(node_idx: list[int], labels: list[int]) -> tuple[int, int]:

    cluster_idx = random.randint(0, max(labels))

    idxs = [i for i, l in enumerate(labels) if cluster_idx == l]
    sampled_idx = random.choice(idxs)

    return node_idx[sampled_idx], cluster_idx


def extract_ball_from_img(img: np.ndarray, verbose: int=0
                          ) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_blue = np.array([90, 100, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)

        (x, y), radius = cv2.minEnclosingCircle(c)
        circle_data = np.array([
            1.,  # Ball visible indicator
            x / img.shape[0] * 2 - 1,
            y / img.shape[1] * 2 - 1,
            radius / img.shape[1] * 2 - 1
        ])
        mask = mask.astype(np.float32) / 255

        if verbose > 0:
            x, y, radius = int(x), int(y), int(radius)
            print(f"Ball position: x={x}, y={y}, radius={radius}")
            print(f"Ball data: {circle_data}")
            
            if verbose > 1:
                output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.circle(output, (x, y), radius, (0, 255, 0), 2)
                cv2.circle(output, (x, y), 2, (0, 0, 255), -1)

                cv2.imshow("Mask", mask)
                cv2.imshow("Detected Ball", output)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        return circle_data, mask
    
    if verbose:
        print("No blue ball detected.")
    
    return np.zeros(4), np.zeros_like(img)

def extract_balls_mask(img: np.ndarray, verbose: int = 0) -> np.ndarray:

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_blue = np.array([90, 100, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([20, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    masks_rgb = (
        np.stack([mask_orange, mask_orange, mask_orange], axis=-1) * np.array([1.0, 0.5, 0.0]) +
        np.stack([mask_blue, mask_blue, mask_blue], axis=-1) * np.array([0.0, 1.0, 1.0])
    ).astype(np.uint8)

    if verbose:
        fig, axes = plt.subplots(1, 2, figsize=(20, 20))
        axes[0].set_title("Original Image", fontsize=24, fontweight="bold")
        axes[0].imshow(img)
        axes[0].axis("off")
        axes[1].set_title("Masked Result", fontsize=24, fontweight="bold")
        axes[1].imshow(masks_rgb)
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    return masks_rgb

