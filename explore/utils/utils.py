import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


class BSpline:
    degree = 2
    knots = np.zeros(1)
    ctrlPoints = np.zeros(1)
    B, Bdot, Bddot = np.zeros(1), np.zeros(1), np.zeros(1)
    JBtimes = np.zeros(1)
    
    def __init__(self):
        pass
        
    def set(self,
            degree: int,
            points: np.ndarray,
            times: np.ndarray,
            startVel: np.ndarray=None,
            endVel: np.ndarray=None):

        assert len(times.shape) == 1
        assert len(points.shape) == 2
        assert points.shape[0] == times.shape[0]

        self.setKnots(degree, times)
        self.setCtrlPoints(points, True, True, startVel, endVel)
        
    def setKnots(self, degree: int, times: np.ndarray):
        self.degree = degree
        nCtrls = int(times.shape[0] + 2 * (degree * 0.5))
        nKnots = nCtrls + degree + 1
        self.knots = np.zeros(nKnots)

        for i in range(nKnots):
            if i <= degree:
                self.knots[i] = times[0]
            elif i >= nCtrls:
                self.knots[i] = times[-1]
            elif degree % 2:
                self.knots[i] = times[i-degree]
            else:
                self.knots[i] = 0.5 * ( times[i-degree-1] + times[i-degree] )
                
    def setCtrlPoints(self,
                      points: np.ndarray,
                      addStartDuplicates: bool=True,
                      addEndDuplicates: bool=True,
                      setStartVel: np.ndarray=None,
                      setEndVel: np.ndarray=None):
        
        self.ctrlPoints = points
        for _ in range(int(np.ceil(self.degree * 0.5))):
            if addStartDuplicates:
                self.ctrlPoints = np.concatenate(([points[0]], self.ctrlPoints))

            if addEndDuplicates:
                self.ctrlPoints = np.concatenate((self.ctrlPoints, [points[-1]]))

        assert self.ctrlPoints.shape[0] == self.knots.shape[0] - self.degree - 1

        if setStartVel is not None:
            self.setDoubleKnotVel(-1, setStartVel)
        if setEndVel is not None:
            self.setDoubleKnotVel(points.shape[0]-1, setEndVel)
    
    def setDoubleKnotVel(self, t: int, vel: np.ndarray):
        a = self.ctrlPoints[int(t + self.degree / 2)]
        b = self.ctrlPoints[int(t + self.degree / 2 + 1)]
        assert np.abs(a - b).max() < 1e-10, "this is not a double knot!"
        if self.degree == 2:
            a -= vel / self.degree * (self.knots[t + self.degree + 1] - self.knots[t + self.degree])
            b += vel / self.degree * (self.knots[t + self.degree + 2] - self.knots[t + self.degree + 1])
        else:
            raise Exception("Not implemented yet")
    
    def overwriteSmooth(self, points: np.ndarray, times_rel: np.ndarray, time_cut: float):
        x_cut, xDot_cut, _ = self.eval3(time_cut, derivative=1)

        _points = points.copy()
        _times = times_rel.copy()
        _times = np.concatenate(([0.0], _times))
        _points = np.concatenate(([x_cut], _points))
        self.set(self.degree, _points, _times + time_cut, xDot_cut)
        
    def calcB(self, t: float, derivatives: int=0):
        self.B = np.zeros((self.knots.shape[0] - self.degree, self.degree+1))
        if derivatives > 0:
            self.Bdot = np.zeros((self.knots.shape[0] - self.degree, self.degree+1))
        if derivatives > 1:
            self.Bddot = np.zeros((self.knots.shape[0] - self.degree, self.degree+1))

        def _DIV(x: float, y: float) -> float:
            if x == 0.0:
                return 0.0
            else:
                if y == 0.0:
                    return 0.0
                else:
                    return x / y

        # Initialize rank 0
        for k in range(self.B.shape[0]):
            if self.knots[k] <= t and k+1 < self.knots.shape[0] and t < self.knots[k+1]:
                self.B[k, 0] = 1.0

        # Recursion
        for p in range(self.degree):
            for i in range(self.B.shape[0]):
                if i+p < self.knots.shape[0]:
                    xnom = t - self.knots[i]
                    xden = self.knots[i+p] - self.knots[i]
                    if xden != 0.:
                        x = _DIV(xnom, xden)
                        self.B[i, p] = x * self.B[i, p-1]
                        if derivatives > 0:
                            self.Bdot[i, p] = _DIV(1., xden) * self.B[i, p-1] + x * self.Bdot[i, p-1]
                        if derivatives > 1:
                            self.Bddot[i, p] = _DIV(2., xden) * self.Bdot[i, p-1] + x * self.Bddot[i, p-1]

                if i+1 < self.knots.shape[0] and i+p+1 < self.knots.shape[0] and i+1 < self.B.shape[0]:
                    ynom = self.knots[i+p+1] - t
                    yden = self.knots[i+p+1] - self.knots[i+1]
                    if yden != 0.:
                        y = _DIV(ynom, yden)
                        self.B[i, p] += y * self.B[i+1, p-1]
                        if derivatives > 0:
                            self.Bdot[i, p] += _DIV(-1., yden) * self.B[i+1, p-1] + y * self.Bdot[i+1, p-1]
                        if derivatives > 1:
                            self.Bddot[i, p] += _DIV(-2., yden) * self.Bdot[i+1, p-1] + y * self.Bddot[i+1, p-1]

        # Special case: outside the knots
        if t >= self.knots[-1]:
            self.B[-2, -1] = 1.0
        
    def eval3(self, t: float, derivative: int=2):
        
        n = self.ctrlPoints.shape[1]
        x = np.zeros(n)
        xDot = np.zeros(n)
        xDDot = np.zeros(n)

        # Handle out-of-interval cases
        if t < self.knots[0]:
            x = self.ctrlPoints[0]
            return x, xDot, xDDot
        
        if t >= self.knots[-1]:
            x = self.ctrlPoints[-1]
            return x, xDot, xDDot

        # Grab range of relevant knots
        center = np.searchsorted(self.knots, t, side="right")
        center -= 1
        lo: int = center - self.degree
        if lo < 0: lo = 0
        up: int = lo + 2 * self.degree
        if up > self.knots.shape[0]-1: up = self.knots.shape[0]-1
        lo: int = up - 2 * self.degree
        if lo < 0: lo = 0

        # Compute B-spline coefficients on relevant knots
        core = BSpline()
        core.degree = self.degree
        core.knots = self.knots[lo:up+1]
        core.calcB(t, derivative)

        # Multiply coefficients with control points
        b, bd, bdd = 0.0, 0.0, 0.0
        for j in range(core.B.shape[0]):
            
            b = core.B[j, self.degree]
                
            if lo + j >= self.ctrlPoints.shape[0]:
                assert np.abs(b) <= 1e-4
                continue
            
            for i in range(n):
                x[i] += b * self.ctrlPoints[lo+j, i]
                
            if derivative >= 1:
                bd = core.Bdot[j, self.degree]
                for i in range(n):
                    xDot[i] += bd * self.ctrlPoints[lo+j, i]
                    
            if derivative >= 2:
                bdd = core.Bddot[j, self.degree]
                for i in range(n):
                    xDDot[i] += bdd * self.ctrlPoints[lo+j, i]
    
        return x, xDot, xDDot
    
    def getKnots(self) -> np.ndarray:
        return self.knots

    def getCtrlPoints(self) -> np.ndarray:
        return self.ctrlPoints


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

