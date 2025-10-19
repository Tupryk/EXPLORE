import cv2
import random
import numpy as np
from scipy.interpolate import make_interp_spline


class ND_BSpline:
    def __init__(self, start_time: float, start_pos: np.ndarray, degree: int=2):
        self.start_time = start_time
        self.start_pos = start_pos
        self.degree = degree
        
    def compute(self, points: np.ndarray, times: np.ndarray, end_time: float):
        points = np.array([self.start_pos, *points])
        times = np.array([self.start_time, *times])
        times += self.start_time
        times[0] -= self.start_time
        self.splines = [make_interp_spline(times, points[:, dim], k=self.degree) for dim in range(points.shape[1])]
        
    def eval(self, t: float):
        point = np.array([s(t) for s in self.splines])
        return point


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
