"""Connected Component Analysis fruit detection for Pipeline B."""

from time import perf_counter

import cv2


def detect_fruits(binary_mask, min_component_area):

    started = perf_counter()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )

    out = []

    for label in range(1, num_labels):

        area = float(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area < min_component_area:
            continue

        x = int(
            stats[label, cv2.CC_STAT_LEFT]
        )

        y = int(
            stats[label, cv2.CC_STAT_TOP]
        )

        w = int(
            stats[label, cv2.CC_STAT_WIDTH]
        )

        h = int(
            stats[label, cv2.CC_STAT_HEIGHT]
        )

        center = (
            float(centroids[label][0]),
            float(centroids[label][1])
        )

        out.append({
            'label': label,
            'area': area,
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'centroid': center
        })

    return out, perf_counter() - started