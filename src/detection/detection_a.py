"""External contour fruit detection for Pipeline A."""
from time import perf_counter
import cv2
def detect_fruits(binary_mask, min_contour_area):
    started=perf_counter(); contours,_=cv2.findContours(binary_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); out=[]
    for c in contours:
        area=float(cv2.contourArea(c))
        if area < min_contour_area: continue
        x,y,w,h=cv2.boundingRect(c); m=cv2.moments(c); center=(m['m10']/m['m00'],m['m01']/m['m00']) if m['m00'] else None
        out.append({'contour':c,'area':area,'x':x,'y':y,'width':w,'height':h,'centroid':center})
    return out, perf_counter()-started
