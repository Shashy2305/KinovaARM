"""
The colour NAME of a detected object (pure numpy/OpenCV, no ROS): "white", "black", "grey", "blue", "red", ... from the pixels of its bounding box.

Why: "pick up the white cup" needs the scene to know which cup is white. YOLO gives the class only. The classifier looks at the pixels inside the box (an ellipse inscribed in the central part, so the corners, which are
mostly table, are left out), throws away the table's own brown, and decides:
  * enough CHROMATIC pixels (saturated, not dark, not table-coloured)  -> the name of the commonest hue;
  * otherwise a neutral: white (bright), black (dark) or grey, by which of them covers more.
A cup's inside is often in shadow, so the neutral decision weighs bright pixels double against dark ones when the object is not mostly black.

Names: white, black, grey, red, orange, yellow, green, blue (teal and cyan count as blue), purple, pink, brown. 'unknown' when there are too few usable pixels.
"""
import cv2
import numpy as np

COLOUR_WORDS = ('white', 'black', 'grey', 'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown')
MIN_PIXELS = 150
CHROMATIC_SHARE = 0.22          # this share of the usable pixels must be one coloured hue for a coloured name
SAT_MIN = 105                   # OpenCV scale 0..255. A camera's colour cast makes white and grey read as saturation 60-90 (RS1: a white cup at S=84), a real blue cup is 130-245
SAT_MIN_BRIGHT = 150            # bright pixels (V >= BRIGHT_V) must be even more saturated to count as coloured: the cast shows most there
VAL_MIN = 45
DARK_UNKNOWN_V = 15             # median brightness (0..255) of the object's pixels below which a neutral name is not trusted
DARK_CHROMA_MIN = 22            # max-min of the B,G,R channels for a dark pixel to count as coloured (a black object's channels are within ~10 of each other)
BRIGHT_V = 150
DARK_V = 70
NEUTRAL_S = 100                 # neutrals may carry a cast too
# the dining table is red-brown: hue 3..22, mid saturation and value
TABLE_HUE = (2, 22)
TABLE_SAT = (90, 235)
TABLE_VAL = (60, 215)

# OpenCV hue (0..179) -> name, for pixels that are coloured
HUE_RANGES = ((0, 8, 'red'), (8, 20, 'orange'), (20, 35, 'yellow'), (35, 80, 'green'), (80, 130, 'blue'), (130, 150, 'purple'), (150, 172, 'pink'), (172, 180, 'red'))


def _hue_name(h):
    for lo, hi, name in HUE_RANGES:
        if lo <= h < hi:
            return name
    return 'red'


def _ellipse_mask(shape, box, inner=0.8):
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
    mask = np.zeros((h, w), np.uint8)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return mask
    cv2.ellipse(mask, ((x1 + x2) // 2, (y1 + y2) // 2), (max(1, int((x2 - x1) * inner / 2.0)), max(1, int((y2 - y1) * inner / 2.0))), 0, 0, 360, 255, -1)
    return mask


def colour_stats(bgr, box, mask=None):
    """{name: pixel count} over the usable pixels of the object (table colour removed), plus the neutral counts 'white', 'black', 'grey'."""
    m = mask if mask is not None else _ellipse_mask(bgr.shape, box)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    sel = m > 0
    table = (h >= TABLE_HUE[0]) & (h <= TABLE_HUE[1]) & (s >= TABLE_SAT[0]) & (s <= TABLE_SAT[1]) & (v >= TABLE_VAL[0]) & (v <= TABLE_VAL[1])
    use = sel & ~table
    stats = {}
    chroma = use & (s >= SAT_MIN) & (v >= VAL_MIN) & ((v < BRIGHT_V) | (s >= SAT_MIN_BRIGHT))
    for lo, hi, name in HUE_RANGES:
        n = int(np.count_nonzero(chroma & (h >= lo) & (h < hi)))
        if n:
            stats[name] = stats.get(name, 0) + n
    # a dark coloured object (a dark teal cup) has pixels below VAL_MIN whose HUE is unreliable but whose channels still differ: count them as coloured when the spread between the channels is clear
    mx, mn = bgr.max(axis=2).astype(int), bgr.min(axis=2).astype(int)
    dark_chroma = use & (v < VAL_MIN) & ((mx - mn) >= DARK_CHROMA_MIN) & (s >= SAT_MIN)
    for lo, hi, name in HUE_RANGES:
        n = int(np.count_nonzero(dark_chroma & (h >= lo) & (h < hi)))
        if n:
            stats[name] = stats.get(name, 0) + n
    chroma = chroma | dark_chroma
    neutral = use & ~chroma
    stats['white'] = int(np.count_nonzero(neutral & (v >= BRIGHT_V) & (s < NEUTRAL_S)))
    stats['black'] = int(np.count_nonzero(neutral & (v < DARK_V)))
    stats['grey'] = int(np.count_nonzero(neutral & (v >= DARK_V) & (v < BRIGHT_V) & (s < NEUTRAL_S)))
    stats['_usable'] = int(np.count_nonzero(use))
    stats['_vmed'] = float(np.median(v[use])) if np.any(use) else 0.0          # how bright the object is in this image (underexposed -> no colour can be told)
    return stats


def colour_name(bgr, box, mask=None):
    """The colour name of the object in `box` (x1, y1, x2, y2 pixels) of the BGR image, or 'unknown'."""
    try:
        st = colour_stats(bgr, box, mask)
    except Exception:
        return 'unknown'
    usable = st.pop('_usable', 0)
    vmed = st.pop('_vmed', 255.0)
    if usable < MIN_PIXELS:
        return 'unknown'
    coloured = {k: v for k, v in st.items() if k not in ('white', 'black', 'grey')}
    if coloured:
        best = max(coloured, key=coloured.get)
        if coloured[best] / usable >= CHROMATIC_SHARE:
            return best
    if vmed < DARK_UNKNOWN_V:
        return 'unknown'                 # RS2 saw a dark teal cup at brightness 12: that is not 'black', it is too dark to tell
    white, black, grey = st['white'], st['black'], st['grey']
    # a cup's inside is in shadow: bright pixels count double against dark ones unless dark clearly dominates
    if white * 2 >= max(black, grey) and white / usable >= 0.12:
        return 'white' if white * 1.0 >= black * 0.5 else 'black'
    return max(('black', 'grey', 'white'), key=lambda k: st[k])


def fallback_colour(label):
    """The colour implied by a colour-blob label from the HSV fallback detector ('red_obj' -> 'red'), else 'unknown'."""
    word = str(label).split('_')[0]
    return word if word in COLOUR_WORDS else 'unknown'
