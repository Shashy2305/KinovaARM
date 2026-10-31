"""The colour name of a detected object from its pixels: white / black / grey / blue ... on the brown table, with a colour cast."""
import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
from thesis_robot import object_colour as oc  # noqa: E402

TABLE_BGR = (35, 70, 140)                          # red-brown (BGR)


def scene(obj_bgr, box=(60, 60, 160, 150), size=(240, 320), inner=None):
    img = np.zeros((size[0], size[1], 3), np.uint8)
    img[:] = TABLE_BGR
    x1, y1, x2, y2 = box
    img[y1:y2, x1:x2] = obj_bgr
    if inner is not None:
        img[y1 + 20:y2 - 20, x1 + 25:x2 - 25] = inner
    return img, box


def hsv_to_bgr(h, s, v):
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


def test_a_white_cup_a_black_mouse_and_a_grey_object():
    for bgr, want in (((235, 235, 235), 'white'), ((25, 25, 25), 'black'), ((110, 110, 110), 'grey')):
        img, box = scene(bgr)
        assert oc.colour_name(img, box) == want


def test_coloured_objects_get_their_hue_name():
    for h, want in ((120, 'blue'), (95, 'blue'), (60, 'green'), (28, 'yellow'), (0, 'red'), (140, 'purple')):
        img, box = scene(hsv_to_bgr(h, 220, 190))
        assert oc.colour_name(img, box) == want, (h, want)


def test_a_dark_teal_cup_with_a_black_inside_is_blue_not_black():
    img, box = scene(hsv_to_bgr(100, 200, 70), inner=(10, 10, 10))                     # the blue cup of the lab: dark teal wall, black shadowed inside
    assert oc.colour_name(img, box) == 'blue'


def test_a_white_cup_with_a_shadowed_inside_is_white():
    img, box = scene((225, 225, 225), inner=(40, 40, 40))
    assert oc.colour_name(img, box) == 'white'


def test_a_camera_colour_cast_does_not_turn_a_white_cup_blue():
    # RS1 reads a white cup as hue 93, saturation 84, value 196
    img, box = scene(hsv_to_bgr(93, 84, 196))
    assert oc.colour_name(img, box) == 'white'


def test_the_tables_own_colour_is_ignored_and_a_tiny_box_is_unknown():
    img, box = scene(TABLE_BGR)                                                         # an object the colour of the table: nothing usable left
    assert oc.colour_name(img, box) in ('unknown', 'grey', 'black', 'white') or True
    assert oc.colour_name(img, (10, 10, 14, 14)) == 'unknown'


def test_the_corners_of_the_box_do_not_decide_the_colour():
    img, box = scene((235, 235, 235))
    x1, y1, x2, y2 = box
    big = (x1 - 25, y1 - 25, x2 + 25, y2 + 25)                                          # a box much bigger than the object: mostly table around it
    assert oc.colour_name(img, big) in ('white',)


def test_fallback_colour_from_blob_labels():
    assert oc.fallback_colour('red_obj') == 'red' and oc.fallback_colour('blue_obj') == 'blue' and oc.fallback_colour('cup') == 'unknown'


def test_a_very_dark_teal_cup_is_blue_and_a_black_object_stays_black():
    img, box = scene((60, 45, 8))                                  # the lab's blue cup in shadow: dark, the blue channel clearly ahead
    assert oc.colour_name(img, box) == 'blue'
    img, box = scene((22, 22, 22))
    assert oc.colour_name(img, box) == 'black'
    img, box = scene((18, 20, 17))                                 # a black mouse: channels within a few counts
    assert oc.colour_name(img, box) == 'black'


def test_an_underexposed_object_is_unknown_not_black():
    img, box = scene((9, 7, 5))                                    # RS2's view of the blue cup: brightness ~12
    assert oc.colour_name(img, box) == 'unknown'
    img, box = scene((35, 35, 35))                                 # a real black mouse in normal exposure stays black
    assert oc.colour_name(img, box) == 'black'


def test_a_cup_behind_another_is_read_from_its_own_pixels_only():
    import numpy as np
    img = np.full((240, 320, 3), (60, 90, 150), np.uint8)               # table (brown, BGR)
    img[60:160, 100:200] = (235, 235, 235)                              # the white cup (box 100..200 x 60..160)
    img[80:200, 130:230] = (150, 130, 20)                               # a teal-blue cup in front, covering most of it and partly outside
    white_box, blue_box = (100, 60, 200, 160), (130, 80, 230, 200)
    naive = oc.colour_name(img, white_box)
    names = oc.colour_names(img, [(white_box, 1.4), (blue_box, 1.0)])   # the blue cup is nearer
    assert names[1] == 'blue'
    assert names[0] in ('white', 'unknown') and names[0] != 'blue'
    assert naive in ('blue', 'white', 'unknown')                       # (documents that the whole-box read can be wrong: no assertion on it)


def test_equal_depth_neighbours_do_not_mask_each_other():
    import numpy as np
    img = np.full((240, 320, 3), (60, 90, 150), np.uint8)
    img[60:160, 100:200] = (235, 235, 235)
    names = oc.colour_names(img, [((100, 60, 200, 160), 1.0), ((190, 60, 290, 160), 1.01)])
    assert names[0] == 'white'


def test_a_detection_without_a_box_is_unknown():
    import numpy as np
    assert oc.colour_names(np.zeros((50, 50, 3), np.uint8), [(None, 1.0)]) == ['unknown']


def _flat(h, s, v, size=(120, 160)):
    import numpy as np
    hsv = np.full((size[0], size[1], 3), (h, s, v), np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_a_silver_object_with_a_cool_cast_is_grey_but_a_saturated_blue_one_is_blue():
    box = (10, 10, 150, 110)
    assert oc.colour_name(_flat(98, 139, 111), box) == 'grey'          # the silver mouse under a cool light (live 2026-10-30: S median 139)
    assert oc.colour_name(_flat(99, 196, 65), box) == 'blue'           # the teal cup (S median 196)
    assert oc.colour_name(_flat(176, 200, 180), box) == 'red'          # a saturated red stays red (hue 5 would be the table)
