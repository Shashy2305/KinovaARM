"""detector_consensus.py: how independent opinions about an object's identity are combined (the shadow-mode second detector)."""
import numpy as np

from thesis_robot import detector_consensus as dc


def test_confusable_labels_agree_by_group_not_by_exact_label():
    r = dc.consensus(a='cup', b='vase')                     # the cup-from-above confusion of every COCO model
    assert r['verdict'] == 'agree' and r['group'] == 'container'
    assert dc.consensus(a='mouse', b='cell phone')['verdict'] == 'agree'
    assert dc.consensus(a='cup', b='mouse')['verdict'] == 'uncertain'


def test_three_opinions_majority_and_label_choice():
    r = dc.consensus(a='vase', b='cup', c='cup')
    assert r['verdict'] == 'agree' and r['label'] == 'cup'                 # the verifier names the label inside the group
    r = dc.consensus(a='bottle', b='cup', c='cup')
    assert r['verdict'] == 'majority' and r['group'] == 'container' and r['label'] == 'cup'
    r = dc.consensus(a='bottle', b='cup', c='mouse')
    assert r['verdict'] == 'uncertain' and r['group'] is None and r['label'] is None


def test_missing_opinions():
    assert dc.consensus()['verdict'] == 'none'
    r = dc.consensus(a='cup')
    assert r['verdict'] == 'single' and r['label'] == 'cup' and r['n'] == 1
    assert dc.consensus(a='cup', c='cup')['verdict'] == 'agree'                # B missed the object: two opinions still agree


def test_an_unknown_label_agrees_only_with_itself():
    assert dc.group_of('traffic light') == 'traffic light'
    assert dc.consensus(a='traffic light', b='traffic light')['verdict'] == 'agree'
    assert dc.consensus(a='traffic light', b='cup')['verdict'] == 'uncertain'


def test_box_matching_prefers_the_tightest_box_and_ignores_enclosing_ones():
    boxes = [(0, 0, 640, 480, 0.9), (100, 100, 200, 200, 0.8), (120, 120, 180, 180, 0.3)]
    assert dc.box_containing((150, 150), boxes) == 2
    assert dc.box_containing((150, 150), boxes, min_conf=0.5) == 1
    assert dc.box_containing((320, 240), boxes) == 0                    # the pixel is at the centre of the big box
    assert dc.box_containing((700, 10), boxes) is None
    # a mouse on a chair: the chair box contains the pixel but the pixel is far from its centre: no opinion, not "chair"
    chair = [(300, 100, 560, 700, 0.95)]
    assert dc.box_containing((423, 383), chair) is not None             # near the centre vertically (400)...
    assert dc.box_containing((330, 130), chair) is None                 # ...but a pixel in a corner of it is not the chair


def test_projection():
    assert dc.project_to_pixel(0.1, -0.05, 1.0, 600, 600, 320, 240) == (380.0, 210.0)
    assert dc.project_to_pixel(0.1, 0.0, 0.0, 600, 600, 320, 240) is None


def test_knn_abstains_on_something_new():
    bank = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    labels = ['cup', 'mouse']
    assert dc.knn_label([0.9, 0.1, 0], bank, labels)[0] == 'cup'
    label, sim = dc.knn_label([0, 0, 1.0], bank, labels)                        # orthogonal to everything verified
    assert label is None and sim < 0.1
    assert dc.knn_label([1, 0, 0], np.zeros((0, 3)), [])[0] is None


def test_near_any():
    assert dc.near_any((100, 100), [(150, 100)]) and not dc.near_any((100, 100), [(300, 100)]) and not dc.near_any((1, 1), [])
