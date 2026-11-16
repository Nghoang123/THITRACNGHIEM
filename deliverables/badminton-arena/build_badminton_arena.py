"""
PROFESSIONAL BADMINTON INDOOR ARENA - procedural scene generator
=================================================================
Blender 4.2 LTS or newer (tested on 4.5 LTS and 5.3).  Units: metres, Z up.
The court is centred on the world origin, its long axis runs along Y and the
net runs along X.

HOW TO RUN
  In Blender : Scripting workspace > Text > Open > this file > Run Script.
  Headless   : blender -b -P build_badminton_arena.py -- --save Arena.blend
               add  --render <dir>        to render every camera to PNG
                    --cams A,B            to render only the named cameras
                    --samples 128         Cycles samples
                    --scale 50            render resolution percentage
                    --export-fbx <dir>    one Unity-ready FBX per collection
                    --quick               thin crowd, for fast previews

WARNING: with CLEAR_SCENE = True (default) the script deletes every object in
the current .blend before it builds.  Run it in a new, empty file.

Collections created: ARENA_STRUCTURE, BADMINTON_COURT, COURT_LINES,
NET_AND_POSTS, LED_ADVERTISING, REFEREE_AND_OFFICIALS, SPECTATORS, LIGHTING,
CAMERAS.

Brand names (YONEX, LI-NING, VICTOR, BWF) appear as plain text only; no
official logo artwork is used or reproduced.
"""

import math
import os
import random
import sys

import bpy
import bmesh
from mathutils import Matrix, Vector, noise

# =============================================================================
# CONFIGURATION
# =============================================================================
CLEAR_SCENE = True
SEED = 2026
RENDER_ENGINE = "CYCLES"          # "CYCLES" or "EEVEE"
RESOLUTION = (1920, 1080)
SAMPLES = 160

# --- court (BWF Laws of Badminton, doubles court, lines 40 mm wide) ---------
COURT_L = 13.40                   # outer edge to outer edge
COURT_W = 6.10
LINE_W = 0.04
SINGLES_INSET = 0.46              # doubles side line -> singles side line
SHORT_SERVICE = 1.98              # net -> near edge of short service line
LONG_SERVICE_DBL = 0.76           # back boundary -> long service line (doubles)
NET_H_POST = 1.55
NET_H_CENTRE = 1.524
NET_DEPTH = 0.76
MAT_HALF = (5.90, 10.5)           # court mat half size (run-off included)
MAT_T = 0.006                     # mat thickness

# --- arena bowl ---------------------------------------------------------------
FX, FY = 10.0, 14.5               # half size of the flat arena floor
ROWS = 12
ROW_D, ROW_H = 0.85, 0.42         # tread depth / riser height
STAND_Z0 = 1.0                    # height of the first row
SEAT_PITCH = 0.50
CONCOURSE_W = 2.4
ROOF_Z = 14.0                     # underside of roof trusses
TRUSS_DEPTH = 2.2
AISLE_W = 1.1
VOM_HALF = 1.3                    # vomitory half width
VOM_ROWS = (2, 7)                 # vomitory cuts rows [2, 7)
OCCUPANCY = 0.90
SPECTATOR_OUTFITS = 10            # colour variations stored per pose mesh

# --- LED perimeter boards -----------------------------------------------------
LED_X, LED_Y = 6.6, 11.2          # distance of the board ring from the centre
LED_LEN, LED_H, LED_TILT = 3.0, 0.90, math.radians(10)
LED_BG_STRENGTH = 0.62            # emission of board backgrounds
LED_TEXT_STRENGTH = 0.85          # emission of board lettering

# --- lighting (see build_lighting) -------------------------------------------
WHITE_BALANCE_K = 5400            # the render is white balanced to the court
COURT_LIGHT_K = 5400              # neutral white floodlights (5000-5600 K)
AUDIENCE_LIGHT_K = 3700           # warm house lights
COURT_LIGHT_W = 100.0             # power of each of the 10 court floodlights
AUDIENCE_LIGHT_W = 2800.0         # power of each of the 4 stand washes
OFFICIALS_LIGHT_W = 260.0         # fill over the umpire / control zones
ROOF_WASH_W = 5000.0              # faint up-light on the roof steelwork
RIG_Z = 10.8                      # height of the flown lighting rig
RIG_X, RIG_Y = 6.5, 11.0

# --- main camera --------------------------------------------------------------
MAIN_CAM_POS = (0.0, -17.2, 6.4)

BRAND_ORDER = ["YONEX", "LI-NING", "VICTOR", "BWF"]


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = {"save": None, "render": None, "cams": None, "samples": None,
           "scale": None, "export_fbx": None, "quick": False, "engine": None}
    i = 0
    while i < len(argv):
        key = argv[i].lstrip("-").replace("-", "_")
        if key == "quick":
            out["quick"] = True
            i += 1
        elif key in out and i + 1 < len(argv):
            out[key] = argv[i + 1]
            i += 2
        else:
            i += 1
    return out


ARGS = _args()
V = Vector
RNG = random.Random(SEED)


# =============================================================================
# SMALL HELPERS
# =============================================================================
def srgb(r, g, b):
    """8-bit sRGB -> linear RGB tuple."""
    def f(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (f(r), f(g), f(b))


def _blackbody(kelvin):
    """Approximate linear RGB of a black body (Tanner Helland fit)."""
    t = kelvin / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * ((t - 60) ** -0.1332047592)
    g = (99.4708025861 * math.log(t) - 161.1195681661 if t <= 66
         else 288.1221695283 * ((t - 60) ** -0.0755148492))
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10) - 305.0447927307
    return srgb(*(max(0.0, min(255.0, c)) for c in (r, g, b)))


def kelvin_rgb(kelvin, white_balance=WHITE_BALANCE_K):
    """Light colour for a colour temperature as seen by a camera that is
    white balanced to `white_balance` (so the court lights render neutral)."""
    a, b = _blackbody(kelvin), _blackbody(white_balance)
    c = [x / y for x, y in zip(a, b)]
    m = max(c)
    return tuple(x / m for x in c)


def rot_z(angle):
    return Matrix.Rotation(angle, 4, "Z")


def facing(direction):
    """Z rotation that turns local +Y towards `direction` (x, y)."""
    return math.atan2(direction[1], direction[0]) - math.pi / 2


def plane_matrix(origin, normal):
    """Matrix whose +Z is `normal`, +X is horizontal and +Y points 'up' the
    plane: used to stick text (modelled in XY) onto walls and boards."""
    n = V(normal).normalized()
    x = V((0, 0, 1)).cross(n)
    if x.length < 1e-6:
        x = V((1, 0, 0))
    x.normalize()
    y = n.cross(x)
    m = Matrix(((x.x, y.x, n.x, origin[0]),
                (x.y, y.y, n.y, origin[1]),
                (x.z, y.z, n.z, origin[2]),
                (0, 0, 0, 1)))
    return m


# =============================================================================
# COLLECTIONS / OBJECTS
# =============================================================================
COLLECTION_NAMES = [
    "ARENA_STRUCTURE", "BADMINTON_COURT", "COURT_LINES", "NET_AND_POSTS",
    "LED_ADVERTISING", "REFEREE_AND_OFFICIALS", "SPECTATORS", "LIGHTING",
    "CAMERAS",
]
COL = {}
WORLD_M = {}     # object name -> world matrix (kept so children can be placed)


def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights,
                  bpy.data.cameras, bpy.data.curves, bpy.data.images):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def make_collections():
    root = bpy.context.scene.collection
    for name in COLLECTION_NAMES:
        c = bpy.data.collections.get(name)
        if c is None:
            c = bpy.data.collections.new(name)
        if c.name not in root.children:
            root.children.link(c)
        COL[name] = c


def sub_collection(parent, name):
    c = bpy.data.collections.new(name)
    COL[parent].children.link(c)
    COL[name] = c
    return c


def add_object(name, data, coll, matrix=None, parent=None):
    ob = bpy.data.objects.new(name, data)
    COL[coll].objects.link(ob)
    m = matrix.copy() if matrix is not None else Matrix.Identity(4)
    if parent is not None:
        ob.parent = parent
        ob.matrix_parent_inverse = Matrix.Identity(4)
        ob.matrix_basis = WORLD_M[parent.name].inverted() @ m
    else:
        ob.matrix_basis = m
    WORLD_M[ob.name] = m
    return ob


# =============================================================================
# MATERIALS
# =============================================================================
MATS = {}


def _principled(mat):
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    for n in nt.nodes:
        if n.bl_idname == "ShaderNodeBsdfPrincipled":
            return n
    nt.nodes.clear()
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    return bsdf


def _set(node, names, value):
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return True
    return False


def make_mat(name, color, rough=0.6, metal=0.0, emit=None, strength=0.0,
             alpha=1.0):
    """PBR material on a single Principled BSDF (exports cleanly to Unity)."""
    if name in MATS:
        return MATS[name]
    mat = bpy.data.materials.new(name)
    bsdf = _principled(mat)
    _set(bsdf, ["Base Color"], (*color, 1.0))
    _set(bsdf, ["Roughness"], rough)
    _set(bsdf, ["Metallic"], metal)
    if emit is not None:
        _set(bsdf, ["Emission Color", "Emission"], (*emit, 1.0))
        _set(bsdf, ["Emission Strength"], strength)
    if alpha < 1.0:
        _set(bsdf, ["Alpha"], alpha)
    mat.diffuse_color = (*(emit if emit is not None and strength > 0.3
                           else color), 1.0)
    mat.roughness = rough
    mat.metallic = metal
    MATS[name] = mat
    return mat


def add_surface_noise(mat, scale=6.0, rough_range=(0.4, 0.6), bump=0.0):
    """Procedural roughness / bump variation.  Blender-only detail: the plain
    Principled values still export to Unity unchanged."""
    nt = mat.node_tree
    bsdf = _principled(mat)
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.location = (-700, 0)
    tex.inputs["Scale"].default_value = scale
    tex.inputs["Detail"].default_value = 6.0
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.location = (-450, 0)
    ramp.inputs["To Min"].default_value = rough_range[0]
    ramp.inputs["To Max"].default_value = rough_range[1]
    nt.links.new(tex.outputs[0], ramp.inputs["Value"])
    nt.links.new(ramp.outputs[0], bsdf.inputs["Roughness"])
    if bump > 0:
        bmp = nt.nodes.new("ShaderNodeBump")
        bmp.location = (-450, -300)
        bmp.inputs["Strength"].default_value = bump
        bmp.inputs["Distance"].default_value = 0.01
        nt.links.new(tex.outputs[0], bmp.inputs["Height"])
        nt.links.new(bmp.outputs[0], bsdf.inputs["Normal"])


def build_materials():
    m = make_mat
    # structure
    m("Concrete_Stand", srgb(52, 54, 60), 0.85)
    m("Concrete_Wall", srgb(38, 40, 46), 0.8)
    m("Wall_Panel", srgb(30, 32, 38), 0.7)
    m("Steel_Truss", srgb(74, 78, 86), 0.5, 0.2)
    m("Steel_Dark", srgb(22, 23, 26), 0.5, 0.8)
    m("Roof_Deck", srgb(20, 21, 24), 0.9)
    m("Rail_Metal", srgb(120, 124, 130), 0.35, 1.0)
    m("Seat_Plastic", srgb(22, 52, 122), 0.45)
    m("Doorway_Dark", srgb(8, 8, 9), 0.9, emit=kelvin_rgb(3000), strength=0.012)
    m("Aisle_StepLight", (0, 0, 0), 0.5, emit=kelvin_rgb(2900), strength=8.0)
    m("Arena_Floor", srgb(40, 44, 54), 0.42)
    # court
    m("Court_Mat", srgb(108, 111, 119), 0.62)
    m("Court_Line", srgb(236, 238, 240), 0.55)
    m("Court_Branding", srgb(196, 199, 206), 0.6)
    # net
    m("Net_Tape", srgb(240, 240, 238), 0.7)
    m("Post_Blue", srgb(16, 62, 170), 0.35, 0.3)
    m("Rubber_Black", srgb(14, 14, 15), 0.8)
    # officials' furniture
    m("Umpire_Blue", srgb(18, 66, 178), 0.38, 0.15)
    m("Chair_Black", srgb(18, 18, 20), 0.5)
    m("Table_Cloth", srgb(16, 30, 74), 0.85)
    m("Print_White", srgb(235, 236, 238), 0.6)
    m("Device_Dark", srgb(20, 21, 24), 0.4, 0.5)
    m("Screen_Glow", (0, 0, 0), 0.3, emit=srgb(120, 160, 230), strength=1.2)
    m("Lens_Glass", srgb(6, 8, 14), 0.08, 0.6)
    # scoreboard
    m("Score_Panel", srgb(6, 7, 10), 0.35, emit=srgb(10, 14, 24), strength=0.25)
    m("Score_White", (0, 0, 0), 0.4, emit=srgb(240, 240, 240), strength=1.1)
    m("Score_Amber", (0, 0, 0), 0.4, emit=srgb(255, 176, 20), strength=1.6)
    m("Score_Blue", (0, 0, 0), 0.4, emit=srgb(30, 110, 255), strength=0.9)
    # LED advertising
    m("LED_Frame", srgb(14, 15, 17), 0.5, 0.6)
    m("LED_Text_White", (0, 0, 0), 0.4, emit=(1.0, 1.0, 1.0),
      strength=LED_TEXT_STRENGTH)
    led = {
        "YONEX": (srgb(0, 74, 200), srgb(0, 170, 84)),
        "LI-NING": (srgb(222, 24, 44), srgb(150, 16, 96)),
        "VICTOR": (srgb(12, 34, 128), srgb(0, 150, 220)),
        "BWF": (srgb(18, 22, 40), srgb(214, 22, 40)),
    }
    for brand, (bg, accent) in led.items():
        m("LED_%s_BG" % brand, (0, 0, 0), 0.4, emit=bg,
          strength=LED_BG_STRENGTH)
        m("LED_%s_Accent" % brand, (0, 0, 0), 0.4, emit=accent,
          strength=LED_BG_STRENGTH)
    m("LED_Screen_Dark", (0, 0, 0), 0.4, emit=srgb(14, 18, 34), strength=0.5)
    m("LED_BWF_Red", (0, 0, 0), 0.4, emit=srgb(214, 22, 40), strength=1.1)
    # lighting hardware
    m("Fixture_Housing", srgb(20, 20, 22), 0.5, 0.7)
    m("Fixture_Lens", (0, 0, 0), 0.3, emit=kelvin_rgb(COURT_LIGHT_K),
      strength=22.0)
    m("HouseLight_Lens", (0, 0, 0), 0.3, emit=kelvin_rgb(AUDIENCE_LIGHT_K),
      strength=9.0)
    # officials
    m("Official_Shirt", srgb(22, 23, 28), 0.75)
    m("Official_Trousers", srgb(16, 16, 19), 0.8)
    m("Shoe_Dark", srgb(18, 18, 20), 0.6)
    # people palettes
    for i, c in enumerate([(226, 188, 160), (208, 164, 132), (186, 140, 108),
                           (150, 106, 80), (104, 74, 58)]):
        m("Skin_%d" % i, srgb(*c), 0.6)
    for i, c in enumerate([(14, 12, 12), (30, 22, 18), (58, 40, 28),
                           (112, 86, 58), (130, 130, 132), (176, 150, 100)]):
        m("Hair_%d" % i, srgb(*c), 0.55)
    tops = [(30, 38, 62), (222, 222, 218), (62, 64, 70), (20, 20, 23),
            (138, 44, 48), (52, 82, 128), (58, 96, 78), (190, 164, 84),
            (122, 126, 132), (160, 98, 62), (88, 68, 108), (188, 180, 164),
            (44, 96, 104), (150, 74, 96), (84, 92, 68), (206, 196, 188)]
    for i, c in enumerate(tops):
        m("Top_%02d" % i, srgb(*c), 0.8)
    for i, c in enumerate([(30, 38, 60), (18, 18, 20), (70, 72, 78),
                           (140, 122, 92), (44, 56, 84), (90, 84, 76)]):
        m("Bottom_%d" % i, srgb(*c), 0.8)

    add_surface_noise(MATS["Arena_Floor"], 3.0, (0.34, 0.5))
    add_surface_noise(MATS["Court_Mat"], 40.0, (0.56, 0.68), bump=0.05)
    add_surface_noise(MATS["Concrete_Stand"], 5.0, (0.75, 0.95), bump=0.1)
    add_surface_noise(MATS["Concrete_Wall"], 2.0, (0.7, 0.9), bump=0.08)


# =============================================================================
# MESH BUILDER
# =============================================================================
class MeshBuilder:
    """Accumulates clean quad / n-gon geometry with outward-facing normals and
    writes it to a mesh with box-projected UVs (1 UV unit = 1 metre)."""

    def __init__(self, materials):
        self.materials = [MATS[n] if isinstance(n, str) else n
                          for n in materials]
        self.verts, self.faces, self.fmat, self.fsmooth = [], [], [], []
        self.fuv = {}
        self.M = None

    # -- low level ----------------------------------------------------------
    def add_verts(self, pts):
        base = len(self.verts)
        if self.M is not None:
            for p in pts:
                self.verts.append(tuple(self.M @ V(p)))
        else:
            for p in pts:
                self.verts.append((p[0], p[1], p[2]))
        return base

    def add_face(self, idx, mat=0, smooth=False, uv=None):
        if uv is not None:
            self.fuv[len(self.faces)] = uv
        self.faces.append(tuple(idx))
        self.fmat.append(mat)
        self.fsmooth.append(smooth)

    # -- primitives -----------------------------------------------------------
    def quad(self, p0, p1, p2, p3, mat=0, uv=None):
        b = self.add_verts((p0, p1, p2, p3))
        self.add_face((b, b + 1, b + 2, b + 3), mat, False, uv)

    def hexa(self, p, mat=0, skip=()):
        """8 points: 0-3 bottom ring (CCW seen from above), 4-7 above them."""
        b = self.add_verts(p)
        quads = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
                 (2, 3, 7, 6), (3, 0, 4, 7))
        for k, q in enumerate(quads):
            if k not in skip:
                self.add_face([b + i for i in q], mat)

    def box(self, mn, mx, mat=0, skip=()):
        x0, y0, z0 = mn
        x1, y1, z1 = mx
        self.hexa(((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                   (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)),
                  mat, skip)

    def cbox(self, centre, size, mat=0, rz=0.0):
        hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
        c, s = math.cos(rz), math.sin(rz)
        pts = []
        for z in (-hz, hz):
            for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
                pts.append((centre[0] + x * c - y * s,
                            centre[1] + x * s + y * c, centre[2] + z))
        self.hexa(pts, mat)

    def beam(self, p0, p1, w, h, mat=0):
        """Rectangular bar between two points."""
        p0, p1 = V(p0), V(p1)
        a = (p1 - p0).normalized()
        side = a.cross(V((0, 0, 1)))
        if side.length < 1e-5:
            side = V((1, 0, 0))
        side.normalize()
        up = a.cross(side)
        pts = []
        for p in (p0, p1):
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                pts.append(p + side * (su * w / 2) + up * (sv * h / 2))
        self.hexa(pts, mat)

    def obox(self, centre, u, v, w, hu, hv, hw, mat=0):
        """Box on an arbitrary right-handed frame (u x v = w)."""
        c = V(centre)
        pts = []
        for sw_ in (-1, 1):
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                pts.append(c + u * (su * hu) + v * (sv * hv) + w * (sw_ * hw))
        self.hexa(pts, mat)

    @staticmethod
    def _frame(axis):
        ref = V((0, 0, 1)) if abs(axis.z) < 0.95 else V((1, 0, 0))
        u = axis.cross(ref).normalized()
        return u, axis.cross(u)

    def tube(self, p0, p1, r0, r1=None, segs=8, mat=0, cap0=True, cap1=True,
             smooth=True):
        p0, p1 = V(p0), V(p1)
        r1 = r0 if r1 is None else r1
        u, v = self._frame((p1 - p0).normalized())
        rings = []
        for p, r in ((p0, r0), (p1, r1)):
            rings.append([p + (u * math.cos(2 * math.pi * k / segs)
                               + v * math.sin(2 * math.pi * k / segs)) * r
                          for k in range(segs)])
        self.loft(rings, mat, cap0, cap1, smooth)

    def loft(self, rings, mat=0, cap0=True, cap1=True, smooth=True):
        """Skin a list of point rings (each wound CCW about the loft axis)."""
        n = len(rings[0])
        base = [self.add_verts(r) for r in rings]
        for j in range(len(rings) - 1):
            a, b = base[j], base[j + 1]
            for i in range(n):
                k = (i + 1) % n
                self.add_face((a + i, a + k, b + k, b + i), mat, smooth)
        # caps use their own vertices so they do not disturb smooth normals
        if cap0:
            b = self.add_verts(rings[0])
            self.add_face([b + i for i in reversed(range(n))], mat)
        if cap1:
            b = self.add_verts(rings[-1])
            self.add_face([b + i for i in range(n)], mat)

    def ellipsoid(self, centre, radii, segs=8, rings=5, mat=0, basis=None,
                  keep=None, shape=None):
        """UV ellipsoid.  `basis` = three axis vectors, `keep(x, y, z)` filters
        faces by their centre in unit-sphere space (for hair caps), `shape`
        maps unit-sphere points (for a jaw line)."""
        centre = V(centre)
        bx, by, bz = basis if basis else (V((1, 0, 0)), V((0, 1, 0)),
                                          V((0, 0, 1)))
        unit = [(0.0, 0.0, 1.0)]
        for j in range(1, rings):
            th = math.pi * j / rings
            for i in range(segs):
                ph = 2 * math.pi * i / segs
                unit.append((math.sin(th) * math.cos(ph),
                             math.sin(th) * math.sin(ph), math.cos(th)))
        unit.append((0.0, 0.0, -1.0))
        pts = []
        for x, y, z in unit:
            if shape:
                x, y, z = shape(x, y, z)
            pts.append(centre + bx * (x * radii[0]) + by * (y * radii[1])
                       + bz * (z * radii[2]))
        b = self.add_verts(pts)
        last = len(unit) - 1

        def ok(ids):
            if keep is None:
                return True
            cx = sum(unit[i][0] for i in ids) / len(ids)
            cy = sum(unit[i][1] for i in ids) / len(ids)
            cz = sum(unit[i][2] for i in ids) / len(ids)
            return keep(cx, cy, cz)

        for i in range(segs):
            k = (i + 1) % segs
            f = (0, 1 + i, 1 + k)
            if ok(f):
                self.add_face([b + q for q in f], mat, True)
            for j in range(rings - 2):
                r0, r1 = 1 + j * segs, 1 + (j + 1) * segs
                f = (r0 + i, r1 + i, r1 + k, r0 + k)
                if ok(f):
                    self.add_face([b + q for q in f], mat, True)
            r = 1 + (rings - 2) * segs
            f = (last, r + k, r + i)
            if ok(f):
                self.add_face([b + q for q in f], mat, True)

    # -- output ---------------------------------------------------------------
    def to_mesh(self, name):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.verts, [], self.faces)
        for mat in self.materials:
            me.materials.append(mat)
        me.polygons.foreach_set("material_index", self.fmat)
        me.polygons.foreach_set("use_smooth", self.fsmooth)
        me.update()
        uvl = me.uv_layers.new(name="UVMap")
        flat = [0.0] * (len(me.loops) * 2)
        vs = self.verts
        for poly in me.polygons:
            custom = self.fuv.get(poly.index)
            n = poly.normal
            ax, ay, az = abs(n.x), abs(n.y), abs(n.z)
            for k, li in enumerate(poly.loop_indices):
                if custom:
                    uv = custom[k]
                else:
                    p = vs[me.loops[li].vertex_index]
                    if az >= ax and az >= ay:
                        uv = (p[0], p[1])
                    elif ax >= ay:
                        uv = (p[1], p[2])
                    else:
                        uv = (p[0], p[2])
                flat[li * 2] = uv[0]
                flat[li * 2 + 1] = uv[1]
        uvl.data.foreach_set("uv", flat)
        me.validate()
        return me

    def to_object(self, name, coll, matrix=None, parent=None):
        return add_object(name, self.to_mesh(name), coll, matrix, parent)


# =============================================================================
# TEXT (brand names, scores, signage) - converted to shared, clean meshes
# =============================================================================
_FONT = {"loaded": False, "font": None}
_TEXT_MESH = {}


def _load_font():
    if _FONT["loaded"]:
        return _FONT["font"]
    _FONT["loaded"] = True
    here = []
    try:
        here.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    if bpy.data.filepath:
        here.append(os.path.dirname(bpy.data.filepath))
    for d in here:
        p = os.path.join(d, "fonts", "Inter-ExtraBoldItalic.otf")
        if os.path.isfile(p):
            try:
                _FONT["font"] = bpy.data.fonts.load(p)
                break
            except Exception:
                pass
    return _FONT["font"]


def text_mesh(body, material="LED_Text_White"):
    """Flat mesh of `body` in the XY plane, centred, cap height = 1.
    Cached per (text, material) so repeated signage shares mesh data."""
    key = (body, material)
    if key in _TEXT_MESH:
        return _TEXT_MESH[key]
    if (body, None) in _TEXT_MESH:
        me = _TEXT_MESH[(body, None)].copy()
        me.materials[0] = MATS[material]
        me.name = "Text_%s_%s" % (body.replace(" ", "_"), material)
        _TEXT_MESH[key] = me
        return me
    cu = bpy.data.curves.new("tmp_text", "FONT")
    cu.body = body
    cu.align_x = "CENTER"
    cu.resolution_u = 3
    font = _load_font()
    if font is not None:
        cu.font = font
    else:                       # built-in font: fake a bold italic
        cu.shear = 0.22
        cu.offset = 0.012
    ob = bpy.data.objects.new("tmp_text", cu)
    bpy.context.scene.collection.objects.link(ob)
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.curves.remove(cu)

    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(0.5),
                             verts=bm.verts, edges=bm.edges)
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    for v in bm.verts:
        v.co.x = (v.co.x - cx) / h
        v.co.y = (v.co.y - cy) / h
        v.co.z = 0.0
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    up = sum(f.normal.z for f in bm.faces)
    if up < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    uv = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        for lp in f.loops:
            lp[uv].uv = (lp.vert.co.x, lp.vert.co.y)
    bm.to_mesh(me)
    bm.free()
    me.name = "Text_%s_%s" % (body.replace(" ", "_"), material)
    me.materials.append(MATS[material])
    me["aspect"] = w / h
    _TEXT_MESH[key] = me
    _TEXT_MESH[(body, None)] = me
    return me


def add_text(name, body, fit_w, fit_h, matrix, material, coll, parent=None):
    """Place text on the plane described by `matrix` (see plane_matrix)."""
    me = text_mesh(body, material)
    s = min(fit_h, fit_w / me["aspect"])
    return add_object(name, me, coll, matrix @ Matrix.Scale(s, 4), parent)


# =============================================================================
# STAND GEOMETRY HELPERS
# =============================================================================
# Each stand is modelled in "side space": +X points away from the court, +Y
# runs along the stand, the front edge sits at x = front.  It is then rotated
# about Z into place.
SIDES = [
    # name, rotation, front distance, half length of the front edge
    ("East", 0.0, FX, FY),
    ("North", math.pi / 2, FY, FX),
    ("West", math.pi, FX, FY),
    ("South", 3 * math.pi / 2, FY, FX),
]


def row_z(i):
    return STAND_Z0 + i * ROW_H


def aisles_for(half_len):
    """Centres of stair aisles along a stand (side-space Y)."""
    outer = 9.5 if half_len > 12 else 6.6
    return [-outer, -(VOM_HALF + 0.12 + AISLE_W / 2),
            (VOM_HALF + 0.12 + AISLE_W / 2), outer]


def seat_positions(half_len, row):
    """Y positions of the seats in one row (aisles, vomitory and the mitred
    corners removed)."""
    depth = row * ROW_D + 0.55
    limit = half_len + depth - 0.75
    aisles = aisles_for(half_len)
    n = int(limit / SEAT_PITCH)
    out = []
    for k in range(-n, n + 1):
        y = k * SEAT_PITCH
        if any(abs(y - a) < AISLE_W / 2 + 0.22 for a in aisles):
            continue
        if VOM_ROWS[0] <= row < VOM_ROWS[1] and abs(y) < VOM_HALF + 0.35:
            continue
        out.append(y)
    return out


# =============================================================================
# ARENA_STRUCTURE
# =============================================================================
def build_structure():
    C = "ARENA_STRUCTURE"
    top_z = row_z(ROWS)
    depth = ROWS * ROW_D
    wall_x = FX + depth + CONCOURSE_W
    wall_y = FY + depth + CONCOURSE_W
    wall_top = ROOF_Z + TRUSS_DEPTH + 0.25

    # ---- arena floor ------------------------------------------------------
    mb = MeshBuilder(["Arena_Floor"])
    mb.box((-FX, -FY, -0.2), (FX, FY, 0.0), 0, skip=(0,))
    mb.to_object("Arena_Floor", C)

    # ---- grandstands, seats, rails (one set per side) ---------------------
    for name, ang, front, half in SIDES:
        M = rot_z(ang)
        mb = MeshBuilder(["Concrete_Stand", "Concrete_Wall", "Doorway_Dark"])
        mb.M = M
        # fascia wall in front of row 0
        fh = half if name in ("East", "West") else half - 0.15
        mb.box((front - 0.15, -fh, 0.0), (front, fh, STAND_Z0), 1, skip=(0,))
        for i in range(ROWS):
            x0, x1 = front + i * ROW_D, front + (i + 1) * ROW_D
            a0, a1 = half + i * ROW_D, half + (i + 1) * ROW_D
            z0, z1 = row_z(i), row_z(i + 1)
            in_vom = VOM_ROWS[0] <= i < VOM_ROWS[1]
            spans = ([(-1, -VOM_HALF), (VOM_HALF, 1)] if in_vom else [(-1, 1)])
            for lo, hi in spans:
                l0 = -a0 if lo == -1 else lo
                l1 = -a1 if lo == -1 else lo
                h0 = a0 if hi == 1 else hi
                h1 = a1 if hi == 1 else hi
                mb.quad((x0, l0, z0), (x1, l1, z0), (x1, h1, z0),
                        (x0, h0, z0), 0)                       # tread
            # riser behind this row
            riser_split = VOM_ROWS[0] <= i < VOM_ROWS[1] - 1
            spans = ([(-a1, -VOM_HALF), (VOM_HALF, a1)] if riser_split
                     else [(-a1, a1)])
            for lo, hi in spans:
                mb.quad((x1, hi, z0), (x1, lo, z0), (x1, lo, z1),
                        (x1, hi, z1), 0)
        # concourse behind the last row
        x0, x1 = front + depth, front + depth + CONCOURSE_W
        a0, a1 = half + depth, half + depth + CONCOURSE_W
        mb.quad((x0, -a0, top_z), (x1, -a1, top_z), (x1, a1, top_z),
                (x0, a0, top_z), 0)
        # aisle half steps
        for ya in aisles_for(half):
            for i in range(ROWS):
                xm = front + i * ROW_D + ROW_D / 2
                mb.box((xm, ya - AISLE_W / 2, row_z(i)),
                       (front + (i + 1) * ROW_D, ya + AISLE_W / 2,
                        row_z(i) + ROW_H / 2), 0, skip=(0, 3))
        # vomitory (entrance tunnel) cut into the stand
        r0, r1 = VOM_ROWS
        vz = row_z(r0)
        vx0, vx1 = front + r0 * ROW_D, front + r1 * ROW_D
        mb.quad((vx0, -VOM_HALF, vz), (vx1, -VOM_HALF, vz),
                (vx1, VOM_HALF, vz), (vx0, VOM_HALF, vz), 0)
        for sgn in (-1, 1):
            ya, yb = sorted((sgn * VOM_HALF, sgn * (VOM_HALF + 0.12)))
            for i in range(r0, r1):
                mb.box((front + i * ROW_D, ya, vz),
                       (front + (i + 1) * ROW_D, yb, row_z(i) + 1.0), 1)
        mb.box((vx1 - 0.12, -VOM_HALF - 0.12, vz),
               (vx1, VOM_HALF + 0.12, row_z(r1) + 1.0), 1)
        xd = vx1 - 0.13
        mb.quad((xd, 0.85, vz), (xd, -0.85, vz), (xd, -0.85, vz + 2.02),
                (xd, 0.85, vz + 2.02), 2)
        mb.to_object("Grandstand_" + name, C)

        # seats (merged: static, one draw call per stand)
        mb = MeshBuilder(["Seat_Plastic", "Steel_Dark"])
        mb.M = M
        for i in range(ROWS):
            xs = front + i * ROW_D + 0.55
            z = row_z(i)
            for y in seat_positions(half, i):
                mb.box((xs - 0.20, y - 0.22, z + 0.39),
                       (xs + 0.22, y + 0.22, z + 0.44), 0)
                mb.hexa(((xs + 0.20, y - 0.22, z + 0.40),
                         (xs + 0.25, y - 0.22, z + 0.40),
                         (xs + 0.25, y + 0.22, z + 0.40),
                         (xs + 0.20, y + 0.22, z + 0.40),
                         (xs + 0.24, y - 0.22, z + 0.86),
                         (xs + 0.285, y - 0.22, z + 0.86),
                         (xs + 0.285, y + 0.22, z + 0.86),
                         (xs + 0.24, y + 0.22, z + 0.86)), 0)
                mb.box((xs + 0.08, y - 0.03, z), (xs + 0.20, y + 0.03,
                                                  z + 0.39), 1, skip=(0, 1))
        mb.to_object("Seats_" + name, C)

        # front rail on top of the fascia wall
        mb = MeshBuilder(["Rail_Metal"])
        mb.M = M
        xr = front - 0.075
        n = int((2 * half - 0.6) / 1.5)
        for k in range(n + 1):
            y = -half + 0.3 + k * (2 * half - 0.6) / n
            mb.box((xr - 0.02, y - 0.02, STAND_Z0), (xr + 0.02, y + 0.02,
                                                     STAND_Z0 + 0.95), 0)
        for zz, t in ((0.95, 0.03), (0.62, 0.012), (0.32, 0.012)):
            mb.box((xr - t, -half + 0.3, STAND_Z0 + zz - t),
                   (xr + t, half - 0.3, STAND_Z0 + zz + t), 0)
        mb.to_object("Railing_" + name, C)

        # aisle step lights (tiny warm markers on every step)
        mb = MeshBuilder(["Aisle_StepLight"])
        mb.M = M
        for ya in aisles_for(half):
            for i in range(ROWS):
                xm = front + i * ROW_D + ROW_D / 2 - 0.004
                zc = row_z(i) + ROW_H / 2 - 0.05
                for sgn in (-1, 1):
                    yc = ya + sgn * (AISLE_W / 2 - 0.1)
                    mb.quad((xm, yc + 0.06, zc - 0.015),
                            (xm, yc - 0.06, zc - 0.015),
                            (xm, yc - 0.06, zc + 0.015),
                            (xm, yc + 0.06, zc + 0.015), 0)
        mb.to_object("AisleStepLights_" + name, C)

    # ---- perimeter walls + columns ---------------------------------------
    mb = MeshBuilder(["Concrete_Wall", "Wall_Panel"])
    t = 0.4
    for sx in (-1, 1):
        x0, x1 = sorted((sx * wall_x, sx * (wall_x + t)))
        mb.box((x0, -wall_y - t, top_z - 0.5), (x1, wall_y + t, wall_top), 0)
    for sy in (-1, 1):
        y0, y1 = sorted((sy * wall_y, sy * (wall_y + t)))
        mb.box((-wall_x, y0, top_z - 0.5), (wall_x, y1, wall_top), 0)
    # acoustic panel band on the upper walls
    band0, band1 = top_z + 3.0, ROOF_Z - 0.6
    for k in range(int(2 * wall_y / 3.0)):
        y = -wall_y + 0.3 + k * 3.0
        for sx in (-1, 1):
            x0, x1 = sorted((sx * (wall_x - 0.06), sx * wall_x))
            mb.box((x0, y, band0), (x1, y + 2.8, band1), 1)
    mb.to_object("Perimeter_Walls", C)

    mb = MeshBuilder(["Steel_Dark"])
    truss_ys = [(-wall_y + 0.6) + k * (2 * wall_y - 1.2) / 9 for k in range(10)]
    for y in truss_ys:
        for sx in (-1, 1):
            mb.cbox((sx * (wall_x - 0.25), y, (top_z + ROOF_Z) / 2),
                    (0.5, 0.5, ROOF_Z - top_z), 0)
    for k in range(8):
        x = -wall_x + 3.0 + k * (2 * wall_x - 6.0) / 7
        if abs(x) < 17.5:
            continue                       # keep the rear-wall displays clear
        for sy in (-1, 1):
            mb.cbox((x, sy * (wall_y - 0.25), (top_z + ROOF_Z) / 2),
                    (0.5, 0.5, ROOF_Z - top_z), 0)
    mb.to_object("Structural_Columns", C)

    # ---- roof: exposed steel trusses, purlins, deck -----------------------
    mb = MeshBuilder(["Steel_Truss"])
    panels = 18
    for y in truss_ys:
        p0, p1 = V((-wall_x, y, ROOF_Z)), V((wall_x, y, ROOF_Z))
        up = V((0, 0, TRUSS_DEPTH))
        mb.beam(p0, p1, 0.26, 0.26)
        mb.beam(p0 + up, p1 + up, 0.26, 0.26)
        for k in range(panels + 1):
            a = p0.lerp(p1, k / panels)
            mb.beam(a, a + up, 0.14, 0.14)
            if k < panels:
                b = p0.lerp(p1, (k + 1) / panels)
                if k % 2 == 0:
                    mb.beam(a, b + up, 0.12, 0.12)
                else:
                    mb.beam(a + up, b, 0.12, 0.12)
    mb.to_object("Roof_Trusses", C)

    mb = MeshBuilder(["Steel_Truss"])
    for k in range(panels + 1):
        x = -wall_x + k * 2 * wall_x / panels
        mb.beam((x, -wall_y, ROOF_Z + TRUSS_DEPTH + 0.2),
                (x, wall_y, ROOF_Z + TRUSS_DEPTH + 0.2), 0.12, 0.2)
    # lower longitudinal ties between trusses
    for x in (-wall_x * 0.5, 0.0, wall_x * 0.5):
        mb.beam((x, -wall_y, ROOF_Z), (x, wall_y, ROOF_Z), 0.14, 0.14)
    mb.to_object("Roof_Purlins", C)

    mb = MeshBuilder(["Roof_Deck"])
    z = ROOF_Z + TRUSS_DEPTH + 0.3
    mb.quad((-wall_x - t, -wall_y - t, z), (-wall_x - t, wall_y + t, z),
            (wall_x + t, wall_y + t, z), (wall_x + t, -wall_y - t, z), 0)
    mb.to_object("Roof_Deck", C)
    return {"wall_x": wall_x, "wall_y": wall_y, "top_z": top_z,
            "truss_ys": truss_ys}


# =============================================================================
# BADMINTON_COURT + COURT_LINES
# =============================================================================
def build_court():
    C = "BADMINTON_COURT"
    mb = MeshBuilder(["Court_Mat"])
    mb.box((-MAT_HALF[0], -MAT_HALF[1], 0.0), (MAT_HALF[0], MAT_HALF[1], MAT_T),
           0, skip=(0,))
    mb.to_object("Court_Mat", C)
    # subtle tournament branding on the run-off behind each base line
    for tag, y in (("South", -(COURT_L / 2 + 1.15)), ("North", COURT_L / 2 + 1.15)):
        add_text("Court_Branding_YONEX_" + tag, "YONEX", 2.6, 0.30,
                 Matrix.Translation((0, y, MAT_T + 0.002)), "Court_Branding", C)
    for tag, x in (("West", -3.85), ("East", 3.85)):
        m = Matrix.Translation((x, -3.2, MAT_T + 0.002)) @ rot_z(
            math.pi / 2 if x > 0 else -math.pi / 2)
        add_text("Court_Branding_BWF_" + tag, "BWF", 1.2, 0.22, m,
                 "Court_Branding", C)


def court_line_rects():
    """name -> (x0, y0, x1, y1) for every painted line."""
    hw, hl, w = COURT_W / 2, COURT_L / 2, LINE_W
    sx = hw - SINGLES_INSET           # outer edge of the singles side line
    ss = SHORT_SERVICE                # near edge of the short service line
    ls = hl - LONG_SERVICE_DBL        # inner edge of the doubles long line
    r = {
        "Line_Sideline_Doubles_West": (-hw, -hl, -hw + w, hl),
        "Line_Sideline_Doubles_East": (hw - w, -hl, hw, hl),
        "Line_Sideline_Singles_West": (-sx, -hl, -sx + w, hl),
        "Line_Sideline_Singles_East": (sx - w, -hl, sx, hl),
        "Line_BackBoundary_South": (-hw, -hl, hw, -hl + w),
        "Line_BackBoundary_North": (-hw, hl - w, hw, hl),
        "Line_LongService_Doubles_South": (-hw, -ls - w, hw, -ls),
        "Line_LongService_Doubles_North": (-hw, ls, hw, ls + w),
        "Line_ShortService_South": (-hw, -ss - w, hw, -ss),
        "Line_ShortService_North": (-hw, ss, hw, ss + w),
        "Line_Centre_South": (-w / 2, -hl, w / 2, -ss),
        "Line_Centre_North": (-w / 2, ss, w / 2, hl),
    }
    return r


def build_lines():
    z = MAT_T + 0.003
    for name, (x0, y0, x1, y1) in court_line_rects().items():
        mb = MeshBuilder(["Court_Line"])
        mb.quad((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z), 0)
        mb.to_object(name, "COURT_LINES")


# =============================================================================
# NET_AND_POSTS
# =============================================================================
def make_net_texture():
    """Tileable RGBA texture of one 19 mm net cell (works in Unity as a
    cut-out / alpha-clipped texture)."""
    size, strand = 64, 5
    img = bpy.data.images.new("Net_Mesh_Cell", size, size, alpha=True)
    px = [0.0] * (size * size * 4)
    col = srgb(48, 34, 26)
    for y in range(size):
        for x in range(size):
            on = x < strand or y < strand
            i = (y * size + x) * 4
            px[i:i + 4] = (col[0], col[1], col[2], 1.0 if on else 0.0)
    img.pixels = px
    try:
        img.pack()
    except Exception:
        pass
    return img


def build_net():
    C = "NET_AND_POSTS"
    hw = COURT_W / 2
    # material with alpha-tested mesh texture
    mat = make_mat("Net_Mesh", srgb(48, 34, 26), 0.9)
    nt = mat.node_tree
    bsdf = _principled(mat)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.location = (-400, 200)
    tex.image = make_net_texture()
    tex.interpolation = "Linear"
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    for attr, val in (("blend_method", "HASHED"), ("shadow_method", "HASHED"),
                      ("surface_render_method", "DITHERED")):
        try:
            setattr(mat, attr, val)
        except Exception:
            pass

    def top(x):
        return NET_H_CENTRE + (NET_H_POST - NET_H_CENTRE) * (x / hw) ** 2

    cell, n = 0.019, 32
    tape = 0.038
    mb = MeshBuilder(["Net_Mesh"])
    mbt = MeshBuilder(["Net_Tape"])
    for k in range(n):
        xa, xb = -hw + 2 * hw * k / n, -hw + 2 * hw * (k + 1) / n
        za, zb = top(xa), top(xb)
        ba, bb = za - NET_DEPTH, zb - NET_DEPTH
        uv = [(xa / cell, ba / cell), (xb / cell, bb / cell),
              (xb / cell, (zb - tape) / cell), (xa / cell, (za - tape) / cell)]
        # single-sided quads: enable two-sided rendering on the Unity material
        mb.quad((xa, 0, ba), (xb, 0, bb), (xb, 0, zb - tape),
                (xa, 0, za - tape), 0, uv=uv)
        mbt.hexa(((xa, -0.006, za - tape), (xb, -0.006, zb - tape),
                  (xb, 0.006, zb - tape), (xa, 0.006, za - tape),
                  (xa, -0.006, za), (xb, -0.006, zb),
                  (xb, 0.006, zb), (xa, 0.006, za)), 0)
        # thin white edging along the bottom
        mbt.hexa(((xa, -0.003, ba), (xb, -0.003, bb), (xb, 0.003, bb),
                  (xa, 0.003, ba), (xa, -0.003, ba + 0.012),
                  (xb, -0.003, bb + 0.012), (xb, 0.003, bb + 0.012),
                  (xa, 0.003, ba + 0.012)), 0)
    for sx in (-1, 1):                      # side edging at the posts
        x0, x1 = sorted((sx * hw, sx * (hw - 0.012)))
        mbt.box((x0, -0.003, NET_H_POST - NET_DEPTH), (x1, 0.003, NET_H_POST), 0)
    mb.to_object("Net_Mesh", C)
    mbt.to_object("Net_Tape", C)

    for tag, sx in (("West", -1), ("East", 1)):
        mb = MeshBuilder(["Post_Blue", "Rubber_Black", "Net_Tape"])
        x = sx * (hw + 0.02)
        mb.tube((x, 0, 0.0), (x, 0, NET_H_POST), 0.02, segs=16, mat=0)
        mb.tube((x, 0, NET_H_POST), (x, 0, NET_H_POST + 0.012), 0.022,
                segs=16, mat=1)
        # weighted base on castors, outside the court
        bx0, bx1 = sorted((x - sx * 0.05, x + sx * 0.62))
        mb.box((bx0, -0.19, 0.03), (bx1, 0.19, 0.16), 0)
        mb.box((bx0 + 0.02, -0.17, 0.16), (bx1 - 0.02, 0.17, 0.175), 1)
        for wx in (bx0 + 0.07, bx1 - 0.07):
            for wy in (-0.15, 0.15):
                mb.tube((wx, wy - 0.015, 0.03), (wx, wy + 0.015, 0.03), 0.03,
                        segs=12, mat=1)
        # brace from base to post
        mb.beam((x + sx * 0.45, 0, 0.16), (x + sx * 0.03, 0, 0.75), 0.03, 0.03, 0)
        mb.to_object("Net_Post_" + tag, C)


# =============================================================================
# LED_ADVERTISING
# =============================================================================
def led_panel(name, brand, length, height, matrix, tilt=LED_TILT, depth=0.10,
              feet=True, flip_accent=False, text_scale=1.0, coll="LED_ADVERTISING"):
    """One LED board.  Local space: length along X, screen faces -Y."""
    mb = MeshBuilder(["LED_Frame", "LED_%s_BG" % brand, "LED_%s_Accent" % brand])
    hl = length / 2
    sh = height * math.tan(tilt)                 # how far the top leans back
    z0 = 0.06 if feet else 0.0
    mb.hexa(((-hl, 0, z0), (hl, 0, z0), (hl, depth, z0), (-hl, depth, z0),
             (-hl, sh, z0 + height), (hl, sh, z0 + height),
             (hl, sh + depth, z0 + height), (-hl, sh + depth, z0 + height)), 0)
    n = V((0, -math.cos(tilt), math.sin(tilt)))
    up = V((0, math.sin(tilt), math.cos(tilt)))
    bez = 0.025

    def P(x, t):                                  # point on the screen plane
        return V((x, 0, z0)) + up * (t / math.cos(tilt)) + n * 0.004

    t0, t1 = bez, height - bez
    xa, xb = -hl + bez, hl - bez
    split = length * 0.24
    slant = (t1 - t0) * 0.45
    if not flip_accent:
        s_lo, s_hi = xb - split - slant, xb - split
        mb.quad(P(xa, t0), P(s_lo, t0), P(s_hi, t1), P(xa, t1), 1)
        mb.quad(P(s_lo, t0), P(xb, t0), P(xb, t1), P(s_hi, t1), 2)
        tx = (xa + s_lo) / 2 + 0.05
    else:
        s_lo, s_hi = xa + split, xa + split + slant
        mb.quad(P(xa, t0), P(s_lo, t0), P(s_hi, t1), P(xa, t1), 2)
        mb.quad(P(s_lo, t0), P(xb, t0), P(xb, t1), P(s_hi, t1), 1)
        tx = (s_hi + xb) / 2 - 0.05
    if feet:
        for x in (-hl + 0.35, hl - 0.35):
            mb.box((x - 0.04, -0.06, 0.0), (x + 0.04, sh + depth + 0.42, 0.06), 0)
            mb.beam((x, sh + depth + 0.38, 0.06),
                    (x, sh * 0.75 + depth, z0 + height * 0.75), 0.04, 0.04, 0)
    ob = mb.to_object(name, coll, matrix)
    fit_w = (length - split - slant) * 0.80
    centre = P(tx, (t0 + t1) / 2) + n * 0.006
    add_text(name + "_Text", brand, fit_w, height * 0.50 * text_scale,
             matrix @ plane_matrix(centre, n), "LED_Text_White", coll, parent=ob)
    return ob


def build_led(info):
    C = "LED_ADVERTISING"
    k = 0
    # side rows (6 boards each), far and near rows (4 boards each)
    rows = [
        ("East", [(LED_X, -7.5 + 3.0 * i) for i in range(6)], -math.pi / 2),
        ("West", [(-LED_X, 7.5 - 3.0 * i) for i in range(6)], math.pi / 2),
        ("North", [(-4.5 + 3.0 * i, LED_Y) for i in range(4)], 0.0),
        ("South", [(4.5 - 3.0 * i, -LED_Y) for i in range(4)], math.pi),
    ]
    for tag, positions, ang in rows:
        for i, (x, y) in enumerate(positions):
            brand = BRAND_ORDER[k % 4]
            led_panel("LED_Board_%s_%02d" % (tag, i + 1), brand, LED_LEN - 0.04,
                      LED_H, Matrix.Translation((x, y, 0)) @ rot_z(ang),
                      flip_accent=(k % 2 == 1))
            k += 1

    # ribbon boards on the stand fascia behind each base line
    for tag, sy, ang in (("North", 1, 0.0), ("South", -1, math.pi)):
        for i in range(5):
            x = (-8.0 + 4.0 * i) * (1 if sy == 1 else -1)
            brand = BRAND_ORDER[(i + (0 if sy == 1 else 2)) % 4]
            led_panel("LED_Fascia_%s_%02d" % (tag, i + 1), brand, 3.9, 0.74,
                      Matrix.Translation((x, sy * (FY - 0.15 - 0.07), 0.13))
                      @ rot_z(ang), tilt=0.0, depth=0.06, feet=False,
                      flip_accent=(i % 2 == 0))

    # rear-wall displays: BWF main screen flanked by sponsor banners
    wy, top_z = info["wall_y"], info["top_z"]
    zc = top_z + 4.9
    for tag, sy, ang in (("North", 1, 0.0), ("South", -1, math.pi)):
        M = Matrix.Translation((0, sy * (wy - 0.02), zc)) @ rot_z(ang)
        W, H = 15.0, 5.0
        mb = MeshBuilder(["LED_Frame", "LED_Screen_Dark", "LED_BWF_Red"])
        mb.box((-W / 2 - 0.15, -0.25, -H / 2 - 0.15), (W / 2 + 0.15, 0.0,
                                                       H / 2 + 0.15), 0)
        y = -0.256
        mb.quad((-W / 2, y, -H / 2), (W / 2, y, -H / 2), (W / 2, y, H / 2),
                (-W / 2, y, H / 2), 1)
        # red slanted block behind the BWF letters
        y2 = -0.262
        mb.quad((-6.3, y2, -1.25), (-1.5, y2, -1.25), (-0.9, y2, 1.25),
                (-5.7, y2, 1.25), 2)
        ob = mb.to_object("RearWall_Screen_" + tag, C, M)
        nrm = (0, -1, 0)
        add_text("RearWall_Screen_%s_BWF" % tag, "BWF", 3.9, 1.5,
                 M @ plane_matrix((-3.6, -0.27, 0.0), nrm), "LED_Text_White",
                 C, parent=ob)
        add_text("RearWall_Screen_%s_Line1" % tag, "BADMINTON", 6.0, 0.78,
                 M @ plane_matrix((3.1, -0.27, 0.62), nrm), "LED_Text_White",
                 C, parent=ob)
        add_text("RearWall_Screen_%s_Line2" % tag, "WORLD FEDERATION", 6.0,
                 0.78, M @ plane_matrix((3.1, -0.27, -0.62), nrm),
                 "LED_Text_White", C, parent=ob)
        for j, (brand, sx) in enumerate((("YONEX", -1), ("VICTOR", 1))):
            Mb = (Matrix.Translation((sx * 13.2 * (1 if sy == 1 else -1),
                                      sy * (wy - 0.25), zc - 1.9)) @ rot_z(ang))
            led_panel("RearWall_Banner_%s_%s" % (tag, brand), brand, 7.2, 3.2,
                      Mb, tilt=0.0, depth=0.2, feet=False,
                      flip_accent=(j == 1), text_scale=0.8)

    # side-wall banners (LI-NING / BWF) above the long stands
    wx = info["wall_x"]
    for tag, sx, ang in (("East", 1, -math.pi / 2), ("West", -1, math.pi / 2)):
        for j, (brand, y) in enumerate((("LI-NING", -9.0), ("BWF", 9.0))):
            Mb = (Matrix.Translation((sx * (wx - 0.75), y * sx, zc - 1.9))
                  @ rot_z(ang))
            led_panel("SideWall_Banner_%s_%s" % (tag, brand), brand, 7.2, 3.2,
                      Mb, tilt=0.0, depth=0.2, feet=False,
                      flip_accent=(j == 0), text_scale=0.8)


# =============================================================================
# PEOPLE (low-poly figures shared by officials and spectators)
# =============================================================================
POSES = {
    # wrist targets (right, left) in figure space, forward lean in radians
    "lap": dict(lean=0.04, R=(0.13, 0.30, 0.615), L=(-0.13, 0.30, 0.615)),
    "lean": dict(lean=0.40, R=(0.05, 0.50, 0.60), L=(-0.05, 0.50, 0.60),
                 pole=(1.0, 0.1, -0.2)),
    "crossed": dict(lean=-0.03, R=(-0.11, 0.135, 0.80), L=(0.11, 0.15, 0.765),
                    pole=(1.0, -0.1, -0.9)),
    "clap": dict(lean=0.08, R=(0.04, 0.33, 0.84), L=(-0.04, 0.33, 0.84)),
    "phone": dict(lean=0.10, R=(0.07, 0.30, 0.92), L=(-0.13, 0.30, 0.615)),
    "cheer": dict(lean=0.0, R=(0.30, 0.12, 1.42), L=(-0.13, 0.30, 0.615),
                  poleR=(1.0, 0.3, 0.2)),
    "chin": dict(lean=0.30, R=(0.05, 0.31, 0.90), L=(-0.10, 0.44, 0.61),
                 pole=(1.0, 0.2, -0.4)),
    "back": dict(lean=-0.14, R=(0.16, 0.22, 0.62), L=(-0.16, 0.22, 0.62)),
    "stand": dict(lean=0.03, R=(0.24, 0.05, 0.88), L=(-0.24, 0.05, 0.88),
                  seated=False),
    "operate": dict(lean=0.08, R=(0.20, 0.36, 1.30), L=(-0.14, 0.36, 1.22),
                    seated=False),
}
SKIN, TOP, BOTTOM, HAIR, SHOE = range(5)


def _ik(shoulder, wrist, l1, l2, pole):
    d = wrist - shoulder
    dist = d.length
    reach = (l1 + l2) * 0.995
    if dist > reach:
        d *= reach / dist
        dist = reach
        wrist = shoulder + d
    n = d / dist
    a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    perp = pole - n * pole.dot(n)
    if perp.length < 1e-5:
        perp = V((1, 0, 0))
    perp.normalize()
    return shoulder + n * a + perp * h, wrist


def _ring(c, u, v, ru, rv, n):
    return [c + u * (ru * math.cos(2 * math.pi * k / n))
            + v * (rv * math.sin(2 * math.pi * k / n)) for k in range(n)]


def build_figure(mb, pose="lap", female=False, hair="short", sleeves="short",
                 shorts=False, segs=8, head_yaw=0.0, knee=0.13, ankle_y=0.42):
    """Adds a low-poly human (about 600 triangles at segs=8) to `mb`.
    Origin: floor under the seat, facing +Y.
    Material slots: skin, top, bottom, hair, shoes."""
    p = POSES[pose]
    seated = p.get("seated", True)
    lean = p["lean"]
    X = V((1, 0, 0))
    spine = V((0, math.sin(lean), math.cos(lean)))
    fwd = V((0, math.cos(lean), -math.sin(lean)))
    pelvis = V((0, -0.04, 0.555)) if seated else V((0, 0, 0.95))
    k = 0.90 if female else 1.0                  # limb thickness
    sw = 0.155 if female else 0.178              # shoulder half width
    ww = 0.128 if female else 0.150
    hw = 0.168 if female else 0.162
    ts = segs + 2
    r = [_ring(pelvis - spine * 0.07, X, fwd, hw, 0.118, ts),
         _ring(pelvis + spine * 0.10, X, fwd, ww, 0.105, ts),
         _ring(pelvis + spine * 0.28, X, fwd, sw * 0.95, 0.118, ts),
         _ring(pelvis + spine * 0.40, X, fwd, sw, 0.105, ts),
         _ring(pelvis + spine * 0.455, X, fwd, 0.105, 0.078, ts),
         _ring(pelvis + spine * 0.495, X, fwd, 0.058, 0.058, ts)]
    mb.loft(r[0:2], BOTTOM, True, False)
    mb.loft(r[1:], TOP, False, True)

    # ---- neck, head, face, hair -------------------------------------------
    mb.tube(pelvis + spine * 0.47, pelvis + spine * 0.565, 0.052, 0.050, segs,
            SKIN, False, False)
    hc = pelvis + spine * 0.648 + fwd * 0.02
    cy, sy = math.cos(head_yaw), math.sin(head_yaw)
    hx = X * cy + fwd * sy
    hy = fwd * cy - X * sy
    basis = (hx, hy, spine)

    def jaw(x, y, z):
        if z < 0:
            q = 1.0 - 0.26 * z * z
            return x * q, y * q + 0.10 * z * z, z
        return x, y, z

    hs, hrg = (8, 6) if segs < 10 else (12, 8)       # head resolution
    mb.ellipsoid(hc, (0.079, 0.096, 0.113), hs, hrg, SKIN, basis, None, jaw)
    # nose and eyes: just enough that a head never reads as a blank egg
    mb.tube(hc + hy * 0.088 + spine * 0.012, hc + hy * 0.110 - spine * 0.026,
            0.010, 0.016, 4, SKIN, False, True)
    for side in (1, -1):
        e = hc + hx * (side * 0.033) + hy * 0.0915 + spine * 0.024
        u, v = hx * 0.011, spine * 0.007
        mb.quad(e - u - v, e - u + v, e + u + v, e + u - v, SHOE)
    hr = (0.084, 0.102, 0.119)
    hcen = hc + spine * 0.003 - hy * 0.004
    if hair == "short":
        mb.ellipsoid(hcen, hr, hs, hrg, HAIR, basis,
                     lambda x, y, z: z - 0.45 * y > 0.12)
    elif hair == "crop":
        mb.ellipsoid(hcen, (0.081, 0.098, 0.116), hs, hrg, HAIR, basis,
                     lambda x, y, z: z - 0.45 * y > 0.28)
    elif hair in ("long", "bun"):
        mb.ellipsoid(hcen, (0.087, 0.105, 0.121), hs, hrg, HAIR, basis,
                     lambda x, y, z: (z - 0.5 * y > 0.10
                                      or (y < 0.25 and z > -0.62)))
        if hair == "long":
            a = hc - hy * 0.035 - spine * 0.05
            b = pelvis + spine * 0.39 - fwd * 0.075
            u = hx
            v = (b - a).normalized().cross(u)
            mb.loft([_ring(a, u, v, 0.086, 0.060, segs),
                     _ring(b, u, v, 0.115, 0.045, segs)], HAIR, False, True)
        else:
            mb.ellipsoid(hc - hy * 0.090 + spine * 0.065, (0.046, 0.046, 0.046),
                         6, 4, HAIR, basis)

    # ---- arms (2-bone IK from shoulder to the pose's wrist target) --------
    l1, l2 = 0.275, 0.245
    for side, key in ((1, "R"), (-1, "L")):
        sh = pelvis + spine * 0.395 + X * (side * (sw - 0.006))
        target = V(p[key])
        pole = V(p.get("pole" + key, p.get("pole", (1.0, -0.25, -0.6))))
        pole.x *= side
        elbow, wrist = _ik(sh, target, l1, l2, pole)
        ua = (elbow - sh).normalized()
        fa = (wrist - elbow).normalized()
        mb.tube(sh - ua * 0.01, elbow + ua * 0.025, 0.058 * k, 0.046 * k, segs,
                TOP, False, True)
        mb.tube(elbow - fa * 0.025, wrist, 0.045 * k, 0.034 * k, segs,
                TOP if sleeves == "long" else SKIN, True, True)
        hu, hv = mb._frame(fa)
        mb.ellipsoid(wrist + fa * 0.038, (0.040, 0.058, 0.030), 6, 4, SKIN,
                     (hv, fa, hu))

    # ---- legs -------------------------------------------------------------
    for side in (1, -1):
        if seated:
            hip = V((side * 0.088, -0.03, 0.540))
            kn = V((side * knee, 0.40, 0.552))
            an = V((side * (knee + 0.005), ankle_y, 0.085))
        else:
            hip = V((side * 0.09, 0.0, 0.92))
            kn = V((side * 0.095, 0.015, 0.50))
            an = V((side * 0.10, 0.0, 0.085))
        th = (kn - hip).normalized()
        shn = (an - kn).normalized()
        mb.tube(hip - th * 0.03, kn + th * 0.04, 0.092 * k, 0.068 * k, segs,
                BOTTOM, False, True)
        mb.tube(kn - shn * 0.04, an, 0.064 * k, 0.046 * k, segs,
                SKIN if shorts else BOTTOM, True, False)
        mb.ellipsoid(an + V((0, 0.062, -0.038)), (0.050, 0.130, 0.047), 6, 4,
                     SHOE)


def soften_terminator(ob):
    """Cycles: avoid blocky self-shadowing on low-poly smooth meshes."""
    for attr in ("shadow_terminator_shading_offset",):
        if hasattr(ob, attr):
            setattr(ob, attr, 0.6)
    cy = getattr(ob, "cycles", None)
    if cy is not None and hasattr(cy, "shadow_terminator_offset"):
        cy.shadow_terminator_offset = 0.6


def figure_mesh(name, mats, **kw):
    mb = MeshBuilder(mats)
    build_figure(mb, **kw)
    return mb.to_mesh(name)


def chair_mesh():
    mb = MeshBuilder(["Chair_Black", "Rail_Metal"])
    mb.box((-0.22, -0.22, 0.40), (0.22, 0.20, 0.445), 0)
    mb.hexa(((-0.21, -0.25, 0.47), (0.21, -0.25, 0.47), (0.21, -0.22, 0.47),
             (-0.21, -0.22, 0.47), (-0.21, -0.31, 0.88), (0.21, -0.31, 0.88),
             (0.21, -0.28, 0.88), (-0.21, -0.28, 0.88)), 0)
    for x in (-0.2, 0.2):
        mb.tube((x, 0.18, 0.0), (x, 0.17, 0.40), 0.011, segs=6, mat=1)
        mb.tube((x, -0.24, 0.0), (x, -0.22, 0.40), 0.011, segs=6, mat=1)
        mb.tube((x, -0.22, 0.40), (x, -0.295, 0.88), 0.011, segs=6, mat=1)
    return mb.to_mesh("Official_Chair")


# =============================================================================
# REFEREE_AND_OFFICIALS
# =============================================================================
def build_umpire_chair(matrix):
    """Blue elevated umpire chair: steps at the back, footrest, backrest."""
    C = "REFEREE_AND_OFFICIALS"
    mb = MeshBuilder(["Umpire_Blue", "Rubber_Black", "Rail_Metal"])
    seat_z, foot_z, hx = 1.55, 1.11, 0.36
    for sx in (-1, 1):
        xa, xb = sorted((sx * hx, sx * (hx + 0.03)))
        # lower trapezoid side panel
        mb.hexa(((xa, -0.82, 0.05), (xb, -0.82, 0.05), (xb, 0.74, 0.05),
                 (xa, 0.74, 0.05), (xa, -0.30, foot_z), (xb, -0.30, foot_z),
                 (xb, 0.60, foot_z), (xa, 0.60, foot_z)), 0)
        # upper panel beside the seat
        mb.hexa(((xa, -0.30, foot_z), (xb, -0.30, foot_z), (xb, 0.26, foot_z),
                 (xa, 0.26, foot_z), (xa, -0.30, seat_z + 0.02),
                 (xb, -0.30, seat_z + 0.02), (xb, 0.22, seat_z + 0.02),
                 (xa, 0.22, seat_z + 0.02)), 0)
        # arm rest on two posts
        mb.box((xa, -0.26, seat_z + 0.23), (xb, 0.20, seat_z + 0.27), 0)
        for y in (-0.22, 0.16):
            mb.box((xa, y - 0.015, seat_z + 0.02), (xb, y + 0.015,
                                                    seat_z + 0.23), 0)
        # feet
        for y in (-0.78, 0.70):
            mb.box((xa - 0.02, y - 0.06, 0.0), (xb + 0.02, y + 0.06, 0.05), 1)
    # slanted front panel (faces the court, carries the court number)
    mb.hexa(((-hx, 0.71, 0.05), (hx, 0.71, 0.05), (hx, 0.74, 0.05),
             (-hx, 0.74, 0.05), (-hx, 0.57, foot_z), (hx, 0.57, foot_z),
             (hx, 0.60, foot_z), (-hx, 0.60, foot_z)), 0)
    # footrest, seat, backrest
    mb.box((-hx, 0.20, foot_z - 0.03), (hx, 0.60, foot_z), 0)
    mb.box((-hx, -0.30, seat_z - 0.05), (hx, 0.22, seat_z), 0)
    mb.box((-hx + 0.02, -0.26, seat_z), (hx - 0.02, 0.20, seat_z + 0.035), 1)
    mb.hexa(((-hx, -0.30, seat_z), (hx, -0.30, seat_z), (hx, -0.26, seat_z),
             (-hx, -0.26, seat_z), (-hx, -0.38, seat_z + 0.58),
             (hx, -0.38, seat_z + 0.58), (hx, -0.34, seat_z + 0.58),
             (-hx, -0.34, seat_z + 0.58)), 0)
    # steps up the back
    for k in range(4):
        z = 0.30 + k * 0.30
        y = -0.82 + (z - 0.05) * (0.52 / (foot_z - 0.05))
        mb.box((-hx, y - 0.02, z - 0.03), (hx, y + 0.20, z), 0)
        mb.box((-hx, y - 0.02, z), (hx, y + 0.20, z + 0.004), 1)
    ob = mb.to_object("Umpire_Chair", C, matrix)
    # court number on the front and both side panels
    nf = V((0, 1.06, 0.14)).normalized()
    add_text("Umpire_Chair_Number_Front", "1", 0.5, 0.42,
             matrix @ plane_matrix(V((0, 0.672, 0.60)) + nf * 0.004, nf),
             "Print_White", C, parent=ob)
    for tag, sx in (("Right", 1), ("Left", -1)):
        add_text("Umpire_Chair_Number_" + tag, "1", 0.5, 0.40,
                 matrix @ plane_matrix((sx * (hx + 0.034), 0.05, 0.62),
                                       (sx, 0, 0)),
                 "Print_White", C, parent=ob)
    return seat_z


def build_scoreboard(name, matrix):
    C = "REFEREE_AND_OFFICIALS"
    W, H, z0 = 3.4, 1.9, 1.15
    mb = MeshBuilder(["LED_Frame", "Score_Panel", "Score_Blue", "Steel_Dark"])
    mb.box((-W / 2 - 0.08, 0.0, z0 - 0.08), (W / 2 + 0.08, 0.16, z0 + H + 0.08), 0)
    y = -0.004
    mb.quad((-W / 2, y, z0), (W / 2, y, z0), (W / 2, y, z0 + H),
            (-W / 2, y, z0 + H), 1)
    y = -0.008
    mb.quad((-W / 2, y, z0 + H - 0.34), (W / 2, y, z0 + H - 0.34),
            (W / 2, y, z0 + H), (-W / 2, y, z0 + H), 2)
    for x in (-1.2, 1.2):
        mb.box((x - 0.06, 0.02, 0.0), (x + 0.06, 0.14, z0 - 0.08), 3)
        mb.box((x - 0.35, -0.35, 0.0), (x + 0.35, 0.5, 0.05), 3)
    ob = mb.to_object(name, C, matrix)
    n = (0, -1, 0)

    def put(tag, body, x, z, w, h, mat):
        add_text("%s_%s" % (name, tag), body, w, h,
                 matrix @ plane_matrix((x, -0.014, z0 + z), n), mat, C,
                 parent=ob)

    put("Title", "MEN'S SINGLES - FINAL", 0.0, H - 0.17, 3.0, 0.15, "Score_White")
    put("HeadGames", "GAMES", 0.55, 1.36, 0.6, 0.09, "Score_White")
    put("HeadPoints", "POINTS", 1.25, 1.36, 0.6, 0.09, "Score_White")
    put("Home", "HOME", -0.95, 1.00, 1.3, 0.30, "Score_White")
    put("HomeGames", "1", 0.55, 1.00, 0.4, 0.34, "Score_Amber")
    put("HomePoints", "21", 1.25, 1.00, 0.6, 0.40, "Score_Amber")
    put("Away", "AWAY", -0.95, 0.42, 1.3, 0.30, "Score_White")
    put("AwayGames", "0", 0.55, 0.42, 0.4, 0.34, "Score_Amber")
    put("AwayPoints", "18", 1.25, 0.42, 0.6, 0.40, "Score_Amber")
    return ob


def build_tv_camera(name, pos, target, operator_mesh):
    C = "REFEREE_AND_OFFICIALS"
    d = V((target[0] - pos[0], target[1] - pos[1]))
    M = Matrix.Translation((pos[0], pos[1], 0)) @ rot_z(facing(d))
    mb = MeshBuilder(["Device_Dark", "Rail_Metal", "Lens_Glass"])
    head = V((0, 0, 1.38))
    for a in (90, 210, 330):
        foot = V((0.55 * math.cos(math.radians(a)),
                  0.55 * math.sin(math.radians(a)), 0))
        mb.tube(foot, head - V((0, 0, 0.12)), 0.018, 0.022, 6, 1)
    mb.tube(head - V((0, 0, 0.14)), head, 0.05, 0.06, 8, 0)
    mb.box((-0.11, -0.28, 1.40), (0.11, 0.24, 1.64), 0)
    mb.tube((0, 0.24, 1.52), (0, 0.50, 1.52), 0.075, 0.085, 12, 0)
    mb.tube((0, 0.50, 1.52), (0, 0.505, 1.52), 0.07, 0.07, 12, 2)
    mb.box((-0.13, -0.40, 1.58), (0.13, -0.28, 1.76), 0)       # viewfinder
    mb.tube((0.11, -0.2, 1.42), (0.34, -0.55, 1.30), 0.012, 0.012, 6, 1)
    mb.to_object(name, C, M)
    op = add_object(name + "_Operator", operator_mesh, C,
                    M @ Matrix.Translation((0, -0.72, 0)))
    soften_terminator(op)
    return op


def build_officials():
    C = "REFEREE_AND_OFFICIALS"
    off_mats = ["Skin_1", "Official_Shirt", "Official_Trousers", "Hair_0",
                "Shoe_Dark"]
    seated = figure_mesh("Official_Seated", off_mats, pose="lap", segs=12,
                         hair="short")
    seated_b = figure_mesh("Official_Seated_B", off_mats, pose="lean", segs=12,
                           hair="crop", head_yaw=0.15)
    seated_f = figure_mesh("Official_Seated_F", off_mats, pose="lap", segs=12,
                           hair="bun", female=True, sleeves="long")
    operator = figure_mesh("Official_Standing_Operator", off_mats,
                           pose="operate", segs=12, hair="short")
    chair = chair_mesh()

    looks = {}

    def person(name, mesh, M, skin="Skin_1", hair="Hair_0"):
        key = (mesh.name, skin, hair)
        if key not in looks:
            me = mesh.copy()
            me.name = "%s_%s_%s" % (mesh.name, skin, hair)
            me.materials[SKIN] = MATS[skin]
            me.materials[HAIR] = MATS[hair]
            looks[key] = me
        ob = add_object(name, looks[key], C, M)
        soften_terminator(ob)
        return ob

    # --- umpire, on the east side of the net, facing the court -------------
    ux = COURT_W / 2 + 1.50
    Mu = Matrix.Translation((ux, 0, 0)) @ rot_z(facing((-1, 0)))
    seat_z = build_umpire_chair(Mu)
    person("Umpire", seated, Mu @ Matrix.Translation((0, -0.06, seat_z - 0.44)),
           "Skin_1")

    # --- service judge, low chair on the opposite side ---------------------
    Ms = Matrix.Translation((-(COURT_W / 2 + 1.25), 0, 0)) @ rot_z(facing((1, 0)))
    add_object("ServiceJudge_Chair", chair, C, Ms)
    person("ServiceJudge", seated_f, Ms, "Skin_1")

    # --- four line judges ---------------------------------------------------
    hw, hl = COURT_W / 2, COURT_L / 2
    spots = [
        ("LineJudge_1_SidelineEast_South", (hw - 0.02, -(hl + 2.6)), (0, 1)),
        ("LineJudge_2_SidelineWest_North", (-hw + 0.02, hl + 2.6), (0, -1)),
        ("LineJudge_3_Baseline_South", (-(hw + 2.3), -hl + 0.02), (1, 0)),
        ("LineJudge_4_Baseline_North", (hw + 2.3, hl - 0.02), (-1, 0)),
    ]
    for i, (name, pos, look) in enumerate(spots):
        M = Matrix.Translation((pos[0], pos[1], 0)) @ rot_z(facing(look))
        add_object(name + "_Chair", chair, C, M)
        person(name, (seated, seated_b, seated_f, seated)[i], M,
               ("Skin_1", "Skin_2", "Skin_0", "Skin_3")[i],
               ("Hair_0", "Hair_1", "Hair_0", "Hair_0")[i])

    # --- tournament control desk (west side, behind the LED boards) --------
    dx = -8.55
    Md = Matrix.Translation((dx, 0, 0)) @ rot_z(facing((1, 0)))
    mb = MeshBuilder(["Table_Cloth", "Device_Dark", "Screen_Glow"])
    mb.box((-2.3, 0.15, 0.0), (2.3, 0.95, 0.76), 0, skip=(0,))
    for x in (-1.5, 0.0, 1.5):                # laptops, screens face the staff
        mb.box((x - 0.17, 0.30, 0.76), (x + 0.17, 0.54, 0.775), 1)
        mb.hexa(((x - 0.17, 0.54, 0.775), (x + 0.17, 0.54, 0.775),
                 (x + 0.17, 0.552, 0.775), (x - 0.17, 0.552, 0.775),
                 (x - 0.17, 0.60, 0.99), (x + 0.17, 0.60, 0.99),
                 (x + 0.17, 0.612, 0.99), (x - 0.17, 0.612, 0.99)), 1)
        mb.quad((x + 0.15, 0.539, 0.79), (x - 0.15, 0.539, 0.79),
                (x - 0.15, 0.596, 0.98), (x + 0.15, 0.596, 0.98), 2)
    desk = mb.to_object("TournamentControl_Desk", C, Md)
    add_text("TournamentControl_Sign", "TOURNAMENT CONTROL", 3.6, 0.2,
             Md @ plane_matrix((0, 0.955, 0.42), (0, 1, 0)), "Print_White", C,
             parent=desk)
    for i, x in enumerate((-1.5, 0.0, 1.5)):
        M = Md @ Matrix.Translation((x, -0.12, 0))
        add_object("TournamentControl_Chair_%d" % (i + 1), chair, C, M)
        person("TournamentControl_Official_%d" % (i + 1),
               (seated_b, seated, seated_f)[i], M,
               ("Skin_2", "Skin_1", "Skin_0")[i])

    # --- scoreboards (diagonal corners, angled to the court) ---------------
    for tag, pos in (("NorthWest", (-8.9, 12.6)), ("NorthEast", (8.9, 12.6))):
        look = V((-pos[0] * 0.35, -pos[1])).normalized()
        M = Matrix.Translation((pos[0], pos[1], 0)) @ rot_z(
            math.atan2(look.y, look.x) + math.pi / 2)
        build_scoreboard("Scoreboard_" + tag, M)

    # --- broadcast camera positions ----------------------------------------
    build_tv_camera("TV_Camera_West", (-8.7, -6.6), (0, 0), operator)
    build_tv_camera("TV_Camera_East", (8.6, 6.4), (0, 0), operator)
    build_tv_camera("TV_Camera_West_Low", (-8.4, 5.4), (0, 2), operator)


# =============================================================================
# SPECTATORS
# =============================================================================
def build_spectators():
    variants = []
    placeholder = ["Skin_1", "Top_00", "Bottom_0", "Hair_0", "Shoe_Dark"]
    spec = [
        # pose, female, hair, sleeves, shorts, head_yaw, knee, ankle, weight
        ("lap", False, "short", "short", False, 0.0, 0.14, 0.42, 14),
        ("lap", True, "long", "short", False, 0.1, 0.10, 0.40, 12),
        ("lap", False, "crop", "long", False, -0.15, 0.17, 0.48, 8),
        ("lean", False, "short", "short", False, 0.0, 0.17, 0.36, 10),
        ("lean", True, "bun", "long", False, 0.2, 0.12, 0.34, 6),
        ("crossed", False, "short", "long", False, 0.1, 0.15, 0.46, 8),
        ("crossed", True, "long", "long", False, -0.2, 0.10, 0.38, 5),
        ("clap", False, "crop", "short", True, 0.0, 0.15, 0.40, 5),
        ("clap", True, "long", "short", False, 0.0, 0.11, 0.40, 5),
        ("phone", True, "bun", "short", False, 0.0, 0.11, 0.36, 5),
        ("phone", False, "short", "long", False, 0.0, 0.14, 0.44, 4),
        ("chin", False, "short", "short", False, -0.1, 0.16, 0.38, 7),
        ("chin", True, "long", "long", False, 0.15, 0.11, 0.36, 4),
        ("back", False, "crop", "short", True, 0.25, 0.18, 0.52, 6),
        ("back", True, "long", "short", False, -0.25, 0.11, 0.46, 5),
        ("cheer", False, "short", "short", False, 0.0, 0.14, 0.42, 2),
    ]
    skins = [MATS["Skin_%d" % i] for i in range(5)]
    skin_w = [30, 34, 20, 10, 6]
    hairs = [MATS["Hair_%d" % i] for i in range(6)]
    hair_w = [46, 22, 12, 6, 9, 5]
    tops = [MATS["Top_%02d" % i] for i in range(16)]
    top_w = [13, 11, 10, 12, 4, 8, 4, 3, 9, 3, 3, 6, 4, 3, 4, 5]
    bottoms = [MATS["Bottom_%d" % i] for i in range(6)]
    bot_w = [28, 26, 14, 10, 14, 8]
    shoe = MATS["Shoe_Dark"]
    orng = random.Random(SEED + 3)
    weights = []
    # Every pose is stored once per outfit (skin / top / bottom / hair colour).
    # Materials live on the mesh data, so the ~2000 spectators are plain
    # instances of these meshes: cheap in Blender, and Unity can GPU-instance.
    for i, (pose, fem, hair, slv, shorts, yaw, knee, ank, w) in enumerate(spec):
        base = figure_mesh("Spectator_%02d_%s" % (i + 1, pose), placeholder,
                           pose=pose, female=fem, hair=hair, sleeves=slv,
                           shorts=shorts, head_yaw=yaw, knee=knee, ankle_y=ank)
        for j in range(SPECTATOR_OUTFITS):
            me = base if j == 0 else base.copy()
            me.name = "Spectator_%02d_%s_%c" % (i + 1, pose, ord("A") + j)
            picks = (orng.choices(skins, skin_w)[0], orng.choices(tops, top_w)[0],
                     orng.choices(bottoms, bot_w)[0],
                     orng.choices(hairs, hair_w)[0], shoe)
            for slot, m in enumerate(picks):     # keep face material indices
                me.materials[slot] = m
            variants.append(me)
            weights.append(w)
    rng = random.Random(SEED + 7)
    occupancy = 0.22 if ARGS["quick"] else OCCUPANCY
    cam = V(MAIN_CAM_POS)
    total = 0
    for name, ang, front, half in SIDES:
        coll = sub_collection("SPECTATORS", "SPECTATORS_" + name)
        M = rot_z(ang)
        k = 0
        for i in range(ROWS):
            x = front + i * ROW_D + 0.55
            z = row_z(i)
            for y in seat_positions(half, i):
                n = noise.noise(V((y * 0.11, i * 0.33, ang * 3.1)))
                if rng.random() > occupancy + 0.16 * n:
                    continue
                world = M @ V((x, y, z))
                # keep the main broadcast camera position clear
                if abs(world.x - cam.x) < 1.6 and abs(world.y - cam.y) < 2.2:
                    continue
                s = rng.uniform(0.94, 1.05)
                if rng.random() < 0.05:
                    s = rng.uniform(0.74, 0.84)            # children
                # turn the body a little towards the court centre
                aim = math.atan2(-world.y, -world.x) - (ang + math.pi)
                aim = (aim + math.pi) % (2 * math.pi) - math.pi
                yaw = ang + math.pi / 2 + 0.35 * aim + rng.gauss(0, 0.09)
                mat = (Matrix.Translation(world + M @ V((0, rng.uniform(-0.03, 0.03), 0)))
                       @ rot_z(yaw) @ Matrix.Scale(s, 4))
                me = rng.choices(variants, weights)[0]
                ob = bpy.data.objects.new("Spec_%s_%04d" % (name[0], k), me)
                coll.objects.link(ob)
                ob.matrix_basis = mat
                soften_terminator(ob)
                k += 1
        total += k
    return total, len(variants)



# =============================================================================
# LIGHTING
# =============================================================================
def add_area_light(name, pos, target, size, power, color, spread=None,
                   glossy=True):
    lt = bpy.data.lights.new(name, "AREA")
    lt.shape = "RECTANGLE"
    lt.size, lt.size_y = size
    lt.energy = power
    lt.color = color
    if spread is not None and hasattr(lt, "spread"):
        lt.spread = spread
    ob = bpy.data.objects.new(name, lt)
    COL["LIGHTING"].objects.link(ob)
    d = V(target) - V(pos)
    ob.matrix_basis = (Matrix.Translation(pos)
                       @ d.to_track_quat("-Z", "Y").to_matrix().to_4x4())
    for attr, val in (("visible_camera", False), ("visible_glossy", glossy)):
        try:
            setattr(ob, attr, val)
        except Exception:
            pass
    return ob


def build_lighting(info):
    """Lighting hierarchy (brightest to dimmest):
         1 court surface   2 net / players   3 LED boards
         4 officials       5 spectator stands
    Court: ten 5400 K floodlight banks flown along both side lines.  Their
    beams are narrowed (`spread`) so they cover the field of play but cut off
    before the opposite stand.  Stands: four low-power warm (3700 K) washes aimed back into the seats."""
    C = "LIGHTING"
    court_col = kelvin_rgb(COURT_LIGHT_K)
    warm_col = kelvin_rgb(AUDIENCE_LIGHT_K)

    # ---- flown lighting rig (box truss ring + floodlight heads) -----------
    mb = MeshBuilder(["Steel_Dark"])
    corners = [V((-RIG_X, -RIG_Y, RIG_Z)), V((RIG_X, -RIG_Y, RIG_Z)),
               V((RIG_X, RIG_Y, RIG_Z)), V((-RIG_X, RIG_Y, RIG_Z))]
    s = 0.15
    for a, b in zip(corners, corners[1:] + corners[:1]):
        axis = (b - a).normalized()
        side = axis.cross(V((0, 0, 1)))
        offs = [side * s + V((0, 0, s)), side * -s + V((0, 0, s)),
                side * -s + V((0, 0, -s)), side * s + V((0, 0, -s))]
        for o in offs:
            mb.tube(a + o - axis * s, b + o + axis * s, 0.025, segs=6)
        n = int((b - a).length / 0.6)
        for k in range(n):
            p, q = a.lerp(b, k / n), a.lerp(b, (k + 1) / n)
            lo, hi = (0, 3) if k % 2 == 0 else (3, 0)
            mb.tube(p + offs[lo], q + offs[hi], 0.012, segs=4, cap0=False,
                    cap1=False)
            mb.tube(p + offs[1 if k % 2 == 0 else 2],
                    q + offs[2 if k % 2 == 0 else 1], 0.012, segs=4,
                    cap0=False, cap1=False)
    for c in corners + [V((-RIG_X, 0, RIG_Z)), V((RIG_X, 0, RIG_Z))]:
        mb.tube(c + V((0, 0, s)), V((c.x, c.y, ROOF_Z)), 0.008, segs=4,
                cap0=False, cap1=False)
    mb.to_object("LightingRig_Truss", C)

    mb = MeshBuilder(["Fixture_Housing", "Fixture_Lens"])
    tilt = math.atan2(RIG_X - 1.0, RIG_Z)       # heads aim at x = +/-1.0
    for sx in (-1, 1):
        for k in range(11):
            y = -10.0 + 2.0 * k
            c = V((sx * RIG_X, y, RIG_Z - 0.36))
            down = V((-sx * math.sin(tilt), 0, -math.cos(tilt)))
            ax_u = V((0, 1, 0))
            ax_v = (-down).cross(ax_u)
            mb.obox(c, ax_u, ax_v, -down, 0.30, 0.19, 0.07, 0)
            f = c + down * 0.074
            q = [f + ax_u * du + ax_v * dv for du, dv in
                 ((-0.27, -0.16), (-0.27, 0.16), (0.27, 0.16), (0.27, -0.16))]
            mb.quad(q[0], q[1], q[2], q[3], 1)
            mb.beam(c - down * 0.07, V((sx * RIG_X, y, RIG_Z - 0.15)), 0.04,
                    0.04, 0)
    mb.to_object("CourtLight_Fixtures", C)

    # ---- court floodlights (the actual light sources) ---------------------
    for tag, sx in (("West", -1), ("East", 1)):
        for k in range(5):
            y = -8.0 + 4.0 * k
            ob = add_area_light("CourtLight_%s_%d" % (tag, k + 1),
                                (sx * RIG_X, y, RIG_Z - 0.5),
                                (sx * 1.0, y * 0.93, 0.0), (3.0, 0.6),
                                COURT_LIGHT_W, court_col,
                                spread=math.radians(72))
            ob["color_temperature_K"] = COURT_LIGHT_K

    # ---- audience washes + visible house-light fittings -------------------
    depth = ROWS * ROW_D
    for name, ang, front, half in SIDES:
        M = rot_z(ang)
        # The wash hangs over the front edge of the stand and is aimed back
        # into it, so faces are lit from the court side and very little
        # spills onto the playing surface.
        pos = M @ V((front - 1.0, 0, ROOF_Z - 1.0))
        tgt = M @ V((front + depth * 0.55, 0, 3.0))
        lt = add_area_light("AudienceLight_" + name, pos, tgt,
                            (2 * half + depth * 1.2, depth * 0.5),
                            AUDIENCE_LIGHT_W * (half + depth * 0.5) / 20.0,
                            warm_col, spread=math.radians(110), glossy=False)
        lt["color_temperature_K"] = AUDIENCE_LIGHT_K
        zax = (pos - tgt).normalized()            # light shines along -Z
        xax = (M.to_3x3() @ V((0, 1, 0))).normalized()   # long side
        yax = zax.cross(xax)
        lt.matrix_basis = Matrix(((xax.x, yax.x, zax.x, pos.x),
                                  (xax.y, yax.y, zax.y, pos.y),
                                  (xax.z, yax.z, zax.z, pos.z),
                                  (0, 0, 0, 1)))
    # ---- officials / technical zones: soft neutral fill, ~1/3 of the court -
    for tag, sx in (("West", -1), ("East", 1)):
        ob = add_area_light("OfficialsLight_" + tag, (sx * 8.5, 0, 9.0),
                            (sx * 8.5, 0, 0), (1.2, 16.0), OFFICIALS_LIGHT_W,
                            court_col, spread=math.radians(60), glossy=False)
        ob["color_temperature_K"] = COURT_LIGHT_K
    # ---- faint up-light so the roof steelwork and upper walls stay readable -
    for tag, sx in (("West", -1), ("East", 1)):
        pos = V((sx * 15.5, 0, 9.5))
        ob = add_area_light("RoofWash_" + tag, pos, pos + V((0, 0, 1)),
                            (3.0, 40.0), ROOF_WASH_W, kelvin_rgb(4300),
                            glossy=False)
        ob.matrix_basis = (Matrix.Translation(pos)
                           @ Matrix.Rotation(math.pi, 4, "X"))
    mb = MeshBuilder(["Fixture_Housing", "HouseLight_Lens"])
    z = ROOF_Z - 0.16
    for y in info["truss_ys"]:
        for x in range(-20, 21, 5):
            if abs(x) < FX - 1 and abs(y) < FY - 1:
                continue                         # none above the court itself
            mb.box((x - 0.3, y - 0.2, z), (x + 0.3, y + 0.2, z + 0.1), 0)
            mb.quad((x - 0.25, y + 0.15, z - 0.003), (x + 0.25, y + 0.15, z - 0.003),
                    (x + 0.25, y - 0.15, z - 0.003), (x - 0.25, y - 0.15, z - 0.003), 1)
    mb.to_object("HouseLight_Fixtures", C)

    # ---- world: almost black, a trace of cool ambient ----------------------
    world = bpy.context.scene.world or bpy.data.worlds.new("Arena_World")
    bpy.context.scene.world = world
    try:
        world.use_nodes = True
    except Exception:
        pass
    bg = next((n for n in world.node_tree.nodes
               if n.bl_idname == "ShaderNodeBackground"), None)
    if bg:
        bg.inputs[0].default_value = (0.012, 0.014, 0.02, 1.0)
        bg.inputs[1].default_value = 0.25


# =============================================================================
# CAMERAS
# =============================================================================
def add_camera(name, pos, target, lens, fstop=None, focus=None):
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    cam.sensor_width = 36.0
    cam.clip_start, cam.clip_end = 0.1, 300.0
    if fstop:
        cam.dof.use_dof = True
        cam.dof.aperture_fstop = fstop
        cam.dof.focus_distance = (V(focus if focus else target) - V(pos)).length
    ob = bpy.data.objects.new(name, cam)
    COL["CAMERAS"].objects.link(ob)
    d = V(target) - V(pos)
    ob.matrix_basis = (Matrix.Translation(pos)
                       @ d.to_track_quat("-Z", "Y").to_matrix().to_4x4())
    return ob


def build_cameras():
    main = add_camera("CAM_Broadcast_Main", MAIN_CAM_POS, (0.0, 5.0, 1.0), 20.0)
    add_camera("CAM_Courtside_Gameplay", (4.75, -9.3, 1.55), (-1.2, 2.0, 1.25),
               26.0, fstop=5.6, focus=(0, 0, 1.4))
    add_camera("CAM_Umpire_Chair", (-0.6, -5.2, 1.75), (4.5, 0.6, 1.45), 35.0,
               fstop=4.0, focus=(4.4, 0, 1.5))
    add_camera("CAM_LED_Detail", (-2.6, -7.9, 1.0), (-6.9, -1.8, 0.75), 30.0,
               fstop=4.0, focus=(-6.6, -3.0, 0.5))
    add_camera("CAM_Audience", (2.0, -3.0, 2.2), (17.0, 2.0, 4.2), 60.0,
               fstop=5.6)
    top = add_camera("CAM_TopDown_Layout", (0, 0, 9.5), (0, 0, 0), 50.0)
    top.data.type = "ORTHO"
    top.data.ortho_scale = 24.0
    top.matrix_basis = Matrix.Translation((0, 0, 9.5)) @ rot_z(-math.pi / 2)
    bpy.context.scene.camera = main
    return main


# =============================================================================
# RENDER SETTINGS
# =============================================================================
def setup_compositor(scene):
    """Gentle bloom on the LED boards and light fittings (kept subtle so the
    court is never washed out)."""
    try:
        if hasattr(scene, "compositing_node_group"):          # Blender 5.x
            tree = bpy.data.node_groups.new("Arena_Compositor",
                                            "CompositorNodeTree")
            scene.compositing_node_group = tree
            tree.interface.new_socket("Image", in_out="OUTPUT",
                                      socket_type="NodeSocketColor")
            out = tree.nodes.new("NodeGroupOutput")
        else:                                                 # Blender 4.x
            scene.use_nodes = True
            tree = scene.node_tree
            tree.nodes.clear()
            out = tree.nodes.new("CompositorNodeComposite")
        rl = tree.nodes.new("CompositorNodeRLayers")
        gl = tree.nodes.new("CompositorNodeGlare")
        rl.location, gl.location, out.location = (-300, 0), (0, 0), (300, 0)
        if "Type" in gl.inputs:
            gl.inputs["Type"].default_value = "Bloom"
        elif hasattr(gl, "glare_type"):
            gl.glare_type = "BLOOM"
        for key, val in (("Threshold", 2.5), ("Strength", 0.35), ("Size", 0.35),
                         ("Smoothness", 0.5), ("Saturation", 0.9)):
            if key in gl.inputs:
                gl.inputs[key].default_value = val
        for attr, val in (("threshold", 2.5), ("mix", -0.6), ("size", 6),
                          ("quality", "HIGH")):
            if hasattr(gl, attr):
                try:
                    setattr(gl, attr, val)
                except Exception:
                    pass
        if "Quality" in gl.inputs:
            try:
                gl.inputs["Quality"].default_value = "High"
            except Exception:
                pass
        tree.links.new(rl.outputs["Image"], gl.inputs["Image"])
        tree.links.new(gl.outputs["Image"], out.inputs[0])
        return True
    except Exception as exc:                                  # non-critical
        print("compositor bloom skipped:", exc)
        return False


def setup_render():
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    r = scene.render
    r.resolution_x, r.resolution_y = RESOLUTION
    r.resolution_percentage = int(ARGS["scale"] or 100)
    r.image_settings.file_format = "PNG"
    r.film_transparent = False
    engine = (ARGS["engine"] or RENDER_ENGINE).upper()
    if engine == "CYCLES":
        r.engine = "CYCLES"
        cy = scene.cycles
        cy.samples = int(ARGS["samples"] or SAMPLES)
        cy.use_adaptive_sampling = True
        cy.adaptive_threshold = 0.02
        cy.max_bounces = 6
        cy.diffuse_bounces = 3
        cy.glossy_bounces = 3
        cy.transparent_max_bounces = 8
        cy.sample_clamp_indirect = 6.0
        cy.caustics_reflective = False
        cy.caustics_refractive = False
        try:
            cy.use_denoising = True
            cy.denoiser = "OPENIMAGEDENOISE"
        except Exception:
            pass
    else:
        for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                r.engine = name
                break
            except Exception:
                continue
        ee = scene.eevee
        for attr, val in (("taa_render_samples", 64), ("use_bloom", True),
                          ("bloom_intensity", 0.03), ("use_gtao", True),
                          ("use_ssr", True), ("use_raytracing", True),
                          ("use_soft_shadows", True)):
            if hasattr(ee, attr):
                try:
                    setattr(ee, attr, val)
                except Exception:
                    pass
    vs = scene.view_settings
    for vt in ("AgX", "Filmic", "Standard"):
        try:
            vs.view_transform = vt
            break
        except Exception:
            continue
    for look in ("AgX - Medium High Contrast", "Medium High Contrast"):
        try:
            vs.look = look
            break
        except Exception:
            continue
    vs.exposure = 0.0
    setup_compositor(scene)
    if bpy.app.background and hasattr(r, "compositor_device"):
        r.compositor_device = "CPU"       # headless machines often lack a GPU


# =============================================================================
# OUTPUT: render / save / FBX
# =============================================================================
def render_cameras(out_dir, names=None):
    scene = bpy.context.scene
    os.makedirs(out_dir, exist_ok=True)
    for ob in COL["CAMERAS"].objects:
        if names and ob.name not in names:
            continue
        if not names and ob.name == "CAM_TopDown_Layout":
            continue
        scene.camera = ob
        scene.render.filepath = os.path.join(out_dir, ob.name + ".png")
        bpy.ops.render.render(write_still=True)
        print("rendered", scene.render.filepath)
    scene.camera = bpy.data.objects.get("CAM_Broadcast_Main")


def export_fbx(out_dir):
    """One FBX per collection, Unity axes (Y up, Z forward), metres."""
    os.makedirs(out_dir, exist_ok=True)
    if not hasattr(bpy.types, "EXPORT_SCENE_OT_fbx"):
        try:
            import addon_utils
            addon_utils.enable("io_scene_fbx")
        except Exception as exc:
            print("FBX add-on unavailable:", exc)
            return
    view_layer = bpy.context.view_layer
    for name in COLLECTION_NAMES:
        if name in ("LIGHTING", "CAMERAS"):
            objs = [o for o in COL[name].all_objects if o.type == "MESH"]
        else:
            objs = [o for o in COL[name].all_objects
                    if o.type in {"MESH", "EMPTY"}]
        if not objs:
            continue
        for o in bpy.data.objects:
            o.select_set(False)
        for o in objs:
            o.select_set(True)
        view_layer.objects.active = objs[0]
        path = os.path.join(out_dir, name + ".fbx")
        bpy.ops.export_scene.fbx(
            filepath=path, use_selection=True, object_types={"MESH", "EMPTY"},
            apply_scale_options="FBX_SCALE_UNITS", axis_forward="-Z",
            axis_up="Y", bake_space_transform=True, mesh_smooth_type="FACE",
            use_mesh_modifiers=False, add_leaf_bones=False, bake_anim=False,
            path_mode="COPY", embed_textures=True)
        print("exported", path)


# =============================================================================
# MAIN
# =============================================================================
def main():
    if CLEAR_SCENE:
        clear_scene()
    make_collections()
    build_materials()
    info = build_structure()
    build_court()
    build_lines()
    build_net()
    build_led(info)
    build_officials()
    n_spec, n_var = build_spectators()
    build_lighting(info)
    build_cameras()
    setup_render()
    bpy.context.view_layer.update()
    tris = 0
    for ob in bpy.data.objects:
        if ob.type == "MESH":
            tris += sum(len(p.vertices) - 2 for p in ob.data.polygons)
    print("Badminton arena built: %d objects, %d spectators (%d shared meshes),"
          " %d triangles in view" % (len(bpy.data.objects), n_spec, n_var, tris))
    if ARGS["save"]:
        path = os.path.abspath(ARGS["save"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
        print("saved", path)
    if ARGS["export_fbx"]:
        export_fbx(os.path.abspath(ARGS["export_fbx"]))
    if ARGS["render"]:
        cams = ARGS["cams"].split(",") if ARGS["cams"] else None
        render_cameras(os.path.abspath(ARGS["render"]), cams)


if __name__ == "__main__":
    main()
