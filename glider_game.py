"""Original one-button glider game for nanoOasis.

Hold SPACE/UP to dive and build speed on a slope; release to launch into a glide.
All visuals are procedurally drawn and original. No external assets are required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

W, H = 960, 540
FPS = 60
GRAVITY = 0.24
DIVE_ACCEL = 0.42
AIR_DRAG = 0.992
MAX_SPEED = 14.0
GROUND_Y = 365

# Original palette: deep dusk sky, citrus hills, a warm coral glider.
SKY_TOP = (18, 27, 61)
SKY_BOTTOM = (107, 185, 205)
HILL_FAR = (42, 92, 126)
HILL_MID = (42, 130, 123)
HILL_NEAR = (63, 160, 106)
GROUND = (92, 174, 92)
GROUND_DARK = (39, 112, 76)
INK = (18, 32, 49)
CREAM = (255, 244, 194)
CORAL = (248, 105, 91)
GOLD = (255, 205, 83)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def lerp(a, b, t):
    return int(a + (b - a) * t)


class Terrain:
    """Low-frequency control points sampled into a smooth, playable hill curve."""
    def __init__(self, seed: int = 7, control_step: int = 72):
        self.rng = np.random.default_rng(seed)
        self.control_step = control_step
        self.control: list[float] = [390.0, 372.0, 410.0, 335.0, 430.0, 360.0]
        self._extend(20)

    def _extend(self, n: int):
        for _ in range(n):
            prev = self.control[-1]
            # Broad hills: each control point moves only a little from its predecessor.
            target = prev + float(self.rng.normal(0, 42))
            self.control.append(float(clamp(target, 270, 455)))

    def height(self, x: float) -> float:
        if x / self.control_step > len(self.control) - 4:
            self._extend(20)
        u = max(0.0, x / self.control_step)
        i = min(int(u), len(self.control) - 2)
        t = u - i
        # Catmull-Rom interpolation gives a smooth curve through broad hills.
        p0 = self.control[max(0, i - 1)]
        p1 = self.control[i]
        p2 = self.control[i + 1]
        p3 = self.control[min(len(self.control) - 1, i + 2)]
        t2, t3 = t * t, t * t * t
        return 0.5 * ((2 * p1) + (-p0 + p2) * t +
                      (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                      (-p0 + 3 * p1 - 3 * p2 + p3) * t3)

    def points(self, start: float, end: float, scale: float = 1.0) -> list[tuple[int, int]]:
        # Dense sampling makes the rendered silhouette visibly smooth.
        xs = range(int(start // 4) * 4, int(end) + 8, 4)
        return [(int(x - start), int(GROUND_Y + (self.height(x) - GROUND_Y) * scale)) for x in xs]


@dataclass
class Glider:
    x: float = 220.0
    y: float = 250.0
    vx: float = 6.0
    vy: float = 0.0
    angle: float = 0.0
    diving: bool = False
    alive: bool = True


class GliderGame:
    """Headless-friendly game state; render() returns an RGB uint8 frame."""
    def __init__(self, seed: int = 7):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.terrain = Terrain(seed)
        self.player = Glider()
        self.camera_x = 0.0
        self.distance = 0.0
        self.best = 0.0
        self.combo = 0
        self.stars = self._make_stars()
        self.particles: list[list[float]] = []
        self.time = 0.0

    def _make_stars(self):
        return [(float(self.rng.integers(0, 10000)), float(self.rng.integers(35, 220)), float(self.rng.uniform(.4, 1.5))) for _ in range(90)]

    def reset(self):
        best = self.best
        self.__init__(self.seed + 1)
        self.best = best

    def step(self, dive: bool = False) -> np.ndarray:
        p = self.player
        if not p.alive:
            return self.render()
        self.time += 1 / FPS
        p.diving = bool(dive)
        if p.diving:
            p.vy += DIVE_ACCEL
            p.angle = min(0.72, p.angle + 0.055)
        else:
            p.vy += GRAVITY
            p.angle *= 0.94
        p.vx = clamp(p.vx * AIR_DRAG + (0.045 if not p.diving else 0.08), 3.0, MAX_SPEED)
        p.vy = clamp(p.vy, -10.0, 11.0)
        p.x += p.vx
        p.y += p.vy
        self.camera_x = max(0.0, p.x - 235)
        ground = self.terrain.height(p.x)
        if p.y + 16 >= ground:
            p.y = ground - 16
            # Diving into the ground is a crash; a shallow, released contact is a bounce.
            # This makes the control rule legible: release before the belly reaches a slope.
            hard_landing = p.diving or p.vy > 4.8 or abs(p.angle) > 0.68
            if hard_landing:
                p.alive = False
                self.best = max(self.best, self.distance)
                self._burst(p.x, p.y, n=24)
            else:
                p.vy = -min(8.5, abs(p.vy) * 0.78 + 2.2)
                p.vx = min(MAX_SPEED, p.vx + 0.45)
                self.combo += 1
                self._burst(p.x, p.y, n=5)
        self.distance = max(self.distance, (p.x - 220) / 10)
        self.best = max(self.best, self.distance)
        for q in self.particles:
            q[0] += q[2]; q[1] += q[3]; q[3] += 0.12; q[4] -= 0.025
        self.particles = [q for q in self.particles if q[4] > 0]
        return self.render()

    def _burst(self, x, y, n=12):
        for _ in range(n):
            a = float(self.rng.uniform(-math.pi, math.pi)); s = float(self.rng.uniform(1, 4))
            self.particles.append([x, y, math.cos(a) * s, math.sin(a) * s - 1, 1.0])

    def render(self) -> np.ndarray:
        # Drawing to a pygame surface keeps the same RGB frame contract as Breakout.
        import pygame
        surf = pygame.Surface((W, H))
        for y in range(H):
            t = y / H
            surf.fill(tuple(lerp(SKY_TOP[c], SKY_BOTTOM[c], t) for c in range(3)), (0, y, W, 1))
        # stars and a softly glowing sun
        for sx, sy, r in self.stars:
            x = int(sx - self.camera_x * 0.12) % W
            pygame.draw.circle(surf, (220, 234, 220), (x, int(sy)), max(1, int(r)))
        pygame.draw.circle(surf, (255, 218, 142), (W - 130, 105), 48)
        pygame.draw.circle(surf, (255, 235, 169), (W - 130, 105), 38)
        for color, parallax, scale in ((HILL_FAR, .16, .62), (HILL_MID, .31, .78), (HILL_NEAR, .55, .92)):
            pts = self.terrain.points(self.camera_x * parallax, self.camera_x * parallax + W / parallax, scale)
            poly = [(x, y) for x, y in pts] + [(W, H), (0, H)]
            pygame.draw.polygon(surf, color, poly)
        pts = self.terrain.points(self.camera_x, self.camera_x + W)
        pygame.draw.polygon(surf, GROUND, pts + [(W, H), (0, H)])
        # ground contour and small flowers/markers
        pygame.draw.lines(surf, GROUND_DARK, False, pts, 3)
        for x, y in pts[::7]:
            if 0 < x < W:
                pygame.draw.circle(surf, (244, 223, 123), (x, y - 4), 2)
        for q in self.particles:
            x, y, _, _, alpha = q
            px, py = int(x - self.camera_x), int(y)
            pygame.draw.circle(surf, GOLD if alpha > .4 else CORAL, (px, py), max(1, int(3 * alpha)))
        self._draw_glider(surf)
        self._draw_hud(surf)
        return pygame.surfarray.array3d(surf).transpose(1, 0, 2).copy()

    def _draw_glider(self, surf):
        p = self.player; x, y = int(p.x - self.camera_x), int(p.y)
        # hand-drawn robot glider: coral shell, cream visor, golden wing.
        body = [(x - 18, y + 4), (x - 3, y - 13), (x + 16, y - 8), (x + 20, y + 7), (x + 3, y + 14)]
        pygame = __import__('pygame')
        pygame.draw.polygon(surf, CORAL if p.alive else (112, 80, 92), body)
        pygame.draw.polygon(surf, GOLD, [(x - 14, y + 4), (x - 32, y + 11), (x - 18, y - 3)])
        pygame.draw.ellipse(surf, CREAM, (x - 4, y - 8, 13, 9))
        pygame.draw.circle(surf, INK, (x + 4, y - 4), 2)
        pygame.draw.line(surf, INK, (x + 12, y + 8), (x + 24, y + 13), 2)

    def _draw_hud(self, surf):
        import pygame
        font = pygame.font.Font(None, 28); small = pygame.font.Font(None, 21)
        title = font.render('SKYSEED', True, CREAM)
        surf.blit(title, (24, 20))
        score = font.render(f'{int(self.distance):04d} m', True, CREAM)
        surf.blit(score, (W - score.get_width() - 24, 20))
        hint = small.render('HOLD SPACE TO DIVE  ·  RELEASE TO GLIDE', True, (218, 231, 226))
        surf.blit(hint, ((W - hint.get_width()) // 2, H - 32))
        if not self.player.alive:
            msg = font.render('CRASHED  ·  PRESS R TO RETRY', True, CREAM)
            surf.blit(msg, ((W - msg.get_width()) // 2, 80))


def main():
    import pygame
    pygame.init(); screen = pygame.display.set_mode((W, H)); pygame.display.set_caption('Skyseed — nanoOasis')
    clock = pygame.time.Clock(); game = GliderGame(); diving = False; running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_SPACE, pygame.K_UP): diving = True
                if ev.key == pygame.K_r and not game.player.alive: game.reset()
                if ev.key == pygame.K_ESCAPE: running = False
            elif ev.type == pygame.KEYUP and ev.key in (pygame.K_SPACE, pygame.K_UP): diving = False
        keys = pygame.key.get_pressed(); diving = diving or bool(keys[pygame.K_SPACE] or keys[pygame.K_UP])
        frame = game.step(diving); surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2)); screen.blit(surf, (0, 0)); pygame.display.flip(); clock.tick(FPS)
    pygame.quit()


if __name__ == '__main__': main()
