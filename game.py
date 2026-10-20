"""Small, headless-friendly Breakout environment used for data generation."""
import numpy as np
W = H = 192
GRID_COLS = GRID_ROWS = 12
CELL = W // GRID_COLS
NUM_ACTIONS = 3
LEFT, NONE, RIGHT = 0, 1, 2
UP, DOWN = 3, 4
DIRS = ((-1, 0), (0, 0), (1, 0))
BG_COLOR = (10, 14, 24)
BODY_COLOR = (232, 238, 252)
HEAD_COLOR = (248, 250, 252)
APPLE_COLOR = (249, 115, 22)

class Game:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed); self.seed=int(seed); self.eats=0; self.deaths=0; self._reset_level()
    def _reset_level(self):
        self.paddle_x=W/2; self.paddle_w=42; self.paddle_h=6; self.ball_x=W/2; self.ball_y=H-42; self.ball_r=4
        self.vx=float(self.rng.choice([-3.4,3.4])); self.vy=-5.0; self.lives=3; self.score=0
        self.bricks={(c,r) for r in range(3) for c in range(2,10)}
    def _serve(self):
        self.ball_x,self.ball_y=self.paddle_x,H-42; self.vx=float(self.rng.choice([-3.4,3.4])); self.vy=-5.0
    def step(self, action):
        if int(action) not in (LEFT, NONE, RIGHT):
            raise ValueError("action must be LEFT, NONE or RIGHT")
        self.last_event = False
        move=int(action); self.paddle_x += -9 if move==LEFT else 9 if move==RIGHT else 0
        self.paddle_x=float(np.clip(self.paddle_x,self.paddle_w/2,W-self.paddle_w/2)); old_y=self.ball_y; old_x=self.ball_x
        self.ball_x+=self.vx; self.ball_y+=self.vy
        if self.ball_x<=self.ball_r or self.ball_x>=W-self.ball_r: self.vx*=-1; self.ball_x=float(np.clip(self.ball_x,self.ball_r,W-self.ball_r))
        if self.ball_y<=self.ball_r: self.vy*=-1; self.ball_y=self.ball_r
        py=H-24
        if self.vy>0 and self.ball_y+self.ball_r>=py and old_y+self.ball_r<=py+self.paddle_h and abs(self.ball_x-self.paddle_x)<=self.paddle_w/2+self.ball_r:
            rel=(self.ball_x-self.paddle_x)/(self.paddle_w/2); self.vx=float(np.clip(rel*6.0,-6.0,6.0)); self.vy=-min(abs(self.vy)*1.05, 6.0); self.ball_y=py-self.ball_r; self.last_event=True
        hit=None
        for c,r in sorted(self.bricks):
            bx,by=c*CELL+1,18+r*11
            if self.ball_x+self.ball_r>=bx and self.ball_x-self.ball_r<=bx+CELL-2 and self.ball_y+self.ball_r>=by and self.ball_y-self.ball_r<=by+8:
                hit=(c,r)
                if old_y+self.ball_r<=by or old_y-self.ball_r>=by+8:
                    self.vy *= -1
                else:
                    self.vx *= -1
                break
        if hit is not None: self.bricks.remove(hit); self.score+=10; self.eats+=1; self.last_event=True
        done=False
        if self.ball_y>H+self.ball_r:
            self.deaths+=1; self.lives-=1; self.last_event=True
            if self.lives<=0: done=True; self._reset_level()
            else: self._serve()
        elif not self.bricks: done=True; self._reset_level()
        return self.render(),float(hit is not None),done
    def render(self):
        fb=np.full((H,W,3),BG_COLOR,dtype=np.uint8)
        for c,r in self.bricks:
            x0,y0=c*CELL+1,18+r*11; color=((248,113,113),(251,146,60),(250,204,21))[r]
            fb[y0:y0+8,x0:x0+CELL-2]=color
        py=H-24; x0=int(self.paddle_x-self.paddle_w/2); fb[py:py+self.paddle_h,x0:x0+self.paddle_w]=BODY_COLOR
        yy,xx=int(self.ball_y),int(self.ball_x)
        for dy in range(-self.ball_r, self.ball_r + 1):
            for dx in range(-self.ball_r, self.ball_r + 1):
                if dx * dx + dy * dy <= self.ball_r * self.ball_r:
                    y, x = yy + dy, xx + dx
                    if 0 <= y < H and 0 <= x < W:
                        fb[y, x] = HEAD_COLOR
        return fb

def safe_actions(g): return [LEFT,NONE,RIGHT]
def _poll_action(pygame,events,current):
    action=current; quit_req=False
    for ev in events:
        if ev.type==pygame.QUIT or (ev.type==pygame.KEYDOWN and ev.key==pygame.K_ESCAPE): quit_req=True
        elif ev.type==pygame.KEYDOWN: action={pygame.K_LEFT:LEFT,pygame.K_RIGHT:RIGHT,pygame.K_a:LEFT,pygame.K_d:RIGHT}.get(ev.key,action)
    keys=pygame.key.get_pressed()
    action=LEFT if keys[pygame.K_LEFT] or keys[pygame.K_a] else RIGHT if keys[pygame.K_RIGHT] or keys[pygame.K_d] else NONE
    return action,quit_req
if __name__=='__main__':
    import pygame
    pygame.init(); screen=pygame.display.set_mode((W*3,H*3)); clock=pygame.time.Clock(); g=Game(); action=NONE
    while True:
        action,quit_req=_poll_action(pygame,pygame.event.get(),action)
        if quit_req: break
        frame,_,_ = g.step(action); surf=pygame.surfarray.make_surface(np.transpose(frame,(1,0,2))); screen.blit(pygame.transform.scale(surf,(W*3,H*3)),(0,0)); pygame.display.flip(); clock.tick(24)
    pygame.quit()
