"""Browser viewer for the original Skyseed pygame game."""
import io, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from PIL import Image
from glider_game import GliderGame

HTML='''<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><title>Skyseed</title><style>body{margin:0;background:#0b1020;color:#fff;text-align:center;font:16px system-ui}main{max-width:960px;margin:16px auto}img{width:min(960px,100%);image-rendering:auto}button{font-size:24px;margin:8px;padding:8px 28px}small{color:#b6c4d9}</style><main><h2>SKYSEED · nanoOasis</h2><img id=f src="/frame.jpg"><div><button id=d>HOLD TO DIVE</button></div><small>Hold Space / ↑ or the button to dive; release to glide · R to restart</small></main><script>let dive=false,busy=false,img=document.querySelector('#f');async function tick(){if(busy)return;busy=true;try{let r=await fetch('/step?dive='+(dive?1:0));let old=img.src;img.src=URL.createObjectURL(await r.blob());if(old.startsWith('blob:'))URL.revokeObjectURL(old)}finally{busy=false}}setInterval(tick,16);let b=document.querySelector('#d');b.onmousedown=()=>dive=true;b.onmouseup=b.onmouseleave=()=>dive=false;onkeydown=e=>{if(e.key===' '||e.key==='ArrowUp'){e.preventDefault();dive=true}if(e.key.toLowerCase()==='r')fetch('/reset')};onkeyup=e=>{if(e.key===' '||e.key==='ArrowUp')dive=false};</script>'''

class App:
    def __init__(self):
        import pygame
        pygame.init()
        self.game=GliderGame(); self.lock=threading.Lock(); self.frame=self.game.render()
    def step(self,dive):
        with self.lock:self.frame=self.game.step(dive);return self.frame
    def reset(self):
        with self.lock:self.game.reset();self.frame=self.game.render();return self.frame
APP=App()
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        path=urlparse(self.path); route=path.path
        if route=='/': return self.out(200,'text/html; charset=utf-8',HTML.encode())
        if route=='/reset': return self.jpg(APP.reset())
        if route=='/step': return self.jpg(APP.step(parse_qs(path.query).get('dive',['0'])[0]=='1'))
        if route=='/frame.jpg': return self.jpg(APP.frame)
        self.send_error(404)
    def jpg(self,frame):
        b=io.BytesIO();Image.fromarray(frame).save(b,'JPEG',quality=88);self.out(200,'image/jpeg',b.getvalue())
    def out(self,code,typ,data):
        self.send_response(code);self.send_header('Content-Type',typ);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def log_message(self,*a):pass
if __name__=='__main__':
    import argparse;p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8766);a=p.parse_args();print(f'Skyseed browser game: http://127.0.0.1:{a.port}',flush=True);ThreadingHTTPServer(('0.0.0.0',a.port),H).serve_forever()
