"""Browser UI for the Breakout DiT demo. Run on the GPU host, then SSH-tunnel port 8765."""
import argparse, base64, io, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from game import LEFT, NONE, RIGHT
from play_breakout_model import load_models, initial_context
from infer_breakout import make_schedule, sample_next_frame, decode_latent

HTML = r'''<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1">
<title>nanoOasis Breakout model</title><style>
body{margin:0;background:#080b14;color:#e8eefc;font:16px system-ui;text-align:center}
main{max-width:780px;margin:24px auto;padding:0 16px}h1{font-size:22px}
img{width:min(768px,100%);image-rendering:pixelated;border:1px solid #33405c;background:#0a0e18}
button{font-size:22px;margin:10px 5px;padding:10px 24px;background:#18243c;color:#e8eefc;border:1px solid #53698f;border-radius:8px}
small{color:#99a6c2}.down{background:#2b4772}
</style><main><h1>nanoOasis Breakout · DiT world model</h1>
<img id=f src="/frame.jpg"><div><button data-a=0>◀</button><button data-a=1>●</button><button data-a=2>▶</button></div>
<small>Arrow keys / A,D control the paddle · Esc stops the browser session · each action predicts one frame</small></main>
<script>
let busy=false, current=1; const img=document.querySelector('#f');
async function step(a){if(busy)return;busy=true;try{let r=await fetch('/step?action='+a);if(r.ok){let old=img.src;img.src=URL.createObjectURL(await r.blob());if(old.startsWith('blob:'))URL.revokeObjectURL(old)}}finally{busy=false}}
// Advance the model continuously; keys change the held paddle action instead of triggering one frame.
setInterval(()=>step(current),16);
document.querySelectorAll('button').forEach(b=>{b.onmousedown=()=>current=+b.dataset.a;b.onmouseup=()=>current=1;b.onmouseleave=()=>current=1});
window.onkeydown=e=>{if(e.key==='ArrowLeft'||e.key.toLowerCase()==='a'){e.preventDefault();current=0}else if(e.key==='ArrowRight'||e.key.toLowerCase()==='d'){e.preventDefault();current=2}else if(e.key===' '||e.key==='ArrowDown'){e.preventDefault();current=1}};
window.onkeyup=e=>{if(['ArrowLeft','ArrowRight','a','d','A','D'].includes(e.key))current=1};
</script>'''

class App:
    def __init__(self, args):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg,self.vae,self.diff=load_models(args,self.device)
        self.z,self.actions=initial_context(self.vae,self.cfg,args.seed,self.device)
        self.sigmas=make_schedule(self.cfg,args.steps,args.sigma_max,str(self.device))
        self.steps=args.steps; self.sigma_stab=args.sigma_stab; self.lock=threading.Lock()
        self.frame=decode_latent(self.vae,self.z[:,-1:])
    @torch.no_grad()
    def step(self, action):
        with self.lock:
            zctx=self.z[:,1:]
            acts=torch.tensor(self.actions[1:]+[int(action)],dtype=torch.long,device=self.device)
            new=sample_next_frame(self.diff,zctx,acts,self.sigmas,self.sigma_stab)
            self.z=torch.cat([zctx,new],dim=1); self.actions=self.actions[1:]+[int(action)]
            self.frame=decode_latent(self.vae,new)
            return self.frame

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/' or self.path.startswith('/?'): return self.send(200,'text/html; charset=utf-8',HTML.encode())
        if self.path.split('?')[0]=='/frame.jpg': return self.jpg(APP.frame)
        if self.path.split('?')[0]=='/step':
            from urllib.parse import parse_qs,urlparse
            q=parse_qs(urlparse(self.path).query); a=int(q.get('action',['1'])[0]); return self.jpg(APP.step(a))
        self.send_error(404)
    def do_POST(self): self.do_GET()
    def jpg(self,frame):
        b=io.BytesIO(); Image.fromarray(frame).save(b,format='JPEG',quality=88); self.send(200,'image/jpeg',b.getvalue())
    def send(self,code,ctype,data):
        self.send_response(code); self.send_header('Content-Type',ctype); self.send_header('Content-Length',str(len(data))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(data)
    def log_message(self,*args): pass

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='breakout_100k_latents'); p.add_argument('--vae',default='checkpoints/vae_breakout_100k.pt'); p.add_argument('--ckpt',default='checkpoints/dit_breakout_100k_latents.pt'); p.add_argument('--steps',type=int,default=2); p.add_argument('--sigma-stab',type=float,default=.1); p.add_argument('--sigma-max',type=float,default=0.); p.add_argument('--seed',type=int,default=42); p.add_argument('--port',type=int,default=8765); args=p.parse_args()
    APP=App(args); print(f'Browser game ready: http://127.0.0.1:{args.port}',flush=True); ThreadingHTTPServer(('0.0.0.0',args.port),Handler).serve_forever()
