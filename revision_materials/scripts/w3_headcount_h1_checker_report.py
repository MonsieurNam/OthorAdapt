#!/usr/bin/env python3
import json, os, re
from datetime import datetime, timedelta, timezone
from pathlib import Path
ROOT=Path(os.environ.get('PROJECT_ROOT','/root/OthorAdapt'))
COMMANDS=ROOT/'revision_materials/scripts/w3_headcount_h1_ramp100_commands.sh'
MANIFEST=ROOT/'revision_materials/results/w3_headcount_h1_ramp100_results.jsonl'
VALIDATION_MANIFEST=ROOT/'revision_materials/results/validation_sweep_ramp100_results.jsonl'
LOGDIR=ROOT/'revision_materials/logs/w3_headcount_h1_ramp100'
VN_TZ=timezone(timedelta(hours=7))
FILENAME_RE=re.compile(r'(?:^|\s)--filename\s+("[^"]+"|\'[^\']+\'|\S+)')
PROG_RE=re.compile(r'Training:\s+(\d+)%.*?\|\s*(\d+)/(\d+).*?elapsed=([0-9:]+), eta=([0-9:]+)')
ITER_RE=re.compile(r'Iter\s+(\d+)/(\d+)')
def unq(x): return x[1:-1] if len(x)>=2 and x[0]==x[-1] and x[0] in "'\"" else x
def load_commands():
    out=[]
    if COMMANDS.exists():
        for line in COMMANDS.read_text(encoding='utf-8').splitlines():
            if 'main.py' in line and '--filename' in line:
                m=FILENAME_RE.search(line)
                if m: out.append((unq(m.group(1)), line.strip()))
    return out
def nested(d,path,default=None):
    cur=d
    for k in path.split('.'):
        if not isinstance(cur,dict) or k not in cur: return default
        cur=cur[k]
    return cur
def completed_from_manifest(path):
    runt=[]
    if not Path(path).exists(): return runt
    for line in Path(path).read_text(encoding='utf-8',errors='ignore').splitlines():
        if not line.strip(): continue
        try: r=json.loads(line)
        except Exception: continue
        if r.get('status')=='completed':
            rt=nested(r,'metrics.runtime_seconds') or nested(r,'metrics.fine_tuning_seconds')
            try:
                if rt is not None and float(rt)>=0: runt.append(float(rt))
            except Exception: pass
    return runt
def completed():
    done=set(); runt=[]
    if not MANIFEST.exists(): return done,runt
    for line in MANIFEST.read_text(encoding='utf-8',errors='ignore').splitlines():
        if not line.strip(): continue
        try: r=json.loads(line)
        except Exception: continue
        fn=nested(r,'config.filename')
        if r.get('status')=='completed' and fn:
            done.add(str(fn)); rt=nested(r,'metrics.runtime_seconds') or nested(r,'metrics.fine_tuning_seconds')
            try:
                if rt is not None and float(rt)>=0: runt.append(float(rt))
            except Exception: pass
    return done,runt
def hms(s):
    p=[int(x) for x in s.split(':')]
    return p[0]*60+p[1] if len(p)==2 else (p[0]*3600+p[1]*60+p[2] if len(p)==3 else 0)
def fmt(sec):
    sec=max(0,int(round(sec))); h,rem=divmod(sec,3600); m,s=divmod(rem,60)
    return f'{h}h{m:02d}m{s:02d}s' if h else (f'{m}m{s:02d}s' if m else f'{s}s')
def active_log_info():
    logs=sorted(LOGDIR.glob('*.log'), key=lambda p:p.stat().st_mtime, reverse=True)
    if not logs: return None
    p=logs[0]
    try: text=p.read_bytes()[-600000:].decode('utf-8','ignore').replace('\r','\n')
    except Exception: return {'path':str(p)}
    info={'path':str(p),'mtime':p.stat().st_mtime}
    prog=PROG_RE.findall(text); iters=ITER_RE.findall(text)
    if prog:
        pct,cur,total,elapsed,eta=prog[-1]
        info.update(percent=int(pct),iter=int(cur),total_iter=int(total),elapsed_seconds=hms(elapsed),eta_seconds=hms(eta),elapsed_human=elapsed,eta_human=eta)
    elif iters:
        cur,total=iters[-1]; info.update(iter=int(cur),total_iter=int(total))
    return info
def main():
    now=datetime.now(timezone.utc); cmds=load_commands(); names=[c[0] for c in cmds]
    done,runt=completed(); done_count=sum(1 for n in names if n in done); pending=max(0,len(names)-done_count)
    active=active_log_info(); per_run=(sum(runt)/len(runt)) if runt else None; rate='unknown'; active_eta=0
    if per_run is None:
        vrunt=completed_from_manifest(VALIDATION_MANIFEST)
        if vrunt:
            per_run=sum(vrunt)/len(vrunt); rate=f'fallback main validation mean {fmt(per_run)}/run'
    if active and active.get('elapsed_seconds') is not None and active.get('eta_seconds') is not None:
        total_current=active['elapsed_seconds']+active['eta_seconds']; active_eta=active['eta_seconds'] if pending>0 else 0
        if per_run is None and total_current>0: per_run=total_current; rate=f'current run estimate {fmt(per_run)}/run'
    if per_run is not None:
        eta=active_eta+max(0,pending-1)*per_run
        if rate=='unknown': rate=f'mean completed {fmt(per_run)}/run'
    else: eta=0
    finish=now+timedelta(seconds=int(eta))
    print(json.dumps({'done':done_count,'total':len(names),'pending':pending,'eta_seconds':int(round(eta)),'eta_human':fmt(eta),'eta_hours':round(eta/3600,1),'estimated_finish_utc':finish.strftime('%Y-%m-%d %H:%M UTC'),'estimated_finish_vn':finish.astimezone(VN_TZ).strftime('%Y-%m-%d %H:%M VN'),'rate':rate,'manifest':str(MANIFEST),'active_log':active},sort_keys=True))
if __name__=='__main__': main()
