"""Future Linux stdlib-only offline-build gate; default performs no probe."""
from __future__ import annotations
import argparse,errno,json,os,signal,socket,subprocess,sys,time
from pathlib import Path

def check(parent_netns,uid,gid):
    netns=os.readlink('/proc/self/ns/net')
    fields={s.split(':',1)[0]:s.split(':',1)[1].strip() for s in Path('/proc/self/status').read_text().splitlines() if ':' in s}
    caps={k:int(fields[k],16) for k in ('CapInh','CapPrm','CapEff','CapBnd','CapAmb')}
    if (netns==parent_netns or uid==0 or os.getuid()!=uid or os.geteuid()!=uid
            or os.getgid()!=gid or os.getegid()!=gid or os.getgroups()
            or fields.get('NoNewPrivs')!='1' or any(caps.values())):
        raise ValueError('new netns/nonroot/no-new-privs/empty caps and groups required')
    routes=Path('/proc/net/route').read_text().splitlines()
    if len(routes)>1:raise ValueError('offline namespace has IPv4 routes')
    return {'PID':os.getpid(),'PPID':os.getppid(),'PGID':os.getpgrp(),'SID':os.getsid(0),
        'network_namespace':netns,'parent_network_namespace':parent_netns,'UID':uid,'GID':gid,
        'supplementary_groups':os.getgroups(),'NoNewPrivs':1,'capabilities':caps,
        'IPv4_route_rows':max(0,len(routes)-1),'scope':'Prevent implicit fetches in trusted registered builds; no general sandbox claim.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--parent-netns');p.add_argument('--uid',type=int);p.add_argument('--gid',type=int)
    p.add_argument('--output',type=Path);p.add_argument('--command',type=Path);p.add_argument('--dummy-signal',action='store_true')
    a=p.parse_args()
    if not a.execute:print('HELD_NAMESPACE_GATE_NO_PROBE_OR_PACKAGE_ACTION');return 0
    row=check(a.parent_netns,a.uid,a.gid)
    if a.dummy_signal:
        child=subprocess.run(['/usr/bin/sudo','-n','/usr/bin/true'],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False)
        if child.returncode==0:raise ValueError('privilege regain succeeded')
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.settimeout(.2);network=s.connect_ex(('198.51.100.1',9))
        if network not in (errno.ENETUNREACH,errno.EHOSTUNREACH):raise ValueError('offline rejection not observed')
        received=[]
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda number,frame:received.append(number))
        a.output.with_suffix('.ready').write_text('namespace-ready\n')
        end=time.monotonic()+10
        while not received and time.monotonic()<end:time.sleep(.01)
        if not received:raise ValueError('external owned-group signal not observed')
        row.update(actual_signal_numbers=received,privilege_probe_actual_exit=child.returncode,
            privilege_probe_console=child.stdout.decode(errors='replace')[:4096],network_probe_errno=network,
            status='ACTUAL_LINUX_DUMMY_NAMESPACE_SIGNAL_PASS')
        with a.output.open('x') as f:json.dump(row,f,indent=2,sort_keys=True);f.write('\n')
        return 0
    command=json.loads(a.command.read_bytes());argv=command['argv']
    if (not isinstance(argv,list) or len(argv)<10 or argv[1:6]!=['-B','-I','-m','pip','wheel']
            or not {'--no-index','--no-deps','--no-build-isolation'}.issubset(argv)
            or not Path(argv[0]).is_absolute()):raise ValueError('only registered offline wheel-build argv accepted')
    with a.output.open('x') as f:json.dump({**row,'argv':argv,'status':'OFFLINE_BUILD_BOUNDARY_BEFORE_EXEC'},f,indent=2,sort_keys=True);f.write('\n')
    os.execvpe(argv[0],argv,os.environ.copy())

if __name__=='__main__':raise SystemExit(main())
