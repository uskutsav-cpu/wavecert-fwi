"""Future preparation mechanics; import is stdlib only and performs no work."""
from __future__ import annotations
import email.parser,hashlib,json,os,re,shutil,tarfile,tomllib,urllib.parse,urllib.request,zipfile
from pathlib import Path,PurePosixPath
MIB=1024**2
CAP={"archive":64*MIB,"json":MIB,"runtime":192*MIB,"build":64*MIB,"workspace":8192*MIB,"free":4096*MIB,"metadata":16*MIB}
TARGET={"numpy":"2.4.3","scipy":"1.17.1","devito":"4.8.23","pyrevolve":"2.2.8","pip":"26.2","cgen":"2025.1","codepy":"2023.1","contexttimer":"0.3.3"}

def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(65536),b""):h.update(b)
    return h.hexdigest()

def write(p,obj):
    with Path(p).open("x") as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")

def read(p):
    p=Path(p)
    if p.is_symlink() or p.stat().st_size>CAP["json"]:raise ValueError("bounded regular JSON required")
    def unique(rows):
        d={}
        for k,v in rows:
            if k in d:raise ValueError("duplicate JSON key")
            d[k]=v
        return d
    return json.loads(p.read_bytes(),object_pairs_hook=unique)

def size(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file() and not p.is_symlink()) if Path(root).exists() else 0

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*a,**k):raise ValueError("redirect prohibited")

class Fetch:
    def __init__(self):self.rows=[];self.bytes=0
    def get(self,url,target,cap,sha=None,nbytes=None):
        u=urllib.parse.urlsplit(url)
        if u.scheme!="https" or u.hostname not in ("pypi.org","files.pythonhosted.org") or u.username or u.password or u.query or u.fragment:raise ValueError("exact approved PyPI URL required")
        row={"url":url,"destination":str(target),"attempted":True,"completed":False,"actual_body_bytes":0};self.rows.append(row)
        part=target.with_name(target.name+".part");h=hashlib.sha256()
        try:
            q=urllib.request.Request(url,headers={"Accept-Encoding":"identity"})
            with urllib.request.build_opener(NoRedirect()).open(q,timeout=30) as r,part.open("xb") as out:
                if r.status!=200 or r.geturl()!=url or r.headers.get("Content-Encoding") not in (None,"identity"):raise ValueError("exact unencoded HTTP200 required")
                length=r.headers.get("Content-Length")
                if length is not None and (int(length)>cap or nbytes is not None and int(length)!=nbytes):raise ValueError("declared transfer size differs")
                while True:
                    b=r.read(min(65536,cap-row["actual_body_bytes"]+1))
                    if not b:break
                    row["actual_body_bytes"]+=len(b);self.bytes+=len(b)
                    if row["actual_body_bytes"]>cap or self.bytes>272*MIB:raise ValueError("explicit transfer cap exceeded")
                    out.write(b);h.update(b)
                out.flush();os.fsync(out.fileno())
            row["sha256"]=h.hexdigest()
            if sha is not None and row["sha256"]!=sha or nbytes is not None and row["actual_body_bytes"]!=nbytes:raise ValueError("actual size/hash differs")
            part.rename(target);row["completed"]=True;return target
        except BaseException as e:
            row["failure"]=type(e).__name__+": "+str(e);row["closed_partial_bytes"]=part.stat().st_size if part.exists() else 0;raise

def safe_names(names):
    seen=set()
    for name in names:
        p=PurePosixPath(name)
        if not name or p.is_absolute() or ".." in p.parts or "\\" in name or str(p) in seen:raise ValueError("unsafe or duplicate archive member")
        seen.add(str(p))

def source_tree(archive,destination):
    destination.mkdir()
    with tarfile.open(archive,"r:gz") as f:
        rows=f.getmembers();safe_names(m.name for m in rows)
        if len(rows)>100000 or sum(m.size for m in rows)>1024*MIB:raise ValueError("source expansion cap")
        if any(not(m.isfile() or m.isdir()) for m in rows):raise ValueError("links/devices forbidden")
        for m in rows:
            p=destination/m.name
            if m.isdir():p.mkdir(parents=True,exist_ok=True);continue
            p.parent.mkdir(parents=True,exist_ok=True)
            with f.extractfile(m) as stream,p.open("xb") as out:
                left=m.size
                while left:
                    b=stream.read(min(left,65536))
                    if not b:raise ValueError("short archive member")
                    out.write(b);left-=len(b)
    roots=list(destination.iterdir())
    if len(roots)!=1 or not roots[0].is_dir():raise ValueError("one source root required")
    root=roots[0];p=root/"pyproject.toml";requirements=[]
    if p.exists():
        if p.stat().st_size>MIB:raise ValueError("bounded pyproject required")
        build=tomllib.loads(p.read_text()).get("build-system",{})
        if build.get("backend-path") or build.get("build-backend","setuptools.build_meta") not in ("setuptools.build_meta","setuptools.build_meta:__legacy__","hatchling.build","flit_core.buildapi"):raise ValueError("unregistered backend")
        requirements=build.get("requires",[])
    if not isinstance(requirements,list) or any(not isinstance(r,str) or not re.match("[A-Za-z0-9]",r) or any(t in r for t in ("\n","\r","@","://")) for r in requirements):raise ValueError("only index build requirements permitted")
    return root,requirements

def wheel(path):
    if path.stat().st_size>CAP["archive"]:raise ValueError("wheel size cap")
    with zipfile.ZipFile(path) as f:
        rows=f.infolist();safe_names(r.filename for r in rows)
        if len(rows)>100000 or sum(r.file_size for r in rows)>1024*MIB or any((r.external_attr>>16)&0o170000==0o120000 for r in rows):raise ValueError("wheel expansion/member/link cap")
        names=[r.filename for r in rows if r.filename.endswith(".dist-info/METADATA")]
        if len(names)!=1:raise ValueError("one wheel METADATA required")
        with f.open(names[0]) as stream:b=stream.read(MIB+1)
        if len(b)>MIB:raise ValueError("bounded wheel METADATA required")
    m=email.parser.BytesParser().parsebytes(b)
    return {"name":re.sub("[-_.]+","-",m["Name"]).lower(),"version":m["Version"],"filename":path.name,"bytes":path.stat().st_size,"sha256":digest(path),"requires_dist":m.get_all("Requires-Dist",[])}

def selections(report,built):
    if report.get("version")!="1":raise ValueError("pip report v1 required")
    result=[];seen=set()
    for item in report["install"]:
        m=item["metadata"];name=re.sub("[-_.]+","-",m["name"]).lower();v=m["version"];d=item["download_info"];url=d["url"];u=urllib.parse.urlsplit(url);h=d.get("archive_info",{}).get("hashes",{}).get("sha256")
        if name in seen or item.get("is_yanked") is True or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*",v) or not re.fullmatch("[0-9a-f]{64}",h or ""):raise ValueError("unique stable nonyanked hash-bound selection required")
        seen.add(name)
        if u.scheme=="file":
            p=Path(urllib.request.url2pathname(u.path)).resolve()
            if u.netloc or str(p) not in built or digest(p)!=h or built[str(p)]["sha256"]!=h:raise ValueError("local wheel lacks successful exact build binding")
        elif u.scheme!="https" or u.hostname!="files.pythonhosted.org" or u.query or u.fragment or not u.path.endswith(".whl"):raise ValueError("non-wheel or unapproved origin")
        result.append({"name":name,"version":v,"url":url,"sha256":h,"local":u.scheme=="file"})
    return result

def inventory(root):
    rows={}
    for p in sorted(Path(root).rglob("*")):
        if p.is_symlink():rows[str(p.relative_to(root))]={"symlink":os.readlink(p),"resolved":str(p.resolve())}
        elif p.is_file():rows[str(p.relative_to(root))]={"bytes":p.stat().st_size,"sha256":digest(p),"mode":p.stat().st_mode}
    return rows

def copy_packet(source,target,cap):
    target.mkdir(exist_ok=True);rows={}
    for p in sorted(source.rglob("*")):
        if not p.is_file():continue
        if p.is_symlink():raise ValueError("artifact symlink")
        q=target/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True)
        before={"bytes":p.stat().st_size,"sha256":digest(p)}
        if size(target)+before["bytes"]>cap:raise ValueError("artifact cap")
        shutil.copyfile(p,q);after={"bytes":q.stat().st_size,"sha256":digest(q)}
        if before!=after:raise ValueError("source changed during copy")
        rows[str(p.relative_to(source))]=after
    write(target/"artifact_inventory.json",{"files":rows,"complete":True,"inventory_excludes_itself":True})
    if size(target)>cap:raise ValueError("final retained cap")

def acquire_selected(route,rows,directory):
    paths=[]
    for row in rows:
        if row["local"]:paths.append(Path(urllib.request.url2pathname(urllib.parse.urlsplit(row["url"]).path)));continue
        meta=route.raw/("release_"+row["name"]+"_"+row["version"]+".json")
        if not meta.exists():route.fetch.get("https://pypi.org/pypi/"+row["name"]+"/"+row["version"]+"/json",meta,MIB)
        filename=urllib.parse.unquote(PurePosixPath(urllib.parse.urlsplit(row["url"]).path).name)
        found=[a for a in read(meta)["urls"] if a["filename"]==filename and a["url"]==row["url"] and a["digests"]["sha256"]==row["sha256"] and not a["yanked"] and a["packagetype"]=="bdist_wheel"]
        if len(found)!=1:raise ValueError("selected wheel differs from official release record")
        a=found[0];p=directory/filename
        if p.exists():
            if digest(p)!=row["sha256"]:raise ValueError("existing wheel changed")
        else:route.fetch.get(row["url"],p,CAP["archive"],row["sha256"],a["size"])
        wheel(p);paths.append(p)
    return paths

def bootstrap(route,prefix,pipwheel,label):
    route.child(label+"_venv",[sys_executable(),"-B","-I","-m","venv","--copies","--without-pip",prefix])
    python=prefix/"bin/python"
    code='import runpy,sys;sys.path.insert(0,sys.argv.pop(1));sys.argv[0]="pip";runpy.run_module("pip",run_name="__main__")'
    route.child(label+"_pip",[python,"-B","-I","-c",code,pipwheel,"install","--no-index","--no-deps","--no-compile",pipwheel])
    return python

def sys_executable():
    import sys
    return sys.executable

def resolve(route,python,requirements,label,local=()):
    report=route.raw/(label+".json")
    route.child(label,[python,"-B","-I","-m","pip","install","--dry-run","--ignore-installed","--only-binary=:all:","--retries","0","--index-url","https://pypi.org/simple","--report",report,"-r",requirements,*local])
    return selections(read(report),route.built)

def install(route,python,paths,label):
    lock=route.raw/(label+".lock");lock.write_text("".join(p.as_uri()+" --hash=sha256:"+digest(p)+"\n" for p in paths))
    route.child(label,[python,"-B","-I","-m","pip","install","--no-index","--no-deps","--require-hashes","--only-binary=:all:","--no-compile","-r",lock])

def pipeline(route,go):
    """Called only after source/root authorization and actual Linux namespace gate."""
    root=route.root;here=route.here
    sources=root/"sources";sources.mkdir();trees=root/"source_trees";trees.mkdir()
    requirements=(here/"build_requirements.in").read_text().splitlines();registered=[]
    for a in read(here/"SOURCE_BUILD_ALLOWLIST.json")["projects"]:
        archive=route.fetch.get(a["url"],sources/a["filename"],CAP["archive"],a["sha256"],a["declared_bytes"])
        tree,declared=source_tree(archive,trees/a["distribution"]);requirements+=declared;registered.append((a,archive,tree))
    req=route.raw/"build_requires.actual.in";req.write_text("\n".join(sorted(set(requirements)))+"\n")
    wheels=root/"acquired_wheels";wheels.mkdir();meta=route.raw/"pip26.2.json"
    route.fetch.get("https://pypi.org/pypi/pip/26.2/json",meta,MIB)
    selected=[a for a in read(meta)["urls"] if a["filename"]=="pip-26.2-py3-none-any.whl" and not a["yanked"]]
    if len(selected)!=1:raise ValueError("exact pip26.2 wheel required")
    a=selected[0];pipwheel=route.fetch.get(a["url"],wheels/a["filename"],CAP["archive"],a["digests"]["sha256"],a["size"])
    buildpython=bootstrap(route,root/"build_env",pipwheel,"build")
    buildpaths=acquire_selected(route,resolve(route,buildpython,req,"resolve_build"),wheels)
    install(route,buildpython,buildpaths,"install_build")
    built=root/"built";built.mkdir()
    for a,archive,tree in registered:
        output=built/a["distribution"];output.mkdir();command=route.raw/(a["distribution"]+"_argv.json")
        write(command,{"argv":[str(buildpython),"-B","-I","-m","pip","wheel","--no-index","--no-deps","--no-build-isolation","--wheel-dir",str(output),str(tree)]})
        wait,_=route.child("build_"+a["distribution"],route.namespace(buildpython,route.raw/(a["distribution"]+"_boundary.json"),["--command",command]))
        files=list(output.glob("*.whl"))
        if len(files)!=1:raise ValueError("one built wheel required")
        info=wheel(files[0])
        if info["name"]!=a["distribution"] or info["version"]!=a["version"] or a["pure_python_wheel_required"] and not files[0].name.endswith("none-any.whl"):raise ValueError("built wheel identity/type differs")
        route.built[str(files[0].resolve())]={**info,"actual_build_wait":wait,"source_sha256":a["sha256"]}
    write(route.raw/"built_wheel_bindings.json",route.built)
    req=route.raw/"runtime_requirements.in";source_names={a["distribution"] for a,_,_ in registered}
    req.write_text("".join(k+"=="+v+"\n" for k,v in TARGET.items() if k not in source_names))
    rows=resolve(route,buildpython,req,"resolve_runtime",list(route.built))
    if {a["name"]:a["version"] for a in rows if a["name"] in TARGET}!=TARGET:raise ValueError("fixed target projection differs")
    runtime=acquire_selected(route,rows,wheels)
    if sum(p.stat().st_size for p in runtime)>CAP["runtime"] or size(sources)+sum(p.stat().st_size for p in buildpaths)>CAP["build"]:raise ValueError("wheel/input budget")
    write(route.raw/"runtime_wheel_manifest.json",[wheel(p) for p in runtime]);write(route.raw/"build_wheel_manifest.json",[wheel(p) for p in buildpaths])
    python=bootstrap(route,route.prefix,pipwheel,"candidate");install(route,python,runtime,"install_candidate")
    route.child("pip_check",[python,"-B","-I","-m","pip","check"])
    _,log=route.child("pip_inspect",[python,"-B","-I","-m","pip","inspect","--local"]);write(route.raw/"pip_inspect.json",json.loads(log.read_bytes()))
    for a in read(here/"frozen_v2/RUNTIME_SOURCE_PINS.json")["files"]:
        if digest(route.prefix/"lib/python3.11/site-packages"/a["path"])!=a["sha256"]:raise ValueError("primary source differs BEFORE imports")
    route.child("import_identity",[python,"-B","-I",here/"import_identity_probe.py","--execute","--root-preparation-go",route.raw/"root_preparation_go.json","--root-preparation-go-sha256",digest(route.raw/"root_preparation_go.json"),"--protocol",here/"frozen_v2/protocol.json","--pins",here/"frozen_v2/RUNTIME_SOURCE_PINS.json","--prefix",route.prefix,"--output",route.raw/"import_probe"])
    receipt=read(route.raw/"import_probe/environment_receipt.json")
    if receipt.get("native_use_review_clear") is not False:raise ValueError("job cannot approve native use")
    write(route.raw/"installed_inventory.json",inventory(route.prefix))
    for label,argv in [("gcc",["/usr/bin/gcc","--version"]),("gxx",["/usr/bin/g++","--version"]),("linker",["/usr/bin/ld","--version"]),("OS_packages",["/usr/bin/dpkg-query","-W"])]:route.child(label,argv)
    elf=[p for p in route.prefix.rglob("*") if p.is_file() and ".so" in p.name]
    if elf:route.child("ELF_dynamic",["/usr/bin/readelf","-d",*elf]);route.child("ELF_linkage",["/usr/bin/ldd",*elf])
    package=root/"wheel_artifact";(package/"runtime").mkdir(parents=True);(package/"build_inputs").mkdir()
    for p in runtime:shutil.copyfile(p,package/"runtime"/p.name)
    for p in [*buildpaths,*[a for _,a,_ in registered]]:
        target=package/"build_inputs"/p.name
        if not target.exists():shutil.copyfile(p,target)
    write(package/"wheel_manifest.json",inventory(package))
    if size(package)>256*MIB:raise ValueError("wheel artifact cap")
    return {"status":"CANDIDATE_ENVIRONMENT_PREPARED_REQUIRES_ROOT_REVIEW","native_use_review_clear":False,"native_execution_authorized":False,"checkpoint_children":0,"PDE_calls":0}
