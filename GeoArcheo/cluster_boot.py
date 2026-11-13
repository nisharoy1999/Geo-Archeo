import pandas as pd, numpy as np, json
S='outputs/'   # run from the project root
def per_site(d,col):
    o={}
    for sid,x in d.groupby('site_id'):
        pos=x[x.patch=='site'][col].values[0]; neg=x[x.patch=='background'][col].values
        o[sid]=np.mean((pos>neg)+0.5*(pos==neg))
    return pd.Series(o)
def pooled(prefix,cols):
    fs=['','_s7','_s11','_s19','_s42']; D=[pd.read_csv(f'{S}{prefix}{f}.csv') for f in fs]
    return pd.DataFrame({c:pd.concat([per_site(d,c) for d in D],axis=1).mean(axis=1) for c in cols})
E=pooled('detect_predictions_dem_fabdem',['rf_1date','rf_4season','dem_only','rf_1date_dem','rf_4season_dem'])
Q=pooled('detect_predictions_seasons',['rf_1date','rf_4season','prithvi_1date','prithvi_4season','prithvi_4season_per'])
d0=pd.read_csv(S+'detect_predictions_dem_fabdem.csv'); reg=d0[d0.patch=='site'].set_index('site_id').region_group
gj=json.load(open('data/labels/eamena_sites.geojson'))
lon=pd.DataFrame([(f['properties']['site_id'],f['geometry']['coordinates'][0]) for f in gj['features']],columns=['s','lon']).set_index('s').lon
cent=lon.loc[reg.index].groupby(reg).mean().sort_values(); rid={g:i+1 for i,g in enumerate(cent.index)}
R=reg.map(rid)
def boots(B=5000,seed=0):
    rng=np.random.default_rng(seed); ids=np.array(sorted(E.index)); regs=np.array(sorted(R.unique()))
    members={r:np.where(R.loc[ids].values==r)[0] for r in regs}
    site_i=[rng.choice(len(ids),len(ids)) for _ in range(B)]
    clu=[];hier=[]
    for _ in range(B):
        pick=rng.choice(regs,len(regs))
        clu.append(np.concatenate([members[r] for r in pick]))
        hier.append(np.concatenate([rng.choice(members[r],len(members[r])) for r in pick]))
    return ids,site_i,clu,hier
ids,SI,CL,HI=boots()
def holm(p):
    o=np.argsort(p); adj=np.empty(len(p)); run=0
    for k,i in enumerate(o): run=max(run,min(1,(len(p)-k)*p[i])); adj[i]=run
    return adj
def stat(vec,idxs):
    return np.array([vec[i].mean() for i in idxs])
def summarize(vec):
    out={}
    for nm,ix in [('site',SI),('cluster',CL),('two-stage',HI)]:
        b=stat(vec,ix); lo,hi=np.percentile(b,[2.5,97.5]); out[nm]=(lo,hi,max(float(np.mean(b<=0)),1/len(b)))
    return out
res={}
# levels
levels={}
for nm,col in [('Random forest, 1 date',('E','rf_1date')),('Random forest, 4 seasons',('E','rf_4season')),('Elevation only (FABDEM)',('E','dem_only')),('Random forest, 1 date + elevation',('E','rf_1date_dem')),('Random forest, 4 seasons + elevation',('E','rf_4season_dem')),('Prithvi (frozen), 1 date',('Q','prithvi_1date')),('Prithvi (frozen), 4 seasons',('Q','prithvi_4season')),('Prithvi (frozen), 4 seasons, per season',('Q','prithvi_4season_per'))]:
    T=E if col[0]=='E' else Q
    v=T.loc[ids,col[1]].values; s=summarize(v); levels[nm]=dict(mean=v.mean(),**{k:[s[k][0],s[k][1]] for k in s})
# comparisons
fam1=[('Prithvi: 4 seasons minus 1 date',Q.prithvi_4season-Q.prithvi_1date),('Random forest: 4 seasons minus 1 date',Q.rf_4season-Q.rf_1date),('4 seasons: Prithvi minus random forest',Q.prithvi_4season-Q.rf_4season),('4 seasons, per season: Prithvi minus random forest',Q.prithvi_4season_per-Q.rf_4season)]
fam2=[('1 date: + elevation minus without',E.rf_1date_dem-E.rf_1date),('4 seasons: + elevation minus without',E.rf_4season_dem-E.rf_4season),('Elevation only minus chance (0.50)',E.dem_only-0.5)]
comp=[]
for fam in (fam1,fam2):
    rows=[]
    for nm,ser in fam:
        v=ser.loc[ids].values; s=summarize(v); rows.append((nm,v.mean(),s))
    for meth in ['site','cluster','two-stage']:
        pa=holm(np.array([r[2][meth][2] for r in rows]))
        for (nm,m,s),p in zip(rows,pa):
            comp.append(dict(comparison=nm,method=meth,diff=m,lo=s[meth][0],hi=s[meth][1],P=s[meth][2],holm=p))
C=pd.DataFrame(comp); C.to_csv(S+'cluster_comparisons.csv',index=False)
json.dump(levels,open(S+'cluster_levels.json','w'),indent=1)
pd.set_option('display.width',200)
print(C.round(3).to_string())
for k,v in levels.items(): print(k,round(v['mean'],3),{m:[round(x,3) for x in v[m]] for m in ['site','cluster','two-stage']})
# leave-one-region-out for elevation gain
v=(E.rf_1date_dem-E.rf_1date).loc[ids].values; Rv=R.loc[ids].values
print('LORO gain 1date+elev:',{int(r):round(v[Rv!=r].mean(),3) for r in sorted(set(Rv))})
v2=(E.dem_only-0.5).loc[ids].values
print('LORO dem_only-0.5 min/max',round(min(v2[Rv!=r].mean() for r in set(Rv)),3),round(max(v2[Rv!=r].mean() for r in set(Rv)),3))
# per-region gain & fraction of regions with positive gain
g=pd.Series(v,index=ids).groupby(R.loc[ids].values).agg(['mean','count']); print(g.round(3).T.to_string()); print('regions positive',(g['mean']>0).sum(),'/14')
