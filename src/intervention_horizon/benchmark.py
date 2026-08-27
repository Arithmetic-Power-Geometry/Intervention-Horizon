"""Fixed-seed experiments for Intervention Horizon theory.
Copyright (C) 2026 Mohammad Amir Khusru Akhtar
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .core import collective_horizon_from_individual, time_costed_voi

DISTRICTS = [
('Garhwa',1322784),('Chatra',1042886),('Koderma',716259),('Giridih',2445474),('Deoghar',1492073),('Godda',1313551),('Sahibganj',1150567),('Pakur',900422),('Dhanbad',2684487),('Bokaro',2062330),('Lohardaga',461790),('East Singhbhum',2293919),('Palamu',1939869),('Latehar',726978),('Hazaribagh',1734495),('Ramgarh',949443),('Dumka',1321442),('Jamtara',791042),('Ranchi',2914253),('Khunti',531885),('Gumla',1025213),('Simdega',599578),('West Singhbhum',1502338),('Seraikela-Kharsawan',1065056)]

EXTERNAL_EVIDENCE = [
    ("Jharkhand population", 32988134, "persons", "Census 2011"),
    ("Jharkhand districts", 24, "districts", "Census 2011"),
    ("Lightning flashes, Jharkhand 2000-2020", 1256404, "flashes", "Mishra et al. 2025, Natural Hazards"),
    ("Peak annual flashes (2006)", 101302, "flashes", "Mishra et al. 2025, Natural Hazards"),
    ("Male share of recorded fatalities", 70.39, "percent", "Mishra et al. 2025, Natural Hazards"),
    ("Jharkhand thunderstorm/lightning deaths, 2025", 170, "> deaths", "IMD Annual Climate Summary 2025")]


def simulate(n=50000, seed=20260827):
    rng=np.random.default_rng(seed)
    event=rng.random(n)<0.72
    true_arrival=np.where(event,rng.uniform(20,100,n),np.inf)
    pred_arrival=np.where(event,true_arrival+rng.normal(0,9,n),rng.uniform(70,150,n))
    hazard_sd=rng.uniform(3,10,n)
    travel_true=rng.uniform(4,38,n)
    travel_pred=np.clip(travel_true+rng.normal(0,3,n),1,None)
    travel_sd=rng.uniform(1,5,n)
    p_event=np.where(event,rng.beta(8,2,n),rng.beta(2,8,n))
    true_deadline=true_arrival-travel_true
    avoidable=event&(true_deadline>=0)
    warn_fixed=np.where(p_event>=.5,np.maximum(0,pred_arrival-30),np.inf)
    warn_risk=np.where(p_event>=.65,np.maximum(0,pred_arrival-20),np.inf)
    warn_margin=np.where(p_event>=.5,np.maximum(0,pred_arrival-travel_pred-5),np.inf)
    z=1.6448536269514722
    iho_deadline=pred_arrival-travel_pred-z*np.sqrt(hazard_sd**2+travel_sd**2)
    warn_iho=np.where(p_event>=.5,np.maximum(0,iho_deadline-5),np.inf)
    return pd.DataFrame(dict(event=event,true_arrival=true_arrival,pred_arrival=pred_arrival,hazard_sd=hazard_sd,
        travel_true=travel_true,travel_pred=travel_pred,travel_sd=travel_sd,p_event=p_event,true_deadline=true_deadline,
        avoidable=avoidable,FixedLead=warn_fixed,RiskOnly=warn_risk,MeanMargin=warn_margin,IHO=warn_iho))


def metrics(df):
    rows=[]
    for m in ['FixedLead','RiskOnly','MeanMargin','IHO']:
        w=df[m].to_numpy(); event=df.event.to_numpy(); avoid=df.avoidable.to_numpy(); deadline=df.true_deadline.to_numpy(); arrival=df.true_arrival.to_numpy()
        warned=np.isfinite(w); safe=avoid&warned&(w<=deadline); op_late=event&warned&(w<arrival)&(w>deadline); false=(~event)&warned
        early=np.full(len(df),np.nan); early[safe]=deadline[safe]-w[safe]
        rows.append(dict(method=m,preventable_harm_recall=safe.sum()/max(1,avoid.sum()),
            operational_lateness_rate=op_late.sum()/max(1,(event&warned).sum()),false_warning_rate=false.sum()/max(1,(~event).sum()),
            median_earliness_min=float(np.nanmedian(early)),warnings=int(warned.sum())))
    return pd.DataFrame(rows)


def matched_pairs(n=5000,seed=20260828):
    rng=np.random.default_rng(seed); p=rng.uniform(.55,.95,n); arrival=rng.uniform(25,70,n)
    near=rng.uniform(3,8,n); far=rng.uniform(18,35,n)
    return pd.DataFrame({'p_hazard':p,'hazard_lead_min':arrival,'near_horizon':arrival-near,'far_horizon':arrival-far,'horizon_gap':far-near*-0 + (far-near)})


def observe_or_act(seed=20260829,n=20000):
    rng=np.random.default_rng(seed)
    deadline=rng.uniform(5,35,n); delay=rng.uniform(1,12,n)
    instantaneous_voi=rng.uniform(.01,.25,n); observation_cost=rng.uniform(0,.02,n)
    base_utility=rng.uniform(.55,.85,n)
    # Delay loss grows continuously and jumps when the observation crosses the prevention deadline.
    smooth_loss=0.012*delay
    horizon_loss=np.where(delay>=deadline,base_utility,0.0)
    delay_loss=np.minimum(1.0,smooth_loss+horizon_loss)
    tcvoi=np.array([time_costed_voi(v,d,c) for v,d,c in zip(instantaneous_voi,delay_loss,observation_cost)])
    return pd.DataFrame({'deadline':deadline,'delay':delay,'instantaneous_voi':instantaneous_voi,'delay_loss':delay_loss,
                         'observation_cost':observation_cost,'tcvoi':tcvoi,'negative_tcvoi':tcvoi<0,'crosses_horizon':delay>=deadline})


def collective(seed=20260830,n=12000):
    rng=np.random.default_rng(seed); rec=[]
    for _ in range(n):
        people=int(rng.integers(10,80)); individual=rng.uniform(4,55,people)
        # Shared queuing/coordination bottleneck is nonnegative and grows with crowd/resource pressure.
        capacity=int(rng.integers(max(2,people//5),max(3,people+1)))
        pressure=max(0.0,(people-capacity)/max(capacity,1))
        shared_delay=float(pressure*rng.uniform(1,8))
        ch=collective_horizon_from_individual(individual,shared_delay)
        mn=float(np.min(individual))
        rec.append((people,capacity,mn,ch,mn-ch,ch<=mn+1e-12))
    return pd.DataFrame(rec,columns=['people','capacity','min_individual_horizon','collective_horizon','collective_gap','dominance_holds'])


def epsilon_sensitivity(df, epsilons=(.01,.025,.05,.1,.2)):
    rows=[]
    zmap={e:float(__import__('statistics').NormalDist().inv_cdf(1-e)) for e in epsilons}
    for e in epsilons:
        z=zmap[e]
        d=df.copy()
        deadline=d.pred_arrival-d.travel_pred-z*np.sqrt(d.hazard_sd**2+d.travel_sd**2)
        d['IHO']=np.where(d.p_event>=.5,np.maximum(0,deadline-5),np.inf)
        r=metrics(d).query("method=='IHO'").iloc[0]
        rows.append({'epsilon':e,'preventable_harm_recall':r.preventable_harm_recall,'operational_lateness_rate':r.operational_lateness_rate,'median_earliness_min':r.median_earliness_min})
    return pd.DataFrame(rows)


def ablation(df):
    d=df.copy(); z=1.6448536269514722
    full=d.pred_arrival-d.travel_pred-z*np.sqrt(d.hazard_sd**2+d.travel_sd**2)-5
    no_unc=d.pred_arrival-d.travel_pred-5
    no_access=d.pred_arrival-20-z*d.hazard_sd-5
    rows=[]
    for name,w in [('IHO-full',full),('No-uncertainty',no_unc),('No-access',no_access)]:
        tmp=d.copy(); tmp['IHO']=np.where(tmp.p_event>=.5,np.maximum(0,w),np.inf)
        r=metrics(tmp).query("method=='IHO'").iloc[0]
        rows.append({'variant':name,'preventable_harm_recall':r.preventable_harm_recall,'operational_lateness_rate':r.operational_lateness_rate,'median_earliness_min':r.median_earliness_min})
    return pd.DataFrame(rows)


def write_outputs(out:Path,n=50000):
    tdir=out/'tables'; fdir=out/'figures'; tdir.mkdir(parents=True,exist_ok=True); fdir.mkdir(parents=True,exist_ok=True)
    df=simulate(n=n); met=metrics(df); pairs=matched_pairs(); oa=observe_or_act(); col=collective(); sens=epsilon_sensitivity(df); abl=ablation(df)
    met.to_csv(tdir/'benchmark_metrics.csv',index=False)
    pd.DataFrame({'metric':['near_horizon_mean','far_horizon_mean','horizon_gap_mean','horizon_gap_sd'],
                  'value':[pairs.near_horizon.mean(),pairs.far_horizon.mean(),(pairs.near_horizon-pairs.far_horizon).mean(),(pairs.near_horizon-pairs.far_horizon).std()]}).to_csv(tdir/'matched_pair_summary.csv',index=False)
    pd.DataFrame({'metric':['negative_tcvoi_rate','horizon_crossing_rate','negative_given_crossing','mean_instantaneous_voi','mean_delay'],
                  'value':[oa.negative_tcvoi.mean(),oa.crosses_horizon.mean(),oa.loc[oa.crosses_horizon,'negative_tcvoi'].mean(),oa.instantaneous_voi.mean(),oa.delay.mean()]}).to_csv(tdir/'observe_or_act.csv',index=False)
    pd.DataFrame({'metric':['dominance_holds_rate','mean_collective_gap','positive_gap_rate'],
                  'value':[col.dominance_holds.mean(),col.collective_gap.mean(),(col.collective_gap>1e-12).mean()]}).to_csv(tdir/'collective.csv',index=False)
    sens.to_csv(tdir/'epsilon_sensitivity.csv',index=False); abl.to_csv(tdir/'ablation.csv',index=False)
    pd.DataFrame(DISTRICTS,columns=['district','population_2011']).to_csv(tdir/'jharkhand_population_2011.csv',index=False)
    pd.DataFrame(EXTERNAL_EVIDENCE,columns=['quantity','value','unit','source']).to_csv(tdir/'jharkhand_external_evidence.csv',index=False)

    ax=met.set_index('method')[['preventable_harm_recall','operational_lateness_rate','false_warning_rate']].plot(kind='bar',figsize=(8,4.8)); ax.set_ylabel('Rate'); ax.set_ylim(0,1); ax.set_title('Warning performance against preventability'); plt.tight_layout(); plt.savefig(fdir/'benchmark_performance.pdf'); plt.savefig(fdir/'benchmark_performance.png',dpi=220); plt.close()
    s=pairs.sample(1400,random_state=1); plt.figure(figsize=(6.4,4.6)); plt.scatter(s.near_horizon,s.far_horizon,s=8,alpha=.3); lo=min(s.far_horizon.min(),s.near_horizon.min()); hi=max(s.far_horizon.max(),s.near_horizon.max()); plt.plot([lo,hi],[lo,hi],'--',linewidth=1); plt.xlabel('Near-access horizon (min)'); plt.ylabel('Far-access horizon (min)'); plt.title('Identical hazard forecast, different preventability'); plt.tight_layout(); plt.savefig(fdir/'matched_risk_pairs.pdf'); plt.savefig(fdir/'matched_risk_pairs.png',dpi=220); plt.close()
    sample=oa.sample(2500,random_state=2); plt.figure(figsize=(6.4,4.6)); plt.scatter(sample.delay/sample.deadline,sample.tcvoi,s=8,alpha=.3); plt.axhline(0,linestyle='--',linewidth=1); plt.axvline(1,linestyle=':',linewidth=1); plt.xlabel('Observation delay / intervention deadline'); plt.ylabel('Time-costed value of information'); plt.title('Observe-or-act: information can arrive too late'); plt.tight_layout(); plt.savefig(fdir/'tcvoi.pdf'); plt.savefig(fdir/'tcvoi.png',dpi=220); plt.close()
    plt.figure(figsize=(6.4,4.6)); plt.plot(sens.epsilon,sens.preventable_harm_recall,marker='o',label='Preventable recall'); plt.plot(sens.epsilon,sens.operational_lateness_rate,marker='o',label='Operational lateness'); plt.xlabel('Risk tolerance epsilon'); plt.ylabel('Rate'); plt.ylim(0,1); plt.legend(); plt.title('Sensitivity to required safety confidence'); plt.tight_layout(); plt.savefig(fdir/'epsilon_sensitivity.pdf'); plt.savefig(fdir/'epsilon_sensitivity.png',dpi=220); plt.close()
    plt.figure(figsize=(6.4,4.6)); plt.hist(col.collective_gap,bins=40); plt.xlabel('Earliest individual horizon - collective horizon (min)'); plt.ylabel('Scenarios'); plt.title('Shared resources can shorten the collective horizon'); plt.tight_layout(); plt.savefig(fdir/'collective_gap.pdf'); plt.savefig(fdir/'collective_gap.png',dpi=220); plt.close()
    pop=pd.DataFrame(DISTRICTS,columns=['district','population_2011']).sort_values('population_2011'); plt.figure(figsize=(7,7)); plt.barh(pop.district,pop.population_2011/1e6); plt.xlabel('Population (millions, Census 2011)'); plt.title('Jharkhand district population anchor'); plt.tight_layout(); plt.savefig(fdir/'jharkhand_population.pdf'); plt.savefig(fdir/'jharkhand_population.png',dpi=220); plt.close()
    print(met.to_string(index=False)); print('\nTCVoI negative rate',oa.negative_tcvoi.mean()); print('Collective theorem pass rate',col.dominance_holds.mean()); print('\nSensitivity\n',sens.to_string(index=False)); print('\nAblation\n',abl.to_string(index=False))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='results'); p.add_argument('--n',type=int,default=50000); a=p.parse_args(); write_outputs(Path(a.out),a.n)
if __name__=='__main__': main()
