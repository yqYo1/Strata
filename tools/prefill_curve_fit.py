"""Owned descriptive timing fits with a prospectively independent length.

Main-authored,repository LICENSE. Reuse the maintained bench client's OLS
models;the reserved point never influences coefficients or model selection.
"""
import math
import statistics

from prefill_scaling_bench import approximation_models

TRAIN_READS=(16385,32769,49153,65537,81921)
HELD_OUT_READ=57345
CHANNELS=('first_prompt_ms','warmed_prompt_ms','first_TTFT_ms','warmed_TTFT_ms')
SELECTION=('Minimum whole-length LOO mean relative error;ties within1e-12 prefer '
    'affine,quadratic,n_log_n. For warmed channels prefer affine when both its '
    'mean and maximum LOO errors are within the maximum training half-range/'
    'median of the best model. The56Kpoint is reserved for validation only.')


def require(value,message):
    if not value:raise ValueError(message)


def positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value>0


def predict(name,coefficients,read):
    x=read/65536
    value=coefficients['a']+coefficients['b']*x
    if name=='quadratic':value+=coefficients['c']*x*x
    elif name=='n_log_n':value+=coefficients['c']*x*math.log2(x)
    else:require(name=='affine','unknown model')
    require(math.isfinite(value),'nonfinite prediction')
    return value


def characterize(points):
    """One first use and two subsequent samples per independent length process."""
    require(len(points)==6 and all(type(p['read']) is int for p in points)
        and sorted(p['read'] for p in points)==sorted((*TRAIN_READS,HELD_OUT_READ)),
        'exact five training lengths and independent56Krequired')
    rows={}
    for point in points:
        first=point['first_use'];prompt=point['subsequent_prompt_ms'];ttft=point['subsequent_TTFT_s']
        require(len(prompt)==len(ttft)==2 and all(positive(x) for x in (*prompt,*ttft,
            first['prompt_ms'],first['ttft_s'])),'finite positive first/two warmed timings required')
        rows[point['read']]=dict(first_prompt_ms=first['prompt_ms'],
            warmed_prompt_ms=statistics.median(prompt),first_TTFT_ms=first['ttft_s']*1000,
            warmed_TTFT_ms=statistics.median(ttft)*1000,
            warmed_prompt_samples=prompt,warmed_TTFT_samples=[x*1000 for x in ttft])
    result={}
    for channel in CHANNELS:
        data=[dict(actual_read_tokens=n,prompt_ms=rows[n][channel]) for n in TRAIN_READS]
        models=approximation_models(data)['candidates']
        require(set(models)=={'affine','quadratic','n_log_n'},'all declared hypotheses required')
        rank={'affine':0,'quadratic':1,'n_log_n':2}
        best_error=min(v['mean_held_out_relative_error'] for v in models.values())
        best=min((k for k,v in models.items() if v['mean_held_out_relative_error']<=best_error+1e-12),
            key=rank.get)
        envelope=None;selected=best
        if channel.startswith('warmed_'):
            samples='warmed_prompt_samples' if channel=='warmed_prompt_ms' else 'warmed_TTFT_samples'
            envelope=max((max(rows[n][samples])-min(rows[n][samples]))/(2*rows[n][channel])
                for n in TRAIN_READS)
            affine,base=models['affine'],models[best]
            if (affine['mean_held_out_relative_error']<=base['mean_held_out_relative_error']+envelope+1e-12
                    and affine['max_held_out_relative_error']<=base['max_held_out_relative_error']+envelope+1e-12):
                selected='affine'
        tolerance=0 if envelope is None else envelope
        near=[k for k,v in models.items()
            if v['mean_held_out_relative_error']<=models[best]['mean_held_out_relative_error']+tolerance+1e-12
            and v['max_held_out_relative_error']<=models[best]['max_held_out_relative_error']+tolerance+1e-12]
        observed=rows[HELD_OUT_READ][channel]
        for name,model in models.items():
            predicted=predict(name,model['coefficients_ms'],HELD_OUT_READ)
            model['independent56K']=dict(predicted_ms=predicted,observed_ms=observed,
                error_ms=predicted-observed,relative_error=abs(predicted-observed)/observed,
                positive_prediction=predicted>0)
        result[channel]=dict(candidates=models,selected_model=selected,minimum_LOO_model=best,
            training_relative_half_range_envelope=envelope,
            candidates_within_training_envelope=near,empirical_spread_available=envelope is not None,
            training_adjacent_ms_per_token=[(rows[b][channel]-rows[a][channel])/(b-a)
                for a,b in zip(TRAIN_READS,TRAIN_READS[1:])],
            held_out_used_for_fit_or_selection=False)
    return dict(training_read_tokens=TRAIN_READS,independent_validation_read_tokens=HELD_OUT_READ,
        selection_rule=SELECTION,channels=result,physical_fixed_cost_identified=False,
        hardware_saturation_qualified=False,
        limits='One independent process per length,two warmed samples;the half-range is empirical,not a confidence bound. '
            'A selected descriptive fit is not a physical law. Keep all residuals and held-out failures;do not extrapolate '
            'outside16K..80K,pool other routes,claim a paired speedup,or qualify a throughput target from this curve.')
