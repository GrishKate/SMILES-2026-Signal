# Solution

## Reproducibility

Needs scipy, numpy, gdown.

Run:

```
python applicant_solution.py
```

## Solution description

The provided signal is

```
rx = s + F_c( TX )  +  E + eta
```
Estimation of ```F_c( TX )``` is already good, so the goal is to find rank-1 component E.


Estimating rank-1 from residual ```rx - F_c( TX )``` results in score 9.69, while estimating it from residual convolved with ```score_bp``` filter (as in ```score_filter``` function) results in score 7.01. This is surprising, but it is due to filter introducing phase shift and rotation. I found out that rotating the rank-1 component back results in much higher score, although this does not remove the phase shift. By rotation I mean multiplying or dividing by a constant ```np.exp(2j * np.pi * CENTER * (len(score_bp) - 1) // 2 / Fs)```. However, this still worked worse than estimating rank-1 directly from residual.
 
In addition, different channels have different magnitude, so normalization of residual before rank-1 component computation improved the score.

Repeating the procedure of finding ```F_c( TX )``` and computing rank-1 several times also slightly improved the score.

The final solutionis as follows: 

```
e = 0
for i in range(2):
  pred_fc = helpers["fit_tx_prediction"](rx - e) # finds fc as in baseline
  resid = rx - pred_fc
  e = rank1_from_band_matrix(resid) # this function normalizes before estimation of rank-1
result = rx - pred_fc - e
```


| Method | Score |
|-----|-----|
| Baseline | 4.02 |
| Subtracting rank-1 of denoised residual | 7.01 |
| Subtracting rank-1 of residual | 9.69 |
| Rotating back rank-1 of denoised residual | 9.34 |
| Normalization of channels before rank-1 computation | 9.85 |
| Alternating rank-1 subtraction and F_c computation 2 times | 9.88 |


## What did not work

I tried to find filters, that would introduce no phase shift. Applying scipy.signal.filtfilt (forward and backward filtering) instead of convolution did not work. In addition, scipy.signal.filtfilt works only with real filters.

I also tried fft and ifft to remove noise, but provided bandpass filter worked much better.

Although the channels in rx have different magnitude, normalization of rx directly did worse.
