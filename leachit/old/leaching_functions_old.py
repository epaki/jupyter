import numpy as np
import pandas as pd

DEFAULT_MODEL_PARAMS = dict(
	n_tanks=11,
	total_residence_time=30,
	locked_gold_threshold=0.02,
	k1=1.13e-3, k2=4.37e-11,
	a=2.13, b=0.961, c=0.228,
	d_exp=2.93,
	max_rate=0.25
)


# Define Lima & Hodouin kinetic model
def lima_hodouin_dissolution(
	Au_s, Au_sl, CN, O2, d,
	k1=1.13e-3,	k2=4.37e-11,
	a=2.13,	b=0.961,
	c=0.228, d_exp=2.93,
	max_rate=0.25
):
	"""
	Calculate gold dissolution rate using the Lima & Hodouin kinetic model.
	Parameters
	----------
	Au_s : float
		Concentration of soluble gold in mg/kg.
	Au_sl : float	
		Concentration of locked gold in mg/kg.
	CN : float
		Cyanide concentration in mg/L.
	O2 : float
		Oxygen concentration in mg/L.
	d : float
		Particle size in microns.
	k1 : float, optional
		Fitted rate constant k1, by default 1.13e-3.
	k2 : float, optional
		Fitted rate constant k2, by default 4.37e-11.
	a : float, optional
		Fitted exponent for remaining gold, by default 2.13.
	b : float, optional
		Fitted exponent for cyanide, by default 0.961.
	c : float, optional
		Fitted exponent for oxygen, by default 0.228.
	d_exp : float, optional
		Fitted exponent for particle size, by default 2.93.
	max_rate : float, optional
		Maximum leaching rate in mg/kg/h, by default 0.25.
	Returns
	-------
	float
		Dissolution rate in mg/kg/h.
	"""
	
	k = k1 - k2 * d**d_exp
	if k <= 0 or CN <= 0 or O2 <= 0:
		return 0
	delta = max(Au_s - Au_sl, 0)
	rate = k * delta**a * CN**b * O2**c
	return np.clip(rate, 0, max_rate)


def hybrid_lh_dissolution(Au_s, Au_sl, CN, O2, d, t, 
						  f_fast=0.6, k1=1.13e-3,
						  k2=4.37e-11, a=2.13, b=0.961,
						  c=0.228, d_exp=2.93,
						  fast_mult=3.0, slow_mult=0.3,
						  max_rate=0.25
):
	"""
	Calculate hybrid dissolution rate using the Lima & Hodouin kinetic model with fast/slow pathways.
	Parameters
	----------
	Au_s : float
		Concentration of soluble gold in mg/kg.
	Au_sl : float
		Concentration of locked gold in mg/kg.
	CN : float
		Cyanide concentration in mg/L.
	O2 : float
		Oxygen concentration in mg/L.
	d : float
		Particle size in microns.
	t : float
		Residence time in hours.
	f_fast : float, optional
		Fraction of gold leached quickly, by default 0.6.
	k1 : float, optional
		Fitted rate constant k1, by default 1.13e-3.
	k2 : float, optional
		Fitted rate constant k2, by default 4.37e-11.
	a : float, optional
		Fitted exponent for remaining gold, by default 2.13.
	b : float, optional
		Fitted exponent for cyanide, by default 0.961.
	c : float, optional
		Fitted exponent for oxygen, by default 0.228.
	d_exp : float, optional
		Fitted exponent for particle size, by default 2.93.
	fast_mult : float, optional
		Scaling multiplier for fast leaching, by default 3.0.
	slow_mult : float, optional
		Scaling multiplier for slow leaching, by default 0.3.
	max_rate : float, optional
		Maximum leaching rate in mg/kg/h, by default 0.25.

	Returns
	-------
	float
		Total gold recovered in mg/kg.
	"""
	
	delta = max(Au_s - Au_sl, 0)
	k = k1 - k2 * d**d_exp
	if k <= 0 or CN <= 0 or O2 <= 0:
		return 0
	
	base_rate = k * delta**a * CN**b * O2**c
	rate_fast = np.clip(base_rate * fast_mult, 0, max_rate)
	rate_slow = np.clip(base_rate * slow_mult, 0, max_rate)
	
	recovered = (
		f_fast * delta * (1 - np.exp(-rate_fast * t)) +
		(1 - f_fast) * delta * (1 - np.exp(-rate_slow * t))
	)
	return recovered


def simulate_lh_recovery(
	feed_grade, d, O2, CN,
	n_tanks=11,
	total_residence_time=30,
	locked_gold_threshold=0.02,
	k1=1.13e-3, k2=4.37e-11,
	a=2.13, b=0.961, c=0.228, d_exp=2.93,
	max_rate=0.25,
	return_history=False,
	return_debug=False
):
	feed_grade_mgkg = feed_grade * 1000
	au_remain = feed_grade_mgkg * (1 - locked_gold_threshold)
	au_sl = feed_grade_mgkg * locked_gold_threshold
	res_time = total_residence_time / n_tanks

	history = []

	for i in range(n_tanks):
		rate = lima_hodouin_dissolution(au_remain, au_sl, CN, O2, d, k1, k2, a, b, c, d_exp, max_rate)
		delta = min(rate * res_time, au_remain)
		au_remain = max(au_remain - delta, au_sl)

		if return_history:
			pct = (feed_grade_mgkg * (1 - locked_gold_threshold) - au_remain) / (feed_grade_mgkg * (1 - locked_gold_threshold)) * 100
			history.append({
				"tank": i+1, "rate": rate, "delta": delta,
				"au_remaining": au_remain, "recovery_pct": pct
			})
		if return_debug:
			print(f"[Tank {i+1}] Rate={rate:.2f}, Delta={delta:.2f}, Remaining={au_remain:.2f}")

	recovered = feed_grade_mgkg * (1 - locked_gold_threshold) - au_remain
	final_pct = (recovered / (feed_grade_mgkg * (1 - locked_gold_threshold))) * 100

	if return_history:
		return final_pct, history
	return final_pct



def simulate_lh_cn_sensitivity(feed_grade, d, O2, CN_values, **kwargs):
	return [
		simulate_lh_recovery(feed_grade, d, O2, CN, **kwargs)
		for CN in CN_values
	]



def solve_cn_for_target_recovery(
	feed_grade,
	O2,
	d,
	target_recovery,
	cn_bounds=(10, 1000),
	tol=1e-2,
	**kwargs
):
	
	"""
	Solve for the cyanide concentration (ppm) required to reach a target recovery using
	the LH kinetic model.

	Parameters:
	- feed_grade : float, g/t
	- O2 : float, ppm
	- d : float, µm
	- target_recovery : float, % (e.g. 92.5)
	- cn_bounds : tuple, min and max CN ppm to search
	- tol : float, root-finding tolerance
	- kwargs : additional parameters for `simulate_lh_recovery` (e.g. kinetic constants)

	Returns:
	- cn_required : float or np.nan if unsolvable
	"""

	from scipy.optimize import brentq
	from . import simulate_lh_recovery

	def objective(cn):
		recovery = simulate_lh_recovery(feed_grade, d, O2, cn, **kwargs)
		return recovery - target_recovery

	try:
		cn_required = brentq(objective, cn_bounds[0], cn_bounds[1], xtol=tol)
		return cn_required
	except ValueError:
		return np.nan


def hybrid_dissolution(
	d, t, au0,
	A=800, B=1.8,
	f_fast=0.6, fast_mult=4.0, slow_mult=0.6,
	threshold=50
):
	"""
	Hybrid exponential gold leaching model.

	Parameters:
	- d : float, particle size (µm)
	- t : float, residence time (h)
	- au0 : float, initial available gold (g/t)
	- A, B : rate constants
	- f_fast : fraction of gold leached quickly
	- fast_mult, slow_mult : scaling multipliers for fast/slow leaching
	- threshold : µm cutoff between fast and slow pathway

	Returns:
	- au_remaining : float, gold remaining after leach (g/t)
	"""
	k = A * d**(-B)
	if d <= threshold:
		k_fast = k * fast_mult
		k_slow = k * slow_mult
		return f_fast * au0 * np.exp(-k_fast * t) + (1 - f_fast) * au0 * np.exp(-k_slow * t)
	else:
		return au0 * np.exp(-k * t)


def simulate_hybrid_recovery(
	feed_grade,
	d,
	n_tanks=11,
	total_residence_time=30,
	locked_gold_threshold=0.02,
	A=800, B=1.8,
	f_fast=0.6, fast_mult=4.0, slow_mult=0.6,
	threshold=50,
	return_history=False
):
	"""
	Simulate recovery using the hybrid exponential model.

	Parameters:
	- feed_grade : float, g/t
	- d : float, particle size (µm)
	- n_tanks, total_residence_time : circuit configuration
	- locked_gold_threshold : fraction of locked gold
	- return_history : bool, return per-tank recovery if True

	Returns:
	- final_pct : float (or list if return_history)
	"""
	au_remaining = feed_grade * (1 - locked_gold_threshold)
	res_time_per_tank = total_residence_time / n_tanks
	recovery_pct = []

	for _ in range(n_tanks):
		au_remaining = hybrid_dissolution(
			d=d, t=res_time_per_tank, au0=au_remaining,
			A=A, B=B, f_fast=f_fast,
			fast_mult=fast_mult, slow_mult=slow_mult,
			threshold=threshold
		)
		pct = ((feed_grade * (1 - locked_gold_threshold) - au_remaining) /
			   (feed_grade * (1 - locked_gold_threshold))) * 100
		recovery_pct.append(pct)

	if return_history:
		return recovery_pct
	else:
		return recovery_pct[-1]



















def simulate_lh_recovery_fitted(feed_grade, d, O2, CN, n_tanks=11,
								locked_gold_threshold=0.02,
								total_residence_time=30,
								k1_fit=1.13e-03, k2_fit=4.37e-11,
								a_fit=2.13, b_fit=0.96, c_fit=0.23,
								d_exp_fit=2.93, max_dissolution_rate=100,
								debug=False):
	"""
	Simulate leaching recovery using the Lima and Hodouin (2005) fitted model, with
	site calibrated parameters.
	
	Defaults are based on fitted Shanta parameters.
	
	Lima, V. and Hodouin, D. (2005). "A new approach to the simulation of gold leaching
	in cyanide solutions." Minerals Engineering, 18(1), 1-10.
	https://doi.org/10.1016/j.mineng.2004.08.001
	
	Parameters
	----------
	feed_grade : float
		Feed grade in g/t.
	d : float
		Particle size in microns.
	O2 : float
		Oxygen concentration in mg/L.
	CN : float
		Cyanide concentration in mg/L.
	n_tanks : int, optional
		Number of leach tanks, by default 11.
	locked_gold_threshold : float, optional
		Threshold for locked gold, by default 0.02 (2%).
	total_residence_time : float, optional  
		Total residence time in hours, by default 30.
	k1_fit : float, optional
		Fitted rate constant k1, by default 1.13e-03.
	k2_fit : float, optional    
		Fitted rate constant k2, by default 4.37e-11.   
	a_fit : float, optional
		Fitted exponent for remaining gold, by default 2.13.
	b_fit : float, optional
		Fitted exponent for cyanide, by default 0.96.
	c_fit : float, optional
		Fitted exponent for oxygen, by default 0.23.
	d_exp_fit : float, optional
		Fitted exponent for particle size, by default 2.93.
	max_dissolution_rate : float, optional
		Maximum leaching rate in mg/kg/h, by default 100.
	debug : bool, optional
		Boolean to enable debug output, by default False.
		
	Returns
	-------
	float
		Final percentage of gold recovered after leaching.

	"""
	
	feed_grade_mg_per_kg = feed_grade * 1000  # g/t → mg/kg
	au_remain = feed_grade_mg_per_kg * (1 - locked_gold_threshold)
	au_sl = feed_grade_mg_per_kg * locked_gold_threshold
	res_time = total_residence_time / n_tanks

	for _ in range(11):
		k = k1_fit - k2_fit * d**d_exp_fit
		if k <= 0 or CN <= 0 or O2 <= 0:
			rate = 0
		else:
			rate = k * max(au_remain - au_sl, 0)**a_fit * CN**b_fit * O2**c_fit
		rate = np.clip(rate, 0, max_dissolution_rate)
		delta = min(rate * res_time, au_remain)
		au_remain = max(au_remain - delta, au_sl)
		if debug:
			print(f"[Tank {_+1}] rate={rate:.2f}, delta={delta:.2f}, au_remain={au_remain:.2f}")

	recovered = feed_grade_mg_per_kg * (1 - locked_gold_threshold) - au_remain
	final_pct = (recovered / (feed_grade_mg_per_kg * (1 - locked_gold_threshold))) * 100
	return final_pct


def calc_gold_dissolution_rate(
	R_current: float,
	CN_ppm: float,
	O2_ppm: float,
	R_max: float = 0.98,
	k_base: float = 0.02,
	d: float = 20,
	d_ref: float = 75,
	a: float = 0.5,
	b: float = 0.5,
	exponent_d: float = -0.5
	) -> float:

	"""
	Calculate gold dissolution rate using the Lima & Hodouin kinetic model.

	Parameters:
	- R_current: current recovery (0–1)
	- CN_ppm: cyanide concentration in ppm
	- O2_ppm: dissolved oxygen in ppm
	- R_max: maximum achievable recovery (liberated gold fraction)
	- k_base: base rate constant for reference particle size (µm)
	- d: actual particle size (µm)
	- d_ref: reference particle size used to define k_base (µm)
	- a: CN exponent
	- b: O2 exponent
	- exponent_d: particle size exponent

	Returns:
	- ΔR: change in recovery from current state (float)
	
	"""
	# Scale rate constant by particle size
	k = k_base * (d / d_ref) ** exponent_d

	# Compute dissolution rate
	delta_R = k * (CN_ppm ** a) * (O2_ppm ** b) * (R_max - R_current)
	return delta_R


def simulate_recovery_through_tanks(
	CN_ppm: float,
	O2_ppm: float,
	d: float = 20,  							# default particle size in µm
	n_tanks: int = 11,							# number of tanks in leach circuit
	total_residence_time: float = (30 / 11),  	# residence time per tank in hours
	R_max: float = 0.98,
	k_base: float = 0.02,
	d_ref: float = 75,
	a: float = 0.5,
	b: float = 0.5,
	exponent_d: float = -0.5
) -> float:
	
	"""
	Simulate cumulative gold recovery across a leach circuit using the kinetic model.

	Parameters:
	- CN_ppm, O2_ppm: reagent concentrations (assumed constant across tanks)
	- d: particle size in µm
	- n_tanks: number of tanks in the leaching circuit
	- total_residence_time: total retention time across the full circuit (hours)
	- Other parameters: kinetic equation coefficients

	Returns:
	- Final recovery (as a fraction, 0-1)
	"""
	
	R = 0  # initial recovery
	residence_per_tank = total_residence_time / n_tanks

	for _ in range(n_tanks):
		delta_R = calc_gold_dissolution_rate(
			R_current=R,
			CN_ppm=CN_ppm,
			O2_ppm=O2_ppm,
			R_max=R_max,
			k_base=k_base,
			d=d,
			d_ref=d_ref,
			a=a,
			b=b,
			exponent_d=exponent_d
		)
		R += delta_R * residence_per_tank
		R = min(R, R_max)  # enforce physical limit

	return R