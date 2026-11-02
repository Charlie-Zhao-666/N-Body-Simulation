"""Analytic isolated two-body ellipse, apocenter start, in the program's code units.

The relative semi-major axis is 1; both masses and G are 1. Eccentricity is
configurable. Positions/velocities are barycentric, with the first body initially
on +x and moving along +y. No numerical integration is used for the reference.
"""
import numpy as np

DEFAULT_ECCENTRICITY = .99
BINARY_PERIOD = 2*np.pi/np.sqrt(2.)
POSITION_SAMPLES = 20001


def validate_eccentricity(value):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError
        value = float(value)
        if not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise ValueError('Binary eccentricity e must satisfy 0 <= e < 1.') from None
    return value


def eccentric_anomaly(mean_anomaly, eccentricity):
    """Solve E-e sin(E)=M on [-pi,pi] by safeguarded monotone bisection."""
    e=validate_eccentricity(eccentricity)
    mean=np.asarray(mean_anomaly,dtype=float)
    if not np.isfinite(mean).all():
        raise ValueError('Kepler mean anomaly must be finite.')
    mean=np.remainder(mean+np.pi,2*np.pi)-np.pi
    lo=np.full_like(mean,-np.pi);hi=np.full_like(mean,np.pi)
    # A fixed iteration count removes Newton convergence failures near e=1.
    for _ in range(64):
        mid=(lo+hi)/2
        # Stable E-sin(E) near zero, where direct subtraction loses digits.
        x2=mid*mid
        difference=np.where(np.abs(mid)<.1,
            mid*x2*(1/6-x2*(1/120-x2*(1/5040-x2*(1/362880-x2/39916800)))),
            mid-np.sin(mid))
        residual=(1-e)*mid+e*difference-mean
        lo=np.where(residual<0,mid,lo)
        hi=np.where(residual>0,mid,hi)
        exact=residual==0
        lo=np.where(exact,mid,lo);hi=np.where(exact,mid,hi)
    return (lo+hi)/2


def binary_state(times, eccentricity=DEFAULT_ECCENTRICITY):
    """Return (...,2,3) barycentric positions and velocities at physical times."""
    e=validate_eccentricity(eccentricity);t=np.asarray(times,dtype=float)
    if not np.isfinite(t).all():
        raise ValueError('Kepler reference times must be finite.')
    n=np.sqrt(2.)
    # t=0 is apocenter: M=pi. Rotate the standard ellipse by pi.
    E=eccentric_anomaly(n*t+np.pi,e)
    sine=np.sin(E);cosine=np.cos(E);minor=np.sqrt((1-e)*(1+e))
    relative=np.stack((-((1-e)-2*np.sin(E/2)**2),-minor*sine,np.zeros_like(E)),axis=-1)
    rate=n/((1-e)+2*e*np.sin(E/2)**2)
    relative_velocity=np.stack((sine*rate,-minor*cosine*rate,np.zeros_like(E)),axis=-1)
    positions=np.stack((relative/2,-relative/2),axis=-2)
    velocities=np.stack((relative_velocity/2,-relative_velocity/2),axis=-2)
    # Enforce the exact specified apocenter initial state, not sin(pi) roundoff.
    initial_pos=np.array([[(1+e)/2,0.,0.],[-(1+e)/2,0.,0.]])
    speed=np.sqrt(2.*(1-e)/(1+e))/2
    initial_vel=np.array([[0.,speed,0.],[0.,-speed,0.]])
    positions=np.where((t==0)[...,None,None],initial_pos,positions)
    velocities=np.where((t==0)[...,None,None],initial_vel,velocities)
    return positions,velocities


def binary_stars(eccentricity=DEFAULT_ECCENTRICITY):
    positions,velocities=binary_state(0.,eccentricity)
    return [dict(mass=1.,pos=p.copy(),v=v.copy()) for p,v in zip(positions,velocities)]


def sample_positions(times, positions, velocities, grid):
    """Cubic Hermite position interpolation; shapes [time,star,xyz]; no extrapolation."""
    times=np.asarray(times,dtype=float);positions=np.asarray(positions,dtype=float)
    velocities=np.asarray(velocities,dtype=float);grid=np.asarray(grid,dtype=float)
    if (times.ndim!=1 or len(times)<2 or positions.shape!=velocities.shape or positions.ndim!=3
        or positions.shape[0]!=len(times) or positions.shape[-1]!=3 or grid.ndim!=1 or not len(grid)
        or not all(np.isfinite(a).all() for a in (times,positions,velocities,grid))
        or np.any(np.diff(times)<=0) or np.any(np.diff(grid)<0)):
        raise ValueError('Invalid trajectory or comparison grid.')
    if grid[0]<times[0] or grid[-1]>times[-1]:
        raise ValueError('Comparison times must lie inside the saved trajectory.')
    right=np.clip(np.searchsorted(times,grid,side='right'),1,len(times)-1);left=right-1
    width=(times[right]-times[left])[:,None,None]
    u=((grid-times[left])/(times[right]-times[left]))[:,None,None]
    aligned=((2*u**3-3*u**2+1)*positions[left]+(u**3-2*u**2+u)*width*velocities[left]
        +(-2*u**3+3*u**2)*positions[right]+(u**3-u**2)*width*velocities[right])
    nodes=np.searchsorted(times,grid);matched=times[np.minimum(nodes,len(times)-1)]==grid
    aligned[matched]=positions[nodes[matched]]
    return aligned


def binary_position_error(times, trajectories, velocities, eccentricity, samples=POSITION_SAMPLES):
    """RMS 3D distance per star on a uniform physical-time grid, including endpoints."""
    e=validate_eccentricity(eccentricity)
    grid=np.linspace(times[0],times[-1],int(samples))
    actual=sample_positions(times,np.asarray(trajectories).transpose(1,0,2),velocities,grid)
    expected,_=binary_state(grid,e)
    squared=np.sum((actual-expected)**2,axis=-1)
    per_star=np.sqrt(np.mean(squared,axis=0));rms=float(np.sqrt(np.mean(squared)))
    length_scale=(1+e)/2
    return dict(reference_kind='analytic_kepler',rmse=rms,nrmse=rms/length_scale,
        per_star_rmse=per_star.tolist(),samples=len(grid),time_start=float(grid[0]),time_end=float(grid[-1]),
        eccentricity=e,semi_major_axis_relative=1.,masses=[1.,1.],G=1.,period=float(BINARY_PERIOD),
        initial_phase='apocenter; barycenter at rest; first star on +x with velocity +y',
        metric='sqrt(mean over stars and uniform physical-time samples of squared 3D position distance)); endpoints included',
        interpolation='cubic Hermite using numerical positions and velocities; exact saved nodes retained',
        normalization='initial RMS radius about barycenter = (1+e)/2',units='code length',
        reference_solver='Kepler equation E-e*sin(E)=M; 64 monotone bisections; analytic positions, not an integrated reference')


_initial=binary_stars()
KEPLER_BINARY_ORBIT=dict(name='High-eccentricity Kepler binary (adjustable e)',N=2,
    masses=[1.,1.],positions=[s['pos'].tolist() for s in _initial],velocities=[s['v'].tolist() for s in _initial],
    period=float(BINARY_PERIOD),dt=float(BINARY_PERIOD/40000),tot_time=40000,kepler_binary=True,
    source='Newtonian two-body ellipse: a_relative=1, G=1, m1=m2=1; analytic Kepler reference',
    note='Adjust Binary eccentricity e (0 <= e < 1). Fixed relative semi-major axis 1; apocenter start. Position RMSE is computed automatically against the analytic Kepler orbit.')
