// Newtonian 3D Taylor integrator. Python owns the timestep controller.
// All physical state, coefficient recurrences and energy statistics remain in MPFR.
#include <mpfr.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#define NB_EXPORT extern "C" __declspec(dllexport)
#else
#define NB_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr auto rnd = MPFR_RNDN;
using Cancel = int (*)();
using Version = std::uint64_t;
thread_local char last_error[512];
std::atomic<Version> next_owner{1};

class Cancelled : public std::runtime_error { using std::runtime_error::runtime_error; };
class FailedTrial : public std::runtime_error { using std::runtime_error::runtime_error; };

template<class Action>
int guarded(Action action) noexcept {
    try {
        action();
        return 0;
    } catch (const Cancelled& e) {
        std::snprintf(last_error, sizeof(last_error), "%s", e.what());
        return 2;
    } catch (const FailedTrial& e) {
        std::snprintf(last_error, sizeof(last_error), "%s", e.what());
        return 3;
    } catch (const std::exception& e) {
        std::snprintf(last_error, sizeof(last_error), "%s", e.what());
    } catch (...) {
        std::snprintf(last_error, sizeof(last_error), "Unknown MPFR backend error");
    }
    return 1;
}

class Real {
public:
    mpfr_t value;
    explicit Real(int bits) { mpfr_init2(value, bits); mpfr_set_zero(value, 1); }
    Real(const Real& other) { mpfr_init2(value, mpfr_get_prec(other.value)); mpfr_set(value, other, rnd); }
    Real& operator=(const Real& other) { mpfr_set(value, other, rnd); return *this; }
    ~Real() { mpfr_clear(value); }
    operator mpfr_ptr() { return value; }
    operator mpfr_srcptr() const { return value; }
    void parse(const char* input) {
        char* end = nullptr;
        if (!input) throw std::invalid_argument("Missing MPFR initial value");
        mpfr_strtofr(value, input, &end, 0, rnd);
        if (end == input || *end || !mpfr_number_p(value))
            throw std::invalid_argument("Invalid or non-finite MPFR initial value");
    }
};
using Series = std::vector<Real>;
void zero(Real& x) { mpfr_set_zero(x, 1); }
void accumulate_product(Real& sum, const Real& a, const Real& b, Real& tmp) {
    mpfr_mul(tmp, a, b, rnd);
    mpfr_add(sum, sum, tmp, rnd);
}
size_t product(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::length_error("Taylor workspace exceeds addressable memory");
    return a*b;
}
void check_cancel(Cancel fn) {
    if (fn && fn()) throw Cancelled("Taylor MPFR calculation cancelled");
}

struct State {
    Series pos, vel;
    Real time, energy_error, energy_integral;
    Version generation = 0;
    State(size_t axes, int bits)
        : pos(axes, Real(bits)), vel(axes, Real(bits)), time(bits),
          energy_error(bits), energy_integral(bits) {}
};

class Orbit {
public:
    const Version owner = next_owner++;
    const int n, order, bits;
    const size_t axes;
    const int stride;
    State state;
    Real gravity, initial_energy, energy_scale;
    Series mass;
    Version builds = 0;

    Orbit(int bodies, int p, int precision, const char* g, const char* const* m,
          const char* const* pos, const char* const* vel)
        : n(bodies), order(p), bits(precision), axes(product(3,n)), stride(order+2),
          state(axes,bits), gravity(bits), initial_energy(bits), energy_scale(bits),
          mass(n,Real(bits)), coeff(product(axes,stride),Real(bits)),
          accel(axes,Real(bits)), new_pos(axes,Real(bits)), new_vel(axes,Real(bits)) {
        gravity.parse(g);
        if (mpfr_sgn(gravity.value) <= 0) throw std::invalid_argument("Gravity must be positive");
        for (int i=0; i<n; ++i) {
            mass[i].parse(m[i]);
            if (mpfr_sgn(mass[i].value)<=0) throw std::invalid_argument("Each mass must be positive");
        }
        for (size_t a=0; a<axes; ++a) {
            state.pos[a].parse(pos[a]);
            state.vel[a].parse(vel[a]);
        }
        energy(state.pos,state.vel,initial_energy,&energy_scale,nullptr);
        const size_t pair_count = product(static_cast<size_t>(n),n-1)/2;
        pairs.assign(product(pair_count,5*order), Real(bits));
    }

    void advance(double step, Cancel cancel) {
        if (!std::isfinite(step) || step<=0)
            throw std::invalid_argument("Taylor step must be positive and finite");
        check_cancel(cancel);
        // A rejected proposal can restore this same expansion origin and try a
        // different h. Horner evaluation changes with h; the series does not.
        if (!coeff_valid || coeff_generation!=state.generation) {
            coeff_valid = false;
            make_coefficients(cancel);
            coeff_generation = state.generation;
            coeff_valid = true;
            ++builds;
        }
        Real h(bits), tmp(bits), tmp2(bits);
        mpfr_set_d(h,step,rnd);
        for (size_t a=0; a<axes; ++a) {
            if ((a%256)==0) check_cancel(cancel);
            const Real* r = &coeff[a*stride];
            Real& x = new_pos[a]; Real& v = new_vel[a];
            x = r[order];
            mpfr_mul_ui(v,r[order+1],order+1,rnd);
            for (int k=order-1; k>=0; --k) {
                mpfr_mul(tmp,x,h,rnd); mpfr_add(x,tmp,r[k],rnd);
                mpfr_mul(tmp,v,h,rnd); mpfr_mul_ui(tmp2,r[k+1],k+1,rnd);
                mpfr_add(v,tmp,tmp2,rnd);
            }
            if (!mpfr_number_p(x) || !mpfr_number_p(v))
                throw FailedTrial("Taylor MPFR: non-finite trial state; reduce dt");
        }
        Real error(bits), integral(bits), time(bits);
        energy(new_pos,new_vel,error,nullptr,cancel);
        mpfr_sub(error,error,initial_energy,rnd);
        // Exact squared integral of the linear interpolant between energy-error nodes.
        mpfr_sqr(tmp,state.energy_error,rnd);
        accumulate_product(tmp,state.energy_error,error,tmp2);
        accumulate_product(tmp,error,error,tmp2);
        mpfr_mul(tmp2,tmp,h,rnd); mpfr_div_ui(tmp,tmp2,3,rnd);
        mpfr_add(integral,state.energy_integral,tmp,rnd);
        mpfr_add(time,state.time,h,rnd);
        if (!mpfr_number_p(integral) || !mpfr_number_p(time))
            throw FailedTrial("Taylor MPFR: non-finite trial diagnostics; reduce dt");
        check_cancel(cancel);
        state.pos.swap(new_pos); state.vel.swap(new_vel);
        state.time = time; state.energy_error = error; state.energy_integral = integral;
        state.generation = ++generation;
    }

    void text(int field, int idx, char* output, int capacity) const {
        if (!output || capacity < 1) throw std::invalid_argument("Invalid MPFR output buffer");
        Real scalar(bits);
        if (field==0 || field==1) {
            if (idx<0 || static_cast<size_t>(idx)>=axes) throw std::out_of_range("Invalid coordinate");
            scalar = (field==0 ? state.pos : state.vel)[idx];
        } else {
            if (idx!=0) throw std::out_of_range("Invalid scalar index");
            switch (field) {
                case 2: scalar=state.time; break;
                case 3: scalar=initial_energy; break;
                case 4: scalar=state.energy_error; break;
                case 5: case 6:
                    if (!mpfr_zero_p(state.time)) {
                        mpfr_div(scalar,state.energy_integral,state.time,rnd);
                        mpfr_sqrt(scalar,scalar,rnd);
                    }
                    if (field==6) {
                        if (mpfr_zero_p(energy_scale)) mpfr_set_nan(scalar);
                        else mpfr_div(scalar,scalar,energy_scale,rnd);
                    }
                    break;
                case 7: energy(state.pos,state.vel,scalar,nullptr,nullptr); break;
                default: throw std::invalid_argument("Unknown MPFR report field");
            }
        }
        const int digits = static_cast<int>(std::ceil(bits*0.3010299956639812))+2;
        const int length = mpfr_snprintf(output,capacity,"%.*RNe",digits,scalar.value);
        if (length<0 || length>=capacity) throw std::length_error("MPFR output buffer too small");
    }
private:
    Series coeff, accel, new_pos, new_vel, pairs;
    bool coeff_valid = false;
    Version coeff_generation=0, generation=0;

    void make_coefficients(Cancel cancel) {
        Real tmp(bits), tmp2(bits), sum(bits);
        for (size_t a=0; a<axes; ++a) {
            coeff[a*stride] = state.pos[a]; coeff[a*stride+1] = state.vel[a];
        }
        for (int k=0; k<order; ++k) {
            check_cancel(cancel);
            for (Real& a : accel) zero(a);
            size_t pair_index=0;
            for (int i=0; i<n; ++i) for (int j=i+1; j<n; ++j,++pair_index) {
                if ((pair_index%256)==255) check_cancel(cancel);
                Real* delta=&pairs[pair_index*(5*order)];
                Real* q=delta+3*order; Real* b=q+order;
                // For this pair, delta = x_j - x_i, q = delta . delta,
                // and b = q^(-3/2). Each array contains power-series coefficients.
                zero(q[k]);
                for (int d=0; d<3; ++d) {
                    Real* axis=delta+d*order;
                    const size_t a=static_cast<size_t>(3)*i+d, a2=static_cast<size_t>(3)*j+d;
                    mpfr_sub(axis[k],coeff[a2*stride+k],coeff[a*stride+k],rnd);
                    for (int l=0; l<=k; ++l) accumulate_product(q[k],axis[l],axis[k-l],tmp);
                }
                if (k==0) {
                    if (mpfr_zero_p(q[0]) || !mpfr_number_p(q[0]))
                        throw std::runtime_error("Coincident or out-of-range particles at Taylor expansion origin");
                    mpfr_sqrt(tmp,q[0],rnd); mpfr_mul(tmp,q[0],tmp,rnd);
                    mpfr_ui_div(b[0],1,tmp,rnd);
                } else {
                    zero(sum);
                    // Equate coefficients in 2*q*b' = -3*q'*b to obtain b[k].
                    for (int l=1; l<=k; ++l) {
                        mpfr_mul_ui(tmp2,q[l],l+2*k,rnd);
                        accumulate_product(sum,tmp2,b[k-l],tmp);
                    }
                    mpfr_neg(sum,sum,rnd); mpfr_mul_ui(tmp,q[0],2*k,rnd);
                    mpfr_div(b[k],sum,tmp,rnd);
                }
                // Both particles use the same delta*b series with opposite sign
                // and the mass of the companion. Divide by (k+1)(k+2) below
                // to integrate the acceleration coefficient twice.
                for (int d=0; d<3; ++d) {
                    zero(sum);
                    const Real* axis=delta+d*order;
                    for (int l=0; l<=k; ++l) accumulate_product(sum,axis[l],b[k-l],tmp);
                    mpfr_mul(sum,sum,gravity,rnd);
                    Real& ai=accel[static_cast<size_t>(3)*i+d];
                    Real& aj=accel[static_cast<size_t>(3)*j+d];
                    accumulate_product(ai,mass[j],sum,tmp);
                    mpfr_mul(tmp,mass[i],sum,rnd); mpfr_sub(aj,aj,tmp,rnd);
                }
            }
            for (size_t a=0; a<axes; ++a) {
                Real& c=coeff[a*stride+k+2];
                mpfr_div_ui(c,accel[a],(k+1)*(k+2),rnd);
                if (!mpfr_number_p(c)) throw std::runtime_error("Non-finite Taylor coefficient at expansion origin");
            }
        }
    }

    void energy(const Series& x, const Series& v, Real& result, Real* scale, Cancel cancel) const {
        Real kinetic(bits), potential(bits), sum(bits), tmp(bits), delta(bits);
        size_t comparisons=0;
        for (int i=0; i<n; ++i) {
            size_t a=static_cast<size_t>(3)*i;
            zero(sum);
            for (int d=0; d<3; ++d) accumulate_product(sum,v[a+d],v[a+d],tmp);
            mpfr_mul(sum,mass[i],sum,rnd); mpfr_div_ui(sum,sum,2,rnd);
            mpfr_add(kinetic,kinetic,sum,rnd);
            for (int j=i+1; j<n; ++j) {
                if ((++comparisons%256)==0) check_cancel(cancel);
                zero(sum);
                for (int d=0; d<3; ++d) {
                    mpfr_sub(delta,x[static_cast<size_t>(3)*j+d],x[a+d],rnd);
                    accumulate_product(sum,delta,delta,tmp);
                }
                if (mpfr_zero_p(sum)) throw FailedTrial("Coincident point masses; Newtonian gravity is singular");
                mpfr_sqrt(sum,sum,rnd); mpfr_mul(tmp,gravity,mass[i],rnd);
                mpfr_mul(tmp,tmp,mass[j],rnd); mpfr_div(tmp,tmp,sum,rnd);
                mpfr_sub(potential,potential,tmp,rnd);
            }
        }
        mpfr_add(result,kinetic,potential,rnd);
        if (!mpfr_number_p(result)) throw FailedTrial("Taylor MPFR: non-finite energy");
        if (scale) mpfr_sub(*scale,kinetic,potential,rnd);
    }
};

struct Snapshot {
    const Version owner;
    State state;
    explicit Snapshot(const Orbit& orbit) : owner(orbit.owner), state(orbit.state) {}
};
Orbit& orbit(void* handle) {
    if (!handle) throw std::invalid_argument("Missing MPFR integrator handle");
    return *static_cast<Orbit*>(handle);
}
} // namespace

NB_EXPORT int nb_abi() noexcept { return 1; }
NB_EXPORT const char* nb_version() noexcept { return mpfr_get_version(); }
NB_EXPORT const char* nb_error() noexcept { return last_error; }
NB_EXPORT void* nb_create(int n,int order,int bits,const char* g,const char* const* mass,
                         const char* const* pos,const char* const* vel) noexcept {
    Orbit* handle=nullptr;
    guarded([&]{
        if (n<1 || bits<2 || bits>8192 || !(order==45 || order==100 || order==150 || order==200))
            throw std::invalid_argument("Invalid Taylor shape, order or working precision");
        if (!mass || !pos || !vel) throw std::invalid_argument("Missing Taylor initial data");
        handle=new Orbit(n,order,bits,g,mass,pos,vel);
    });
    return handle;
}
NB_EXPORT void nb_destroy(void* h) noexcept { delete static_cast<Orbit*>(h); }
NB_EXPORT void* nb_checkpoint(void* h) noexcept {
    Snapshot* point=nullptr;
    guarded([&]{ point=new Snapshot(orbit(h)); });
    return point;
}
NB_EXPORT void nb_free_checkpoint(void* p) noexcept { delete static_cast<Snapshot*>(p); }
NB_EXPORT int nb_restore(void* h,void* p) noexcept {
    return guarded([&]{
        auto& o=orbit(h); auto* point=static_cast<Snapshot*>(p);
        if (!point || point->owner!=o.owner) throw std::invalid_argument("Invalid checkpoint for this Taylor integrator");
        o.state = point->state;
    });
}
NB_EXPORT int nb_advance(void* h,double dt,Cancel cancel) noexcept {
    return guarded([&]{ orbit(h).advance(dt,cancel); });
}
NB_EXPORT unsigned long long nb_coefficient_builds(void* h) noexcept {
    return h ? static_cast<Orbit*>(h)->builds : 0;
}
NB_EXPORT int nb_state(void* h,double* pos,double* vel) noexcept {
    return guarded([&]{
        auto& o=orbit(h);
        if (!pos || !vel) throw std::invalid_argument("Missing coordinate output buffer");
        for (size_t a=0; a<o.axes; ++a) {
            pos[a]=mpfr_get_d(o.state.pos[a],rnd); vel[a]=mpfr_get_d(o.state.vel[a],rnd);
        }
    });
}
NB_EXPORT int nb_text(void* h,int field,int idx,char* dest,int capacity) noexcept {
    return guarded([&]{ orbit(h).text(field,idx,dest,capacity); });
}
