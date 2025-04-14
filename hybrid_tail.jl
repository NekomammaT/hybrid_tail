using DifferentialEquations, LinearAlgebra, ForwardDiff, Plots, Statistics, LaTeXStrings, Base.Threads, DelimitedFiles;
@show Threads.nthreads()


sample = 10^5;

# n = 2
ny = 2;
filename = "N=2_test.csv";


Mpl = 2.4e18;
M = 1e16 / Mpl;
phic = sqrt(2)M;
mu2 = 10;

Pi2 = 100;
mu1 = Pi2/M^2/phic;
Lambda = 5.4e15 / Mpl * (10/sqrt(Pi2)) * M * sqrt(phic);
psir = sqrt(Lambda^4 * sqrt(Pi2) / 48 / sqrt(2*π^3));


function V(x,y::Vector) 
    return Lambda^4 * ((1-dot(y,y)/M^2)^2 + 2*x^2*dot(y,y) / M^2 / phic^2 + (x-phic)/mu1 - (x-phic)^2/mu2^2)
end;

function Vx(x,y::Vector)
    Vinx(x) = V(x,y)
    return ForwardDiff.derivative(Vinx,x)
end;

function Vy(x,y::Vector)
    Viny(y) = V(x,y)
    return ForwardDiff.gradient(Viny,y)
end;

function Vr(x,yr)
    return Lambda^4 * ((1-yr^2/M^2)^2 + 2*x^2*yr^2 / M^2 / phic^2 + (x-phic)/mu1 - (x-phic)^2/mu2^2)
end;

function etar(x,yr)
    Vrx(yr) = Vr(x,yr)
    Vp = yr -> ForwardDiff.derivative(Vrx, yr)
    Vpp = ForwardDiff.derivative(Vp, yr)
    return Vpp / Vr(x,yr)
end;

function H(x,px,y::Vector,py::Vector)
    return sqrt((px^2/2 + dot(py,py)/2 + V(x,y))/3)
end;

function drift(du,u,p,t) # u[1] = x, u[2] = px, u[3:2+ny] = y, u[3+ny:2+2ny] = py
    ny = Int((length(u)-2)/2)
    x = u[1]
    px = u[2]
    y = u[3:2+ny]
    py = u[3+ny:2+2ny]
    Hubble = H(x,px,y,py)
    
    du[1] = px/Hubble
    du[2] = -3px-Vx(x,y)/Hubble

    for i in 1:ny
        du[2+i] = py[i]/Hubble
        du[2+ny+i] = -3py[i]-Vy(x,y)[i]/Hubble
    end
end;

function diffusion(du,u,p,t)
    ny = Int((length(u)-2)/2)
    x = u[1]
    px = u[2]
    y = u[3:2+ny]
    py = u[3+ny:2+2ny]
    Hubble = H(x,px,y,py)

    du[1] = Hubble/2/π
    du[2] = 0

    for i in 1:ny
        du[2+i] = Hubble/2/π
        du[2+ny+i] = 0
    end
end;

function SRvio(u,t,integrator)
    ny = Int((length(u)-2)/2)
    x = u[1]
    y = u[3:2+ny]
    yr = sqrt(dot(y,y))
    return etar(x,yr) < -2
end;

y0 = [psir for i in 1:ny];
py0 = [0 for i in 1:ny];
u0 = [[phic + 15/mu1, 0]; y0; py0];
Nspan = (0,100);
prob = SDEProblem(drift, diffusion, u0, Nspan, 0);

affect!(integrator) = terminate!(integrator)
cb = DiscreteCallback(SRvio, affect!)

dNlist = zeros(sample);
@time Threads.@threads for i in 1:sample
    dNlist[i] = solve(prob,reltol=1e-10,abstol=1e-10,callback=cb,save_everystep=false).t[end]
end;

writedlm(filename,dNlist);
