#=
Original code citations: 
https://github.com/SciML/DiffEqFlux.jl#optimizing-parameters-of-an-ode-for-an-optimal-control-problem
https://julialang.org/blog/2019/01/fluxdiffeq/

Code adapted from Joyce Luo: https://github.com/joyceluo1/mip_opioid

Fits model parameters using neural ODEs for a set of US states.
=#

using CSV, DifferentialEquations, Lux, Optim, Optimization, OptimizationOptimJL, DiffEqFlux, Plots, CSV, DataFrames, DelimitedFiles, Sundials, Tables

tstart = 0.0
tend = 12.0
sampling = 1

model_params = [sqrt(0.03), sqrt(0.09), sqrt(0.09), sqrt(0.05), sqrt(0.01159), sqrt(0.05), sqrt(0.05), sqrt(0.05)]

# take square root of all the numbers and then square them in the model
mu_sr = sqrt(0.1)

function model(du, u, p, t)
    #unsheltered, emergency shelter, transitional housing, safe havens, exits 
    U, E, T, S, E_1 = u
    
    alpha_sr, delta_sr, gamma_sr, sigma_sr, phi_sr, epsilon_sr, beta_sr, zeta_sr= p
 
    du[1] = (alpha_sr^2)*U - (delta_sr^2)*U - (gamma_sr^2)*U - (sigma_sr^2)*U + (mu_sr^2)*E_1
    du[2] = (delta_sr^2)*U - (phi_sr^2)*E
    du[3] = (gamma_sr^2)*U - (epsilon_sr^2)*T
    du[4] = (beta_sr^2)*U - (zeta_sr^2)*S
    du[5] = (phi_sr^2)*E + (epsilon_sr^2)*T + (zeta_sr^2)*S - (mu_sr^2)*E_1

end

states = ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE","FL", "GA", "HI", "IA","IL", "IN", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV",
"NY", "OH", "OK", "OR", "PA", "PR", "RI","SD", "TN", "TX", "UT", "VA", "WA", "WI", "WV", "WY"]
# Larger maxiters error: "ID", "VT"
# Long eval time: "KS"
# instability error: SC
global cat = Array{Float64}(undef, 8, 0)
for i in 1:length(states)
    #data = readdlm("ODE Data/info_" * states[i] * ".csv", ',', Float64)
    data = Array(CSV.read(raw"ODE Data/info_" * states[i] * ".csv", DataFrame, types=Float64, header=false))
    #print(data)
    Q = ifelse.(data .> 0, 1, data)
    #print(Q)
    u0 = data[1, :]
    #print(u0)
    
    function predict_adjoint(param) # Our 1-layer neural network
        prob=ODEProblem(model,u0,(tstart,tend), model_params)
        Array(concrete_solve(prob, Tsit5(), u0, param, saveat=tstart:sampling:tend, 
            abstol=1e-9,reltol=1e-9, sensealg = ForwardDiffSensitivity()))
    end
    
    function loss_adjoint(param)
        prediction = predict_adjoint(param)
        prediction_t = prediction'
        loss = sum(abs2, Q .* (prediction_t - data)) 
        loss
    end

    losses = []
    callback(θ,l) = begin
        push!(losses, l)
        # if length(losses)%50==0
        #     println(losses[end])
        # end
        false
    end

    function plotFit(param, u0, st)

        tspan=(tstart,tend)
        sol_fit=solve(ODEProblem(model,u0, tspan, param), Tsit5(), saveat=tstart:sampling:tend)

        tgrid=tstart:sampling:tend
        pl=plot(sol_fit, idxs=[1 2 3 4 5], lw=2, legend=:outertopright, label = ["U" "E" "T" "S" "E_1"])
        scatter!(pl,tgrid, data[:,1], color=:blue, label = "U")
        scatter!(pl,tgrid, data[:,2], color=:orange, label = "E")
        scatter!(pl,tgrid, data[:,3], color=:green, label = "T")
        scatter!(pl,tgrid, data[:,4], color=:pink, label = "S")
        scatter!(pl,tgrid, data[:,5], color=:brown, label = "E_1")
        xlabel!(pl,"Time")
        ylabel!(pl,"Point-in-Time Count")
        savefig(pl, "ODE Figs/odefit_" * st * ".pdf")
        display(pl)
        # return(Array(sol_fit))
    end

    
    function train_model()
        pguess = [sqrt(0.3), sqrt(0.9), sqrt(0.19), sqrt(0.5), sqrt(0.01159), sqrt(0.5), sqrt(0.5), sqrt(0.5)]
    #     println("Losses (every 50 iters):")
    #     println("$(loss_adjoint(pguess)[1])")
        #Train the ODE
        optf = OptimizationFunction((x, p) -> loss_adjoint(x), Optimization.AutoZygote())
        optprob = Optimization.OptimizationProblem(optf, pguess)
        #println("here")
        result_neuralode = Optimization.solve(optprob, ADAM(0.0001), callback = callback, maxiters = 20000)
        #print("here2")
        optprob2 = remake(optprob, u0 = result_neuralode.u)
        result_neuralode2 = Optimization.solve(optprob2, BFGS(initial_stepnorm = 1e-5), callback = callback)
        println("Fitted parameters:")
        println("$((result_neuralode2.minimizer).^2)")
        return(result_neuralode2)
    end
    
    println(states[i])
    res_norm = train_model()
    result = (res_norm.minimizer).^2
    global cat = hcat(cat, result) 
    plotFit(res_norm.minimizer, u0, states[i])
end
CSV.write("ODE_state_parameters.csv", Tables.table(cat), writeheader = true, header = states)