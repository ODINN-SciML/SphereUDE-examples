using Pkg; Pkg.activate(".")

using SphereUDE
using Serialization
using Plots 
using Plots.PlotMeasures
using JLD2
using LinearAlgebra

JLD2.@load "double-rotation/results/results_dict_1000.jld2" 

data_directions_sph = cart2sph(results_dict["directions"], radians=false)
fit_directions_sph = cart2sph(results_dict["fit_directions"], radians=false)

data_lat = data_directions_sph[1,:]
fit_lat  = fit_directions_sph[1,:]

data_lon = data_directions_sph[2,:]
fit_lon  = fit_directions_sph[2,:]

blue_belize = RGBA(41/255, 128/255, 185/255, 1)
blue_midnigth = RGBA(44/255, 62/255, 80/255, 1)
purple_wisteria = RGBA(142/255, 68/255, 173/255, 1)
red_pomegrade = RGBA(192/255, 57/255, 43/255, 1)
orange_carrot = RGBA(230/255, 126/255, 34/255, 1)
green_nephritis = RGBA(39/255, 174/255, 96/255, 1)
green_sea = RGBA(22/255, 160/255, 133/255, 1)
# colors 
lat_color_scatter = blue_belize
lat_color_line = purple_wisteria
lon_color_scatter = red_pomegrade
lon_color_line = orange_carrot
angular_color_scatter = green_nephritis
angular_color_line = green_sea
loss_color = blue_midnigth


### Angular velocity Plot

angular_velocity = mapslices(x -> norm(x), results_dict["fit_rotations"], dims=1)[:]
angular_velocity_true = 10.0 * π / 180.0

plot_angular = Plots.plot(results_dict["fit_times"], angular_velocity, label="Estimated angular velocity", 
                    xlabel="Age (Myr)", 
                    ylabel="Angular velocity", lw = 5, c=angular_color_scatter,
                    legend=:topright)
hline!([angular_velocity_true], label="Reference angular velocity", lw = 4, c=angular_color_line, ls=:dot)
plot!(fontfamily="Computer Modern",
    #title="PIL51",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    ylimits=(0.15,0.201),
    xlimits=(0,100),
    margin= 10mm,
    size=(1200,500),
    dpi=300)


Plots.savefig(plot_angular, "double-rotation/plots/angular.pdf")

