using Pkg; Pkg.activate(".")

using SphereUDE
using Serialization
using Plots 
using Plots.PlotMeasures
using JLD2
using LinearAlgebra

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

JLD2.@load "Gondwana/results/results_dict.jld2" 

data_directions_sph = cart2sph(results_dict["directions"], radians=false)
fit_directions_sph = cart2sph(results_dict["fit_directions"], radians=false)

data_lat = data_directions_sph[1,:]
fit_lat  = fit_directions_sph[1,:]

data_lon = data_directions_sph[2,:]
fit_lon  = fit_directions_sph[2,:]

α_error = 140 ./ sqrt.(results_dict["kappas"])

# Latitude Plot 

Plots.scatter(results_dict["times"], data_lat, label="Paleopole Latitudes", yerr=α_error, c=lat_color_scatter, ms=5, msw=0.3)
plot_latitude = Plots.plot!(results_dict["fit_times"], fit_lat, label="Estimated APWP using SphereUDE", 
                    xlabel="Age (Myr)", yticks=[-90, -60, -30, 0, 30, 60], xlims=(0,520),
                    ylabel="Latitude (degrees)", ylims=(-90,60), lw = 4, c=lat_color_line, 
                    legend=:topleft)
plot!(fontfamily="Computer Modern",
     #title="PIL51",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    #ylimits=(0.1,10),
    #xlimits=(10^(-4),10^(-1)),
    legend=true,
    margin= 7mm,
    size=(1200,500),
    dpi=600)

Plots.savefig(plot_latitude, "Gondwana/plots/latitude.pdf")

### Longitude Plot 

α_error_lon = α_error ./ cos.(π ./ 180. .* data_lat)

Plots.scatter(results_dict["times"], data_lon, label="Paleopole Longitudes", yerr=α_error_lon, c=lon_color_scatter, markersize=5, msw=0.3)
plot_longitude = Plots.plot!(results_dict["fit_times"], fit_lon, label="Estimated APWP using SphereUDE", 
                    xlabel="Age (Myr)", yticks=[-180, -90 , 0, 90, 180], xlims=(0,520),
                    ylabel="Longitude (degrees)", ylims=(-220,180), lw = 4, c=lon_color_line,
                    legend=:bottomright)
plot!(fontfamily="Computer Modern",
     #title="PIL51",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    #ylimits=(0.1,10),
    #xlimits=(10^(-4),10^(-1)),
    margin= 7mm,
    size=(1200,500),
    dpi=600)

Plots.savefig(plot_longitude, "Gondwana/plots/longitude.pdf")


### Angular velocity Plot

angular_velocity = mapslices(x -> norm(x), results_dict["fit_rotations"], dims=1)[:]
angular_velocity_path = [norm(cross(results_dict["fit_directions"][:,i], results_dict["fit_rotations"][:,i] )) for i in axes(results_dict["fit_rotations"],2)]

plot_angular = Plots.plot(results_dict["fit_times"], angular_velocity, label="Maximum total angular velocity", 
                    xlabel="Age (Myr)", yticks=[0.0, 0.01, 0.02, 0.03, 0.04], xlims=(0,520), ylims=(0.0, 0.041),
                    ylabel="Angular velocity (degrees/My)", lw = 5, c=angular_color_scatter,
                    legend=:topleft)
plot!(results_dict["fit_times"], angular_velocity_path, label="Pole angular velocity", lw = 4, c=angular_color_line, ls=:dot)
plot!(fontfamily="Computer Modern",
    #title="PIL51",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    #ylimits=(0.1,10),
    #xlimits=(10^(-4),10^(-1)),
    margin= 7mm,
    size=(1200,500),
    dpi=600)


Plots.savefig(plot_angular, "Gondwana/plots/angular.pdf")

### Lat and long combined

combo_plot = plot(plot_latitude, plot_longitude, plot_angular, layout = (3, 1))
plot!(fontfamily="Computer Modern",
     #title="PIL51",
    # titlefontsize=18,
    # tickfontsize=15,
    # legendfontsize=15,
    # guidefontsize=18,
    #ylimits=(0.1,10),
    #xlimits=(10^(-4),10^(-1)),
    # margin= 7mm,
    size=(1200,1000))
Plots.savefig(combo_plot, "Gondwana/plots/latitude_longitude.pdf")

### Loss function

losses = results_dict["losses"]

plot_loss = Plots.plot(1:length(losses), losses, label="Loss Function", 
                    xlabel="Epoch", 
                    ylabel="Loss", lw = 5, c=loss_color,
                    yaxis=:log,
                    # yticks=[1,10,100],
                    xlimits=(0,10000),
                    ylimits=(10, 1000),
                    xticks=[0,2500,5000,7500,10000],
                    legend=:topright)

vspan!(plot_loss, [0,5000], color = :navajowhite3, alpha = 0.2, labels = "ADAM");
vspan!(plot_loss, [5000,10000], color = :navajowhite4, alpha = 0.2, labels = "BFGS");

plot!(fontfamily="Computer Modern",
    #title="PIL51",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    #ylimits=(0.1,10),
    #xlimits=(10^(-4),10^(-1)),
    margin= 10mm,
    size=(700,500),
    dpi=300)

Plots.savefig(plot_loss, "Gondwana/plots/loss.pdf")
