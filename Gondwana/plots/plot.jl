using Pkg; Pkg.activate(".")

using SphereUDE
using Serialization
using Plots
using Plots.PlotMeasures
using JLD2
using DataFrames, CSV
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

sphereude_color = purple_wisteria
splines_color = blue_belize
runingmean_color = green_nephritis
marker_color = RGBA(1,1,1,0)

# JLD2.@load "Gondwana/results/results_dict.jld2"
JLD2.@load "Gondwana/results/hyper/results_custom_dict_1.0e6.jld2"

data_directions_sph = cart2sph(results_dict["directions"], radians=false)
fit_directions_sph = cart2sph(results_dict["fit_directions"], radians=false)

data_lat = data_directions_sph[1,:]
fit_lat  = fit_directions_sph[1,:]

data_lon = data_directions_sph[2,:]
fit_lon  = fit_directions_sph[2,:]

α_error = 140 ./ sqrt.(results_dict["kappas"])

# We load the interpolation data from Torsvik et al (2012)
df_Torsvik = CSV.read("Gondwana/data/Torsvik-etal-2012_rm_splines.csv", DataFrame, delim=",")

runing_mean_age = df_Torsvik.Age
runing_mean_lat = df_Torsvik.Plat_RM06
runing_mean_lon = df_Torsvik.Plon_RM06
runing_mean_lon[runing_mean_lon .> 180.0] .-= 360.0

splines_age = df_Torsvik.Age
splines_lat = df_Torsvik.Plat_Spline
splines_lon = df_Torsvik.Plon_Spline
splines_lon[splines_lon .> 180.0] .-= 360.0

# Latitude Plot 

Plots.scatter(
    results_dict["times"], data_lat,
    label="Paleopole", yerr=α_error,
    c=marker_color, ms=5, msw=0.3
    )
plot!(runing_mean_age, runing_mean_lat, label = "Running mean path", lw = 4, c=runingmean_color, linestyle=:dot)
plot!(splines_age, splines_lat, label = "Spline path", lw = 4, c=splines_color, linestyle=:dash)
plot_latitude = Plots.plot!(
    results_dict["fit_times"], fit_lat,
    label="SphereUDE path", 
    xlabel="Age (Myr)", yticks=[-90, -60, -30, 0, 30, 60], xlims=(0,550),
    ylabel="Latitude (degrees)", ylims=(-90,60), lw = 4, c=sphereude_color, 
    legend=:topleft
    )
plot!(
    fontfamily="Computer Modern",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    margin= 7mm,
    size=(1200,500),
    dpi=600
    )

Plots.savefig(plot_latitude, "Gondwana/plots/latitude.pdf")

### Longitude Plot 

α_error_lon = α_error ./ cos.(π ./ 180. .* data_lat)

Plots.scatter(
    results_dict["times"], data_lon,
    label="Paleopole Longitudes", yerr=α_error_lon,
    c=marker_color, markersize=5, msw=0.3
    )
plot!(runing_mean_age, runing_mean_lon, label = "Running mean", lw = 4, c=runingmean_color, linestyle=:dot)
plot!(splines_age, splines_lon, label = "Splines", lw = 4, c=splines_color, linestyle=:dash)
plot_longitude = Plots.plot!(
    results_dict["fit_times"], fit_lon,
    label="SphereUDE", 
    xlabel="Age (Myr)", yticks=[-180, -90 , 0, 90, 180], xlims=(0,550),
    ylabel="Longitude (degrees)", ylims=(-220,180), lw = 5, c=sphereude_color,
    legend=nothing
    )
plot!(
    fontfamily="Computer Modern",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    margin= 7mm,
    size=(1200,500),
    dpi=600
    )

Plots.savefig(plot_longitude, "Gondwana/plots/longitude.pdf")


### Angular velocity Plot

angular_velocity = mapslices(x -> norm(x), results_dict["fit_rotations"], dims=1)[:]
angular_velocity_path = [norm(cross(results_dict["fit_directions"][:,i], results_dict["fit_rotations"][:,i] )) for i in axes(results_dict["fit_rotations"],2)]
angular_velocity_x = (180.0 / π) .* mapslices(x -> x[2], results_dict["fit_rotations"], dims=1)[:]

angular_velocity .*= (180.0 / π)
angular_velocity_path .*= (180.0 / π)

plot_angular = Plots.plot(
    results_dict["fit_times"], angular_velocity, label="Maximum total angular velocity", 
    xlabel="Age (Myr)",
    xlims=(0,550),# ylims=(0.0, 0.041),
    ylabel="Angular velocity (degrees/My)", lw = 5, c=angular_color_scatter,
    legend=:topleft
    )
plot!(results_dict["fit_times"], angular_velocity_path, label="Pole angular velocity", lw = 4, c=angular_color_line, ls=:dot)
# plot!(results_dict["fit_times"], angular_velocity_x, label="X", lw = 2, c=angular_color_line, ls=:dot)

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

combo_plot = Plots.plot(plot_latitude, plot_longitude, plot_angular, layout = (3, 1))
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

plot_loss = Plots.plot(
    1:length(losses), losses, label="Loss Function", 
    xlabel="Epoch", 
    ylabel="Loss", lw = 5, c=loss_color,
    yaxis=:log,
    # yticks=[1,10,100],
    xlimits=(0,10000),
    ylimits=(10, 1000),
    xticks=[0,2500,5000,7500,10000],
    legend=:topright
    )

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

# Plot data in sphere

plt[].figure(figsize = (10, 10))
ax = plt[].axes(
    projection = ccrs[].Orthographic(
        central_latitude = -30.0,
        central_longitude = 0.0,
    ),
)

ax.coastlines()
ax.gridlines()
ax.set_global()

X_true_points = data_directions_sph
X_sphereude_path = fit_directions_sph
X_running_mean_path = [runing_mean_lat runing_mean_lon]'
X_splines_path = [splines_lat splines_lon]'

sns[].scatterplot(
    ax = ax,
    x = X_true_points[2, :],
    y = X_true_points[1, :],
    hue = results_dict["times"],
    alpha = 0.5,
    s = 150,
    palette = "viridis",
    transform = ccrs[].PlateCarree(),
)

X_all = [X_sphereude_path, X_running_mean_path, X_splines_path]
# X_all = [X_running_mean_path]
# X_all = [X_splines_path]
color_all = ["black", "black", "black"]
lw_all = [3.0, 0.5, 0.5]

for j in 1:1
    X = X_all[j]
    for i = 1:(size(X)[2] - 1)
        plt[].plot(
            [X[2, i], X[2, i+1]],
            [X[1, i], X[1, i+1]],
            linewidth = lw_all[j],
            color = color_all[j],
            transform = ccrs[].Geodetic(),
        )
    end
end

plt[].savefig("Gondwana/plots/Gondwana_sphere_comparison.pdf", format = "pdf")