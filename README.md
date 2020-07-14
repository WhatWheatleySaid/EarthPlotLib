# WIP
This project is WIP and is trying to build a genuine way to make good 3D visualisations of Geospatial data.

# Example
Making a Plot is as easy as you know it from matplotlib or similiar plotting libraries:
```
    from earthplot import EarthPlot
    plot = EarthPlot(d_light_strength = (.3,.3,.3,1))
    datasize = 10000
    lons2 = np.linspace(0,40,50)
    lats2 = lons2
    lons = np.concatenate(((np.random.randn(datasize)-.5)*10 , (np.random.randn(datasize)-.5)*10 + np.random.rand(1)*80))
    lats = np.concatenate(((np.random.randn(datasize)-.5)*10 , (np.random.randn(datasize)-.5)*10 + np.random.rand(1)*80))
    plot.plot_heatmap(lons,lats, n_bins = 200, texture_filter = 'linear', alpha = .7, mpl_cm = 'jet')
    plot.add_coastline(res = '50m', color =[0,0,0], linewidth = 2, shading = 'None')
    plot.plot_surface_lines(lons2,lats2, linewidth = 3, color = [1,1,1])
    plot.show()
   ```
