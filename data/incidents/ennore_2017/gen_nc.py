import netCDF4 as nc
import numpy as np
from datetime import datetime

def generate_nc():
    filename = r'c:\Users\prave\Desktop\Oil Spill\data\incidents\ennore_2017\ocean_currents.nc'
    dataset = nc.Dataset(filename, 'w', format='NETCDF4')
    
    dataset.createDimension('time', 49)
    dataset.createDimension('lat', 20)
    dataset.createDimension('lon', 20)
    
    times = dataset.createVariable('time', np.float64, ('time',))
    lats = dataset.createVariable('latitude', np.float32, ('lat',))
    lons = dataset.createVariable('longitude', np.float32, ('lon',))
    
    uo = dataset.createVariable('uo', np.float32, ('time', 'lat', 'lon'))
    vo = dataset.createVariable('vo', np.float32, ('time', 'lat', 'lon'))
    
    times.units = 'hours since 2017-01-27 12:00:00'
    lats.units = 'degrees_north'
    lons.units = 'degrees_east'
    uo.units = 'm/s'
    vo.units = 'm/s'
    
    times[:] = np.arange(49)
    lats[:] = np.linspace(12.5, 14.0, 20)
    lons[:] = np.linspace(79.5, 81.0, 20)
    
    np.random.seed(42)
    base_uo = np.random.uniform(0.05, 0.25, (49, 20, 20))
    base_vo = np.random.uniform(-0.3, -0.05, (49, 20, 20))
    
    uo[:] = base_uo
    vo[:] = base_vo
    
    dataset.close()

if __name__ == '__main__':
    generate_nc()
