# BP2004 velocity crops: modification and scope notice

Data courtesy of BP. Acknowledge BP and Frederic Billette. Reference: Billette and Brandsberg-Dahl (2005), The 2004 BP velocity benchmark, 67th EAGE Conference, B035, doi:10.3997/2214-4609-pdb.1.B035.

The complete original BP documentation and disclaimer are preserved in original_BP2004_READMES.pdf. These files inherit its terms; no new data license is asserted.

Modifications: decode the registered big-endian SEG-Y samples; retain the three predeclared 192-by-192 windows as little-endian float32 NPY files; additionally retain raw[::2,::1] as 96-by-192 NPY files at 12.5 m isotropic spacing. Velocity stays in m/s. No interpolation, clipping, normalization or value-driven window selection occurs. Exact zero-based parent coordinates, hashes and all source headers are in receipt.json.

All six arrays belong to ONE BP2004 composite synthetic numerical ancestor. They are named stress crops, not independent parents/sites, measured field data, a reproduction of the original shot gathers or an evaluated algorithm result.

Ancestor: BP2004_Billette_BrandsbergDahl_composite_exact_velocity
